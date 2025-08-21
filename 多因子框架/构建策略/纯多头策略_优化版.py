from rqalpha.api import *
import pandas as pd
import numpy as np
import rqdatac as rq
from rqalpha import run_func
import gc

# 保持你的 rq.init（License 与环境一致）
rq.init('license', 'B1T4WrPGQ0YBin6JPZm_DlLj3JGxAiuGzi9-SuUNqOUce6MrZ7yLejN2O9OWDPBJ3U6cVO-6uaK8Wn29JTxgNRHrqWJgGHTtf483vOI3bFPOMknL3dEAgQZJTLHJ7LyMZcalsTdqlVOyhT0mlNm_9iNEZBgTxhQW0X_DHOzxLBg=efJUVSAH7ub2Gg33v_nzcsj25LD0caxMdEYz93JazlzGzbk5tG6kxZKqGx9LcGX58GmZmd2IxNETyXC1jdWnnH7u97TeyhpkfP1lfJ5h5sp-1vSqT0tXvJH0SOGoIY4spxMnMeEprvKck7cwl1As3xh8y068HH8MApBtIrP7aOA=')

config_0 = {
    "base": {
        "start_date": "2010-01-01",
        "end_date": "2025-07-22",
        "stock_commission_multiplier": 0.125,
        "frequency": "1d",
        "accounts": {"stock": 10000000},
        "benchmark": "000300.XSHG"
    },
    "extra": {
        "log_level": "warning"
    },
    "mod": {
        "sys_analyser": {"enabled": True, "plot": True, "output_file": r"纯多头_优化版.pkl"}
    }
}

def _precompute_top_decile_lists_optimized(df: pd.DataFrame) -> dict:
    """
    优化版本：使用向量化操作和更高效的数据结构
    """
    # 1. 提前过滤无效数据，减少后续处理量
    mask = (
        df['factor'].notnull() & 
        df['close'].notnull() &
        (~df['suspended'] if 'suspended' in df.columns else True) &
        (~df['ST'] if 'ST' in df.columns else True)
    )
    df_filtered = df[mask].copy()
    
    # 2. 类型转换优化 - 只转换必要的列
    df_filtered['order_book_id'] = df_filtered['order_book_id'].astype('string')
    df_filtered['date'] = df_filtered['date'].astype('string')
    
    # 3. 使用更高效的分组和排序方法
    result = {}
    
    # 按日期分组，使用numpy的快速排序
    for date, group in df_filtered.groupby('date', sort=False):
        if len(group) == 0:
            result[date] = []
            continue
            
        # 计算top 10%的数量
        n = len(group)
        top_n = max(1, int(np.ceil(n * 0.10)))
        
        # 使用numpy的argsort，比pandas sort_values快
        factor_values = group['factor'].values
        order_book_ids = group['order_book_id'].values
        
        # 获取top_n的索引（降序）
        top_indices = np.argpartition(factor_values, -top_n)[-top_n:]
        
        # 如果top_n > 1，需要排序
        if top_n > 1:
            top_indices = top_indices[np.argsort(factor_values[top_indices])[::-1]]
        
        result[date] = order_book_ids[top_indices].tolist()
    
    return result

def _precompute_top_decile_lists_vectorized(df: pd.DataFrame) -> dict:
    """
    完全向量化版本：使用pandas的高级功能
    """
    # 1. 一次性过滤所有无效数据
    mask = (
        df['factor'].notnull() & 
        df['close'].notnull() &
        (~df['suspended'] if 'suspended' in df.columns else True) &
        (~df['ST'] if 'ST' in df.columns else True)
    )
    df_filtered = df[mask].copy()
    
    # 2. 类型转换
    df_filtered['order_book_id'] = df_filtered['order_book_id'].astype('string')
    df_filtered['date'] = df_filtered['date'].astype('string')
    
    # 3. 使用pandas的groupby + apply，向量化处理
    def get_top_stocks(group):
        n = len(group)
        if n == 0:
            return []
        top_n = max(1, int(np.ceil(n * 0.10)))
        return group.nlargest(top_n, 'factor')['order_book_id'].tolist()
    
    result = df_filtered.groupby('date', sort=False).apply(get_top_stocks).to_dict()
    
    return result

def init(context):
    # 周一开盘调仓
    scheduler.run_monthly(rebalance, tradingday=1)

    print("加载因子数据并预处理（优化版本）...")
    
    # 1. 只读取必要的列，减少内存使用
    cols = ['date', 'order_book_id', 'factor', 'close', 'ST', 'suspended', 'limit_up_flag', 'limit_down_flag']
    
    # 2. 使用更高效的数据加载方式
    df = pd.read_pickle(
        r"C:\Users\9shao\Desktop\github公开项目\Multi-Factor-Strategy-Development-Framework\测试代码\因子数据\multivariate_rolling_120_复合因子_长数据.pkl"
    )
    
    # 3. 只保留需要的列
    available_cols = [c for c in cols if c in df.columns]
    df = df[available_cols].copy()
    
    # 4. 提前过滤无效数据
    df = df.dropna(subset=['factor'])
    
    print(f"原始数据大小: {len(df)} 行")
    
    # 5. 选择优化方法（可以根据数据大小选择）
    if len(df) > 1000000:  # 大数据集使用向量化版本
        print("使用向量化优化版本...")
        context.top10_dict = _precompute_top_decile_lists_vectorized(df)
    else:  # 小数据集使用混合优化版本
        print("使用混合优化版本...")
        context.top10_dict = _precompute_top_decile_lists_optimized(df)
    
    # 6. 释放大对象内存
    del df
    gc.collect()
    
    # 7. 最近一次下达的目标集合，用于"无变化则跳过下单"
    context._last_targets = None
    
    print(f"预计算完成，共 {len(context.top10_dict)} 个交易日可用目标池。")
    
    # 8. 预计算权重，避免重复计算
    context._weight_cache = {}

def handle_bar(context, bar_dict):
    pass

def rebalance(context, bar_dict):
    # 用日期字符串做索引
    cur = context.now.strftime("%Y-%m-%d")

    target_list = context.top10_dict.get(cur)
    if not target_list:
        return  # 当天无可交易目标，直接跳过

    # 若本次目标与上次相同，直接跳过下单
    if context._last_targets == tuple(target_list):
        return
    
    # 使用缓存的权重计算
    target_key = tuple(target_list)
    if target_key not in context._weight_cache:
        weight = 1.0 / len(target_list)
        weights = {stock: weight for stock in target_list}
        context._weight_cache[target_key] = weights
    else:
        weights = context._weight_cache[target_key]
    
    # 下单
    order_target_portfolio(weights)
    
    # 记录这次目标
    context._last_targets = target_key
    
    # 清理权重缓存，避免内存泄漏
    if len(context._weight_cache) > 100:
        context._weight_cache.clear()

if __name__ == "__main__":
    run_func(init=init, handle_bar=handle_bar, config=config_0)
