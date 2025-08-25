from rqalpha.api import *
import pandas as pd
import os
import numpy as np
import rqdatac as rq
from rqalpha import run_func

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
        "log_level": "error"
    },
    "mod": {
        "sys_analyser": {"enabled": True, "plot": True, "output_file": r"纯多头_优化.pkl"}
    }
}

def _precompute_top_decile_lists(df: pd.DataFrame) -> dict:
    """
    按照“每个交易日非停牌、非ST、close非None的股票总数”来动态确定Top10%股票，
    返回 {date_str: [order_book_id, ...]} 字典。
    """


    # 类型标准化
    df['order_book_id'] = df['order_book_id'].astype(str)
    if not np.issubdtype(df['date'].dtype, np.dtype('O')):
        df['date'] = df['date'].astype(str)

    # 过滤：非停牌、非ST、close非None
    if 'suspended' in df.columns:
        df = df[df['suspended'] == False]
    if 'ST' in df.columns:
        df = df[df['ST'] == False]
    if 'close' in df.columns:
        df = df[df['close'].notnull()]

    # 按日期分组，动态取每组Top10%
    result = {}
    for date, group in df.groupby('date'):
        # 排除无因子值
        group = group[group['market_cap'].notnull()]
        group = group[group['close'].notnull()]
        n = len(group)
        if n == 0:
            result[date] = []
            continue
        top_n = 50
        # 按因子降序取前top_n
        top_stocks = group.sort_values('market_cap', ascending=True).head(top_n)['order_book_id'].tolist()
        result[date] = top_stocks

    return result

def init(context):
    # 周一开盘调仓（与原脚本一致）
    scheduler.run_daily(rebalance)

    # （一次性）加载因子数据，过滤不可交易标的 —— 与原脚本逻辑一致
    print("加载因子数据并预处理（一次性）...")
    df = pd.read_pickle(
        r"C:\Users\9shao\Desktop\github公开项目\Multi-Factor-Strategy-Development-Framework\测试代码\因子数据\multivariate_rolling_120_复合因子_长数据.pkl"
    )

    # 预处理：这些过滤在原脚本中已做，这里保留（集中到 init 里，调仓不再重复）
    # 若输入已过滤也没关系，这里是幂等的

    # 仅保留必要列，减小对象体积
    cols = ['date', 'order_book_id', 'market_cap','close','ST','suspended','limit_up_flag','limit_down_flag']
    df = df[[c for c in cols if c in df.columns]].dropna(subset=['market_cap'])

    # —— 关键加速：一次性预计算每日 Top10% 股票清单 ——
    context.top10_dict = _precompute_top_decile_lists(df)

    # 最近一次下达的目标集合，用于“无变化则跳过下单”
    context._last_targets = None

    print(f"预计算完成，共 {len(context.top10_dict)} 个交易日可用目标池。")

def handle_bar(context, bar_dict):

    pass

def rebalance(context, bar_dict):
    # 用日期字符串做索引（与原数据一致）
    cur = context.now.strftime("%Y-%m-%d")

    target_list = context.top10_dict.get(cur)
    if not target_list:
        return  # 当天无可交易目标，直接跳过

    # 若本次目标与上次相同，直接跳过下单（消除重复 API 调用）
    if context._last_targets == tuple(target_list):
        return
    # 使用 order_target_portfolio 方法，一次性传入股票和等权重
    weight = 1.0 / len(target_list)
    weights = {stock: weight for stock in target_list}
    order_target_portfolio(weights)

    # 记录这次目标
    context._last_targets = tuple(target_list)

if __name__ == "__main__":
    run_func(init=init, handle_bar=handle_bar, config=config_0)
