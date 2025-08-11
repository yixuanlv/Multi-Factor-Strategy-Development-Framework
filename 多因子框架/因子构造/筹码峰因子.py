
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from tqdm import tqdm
import rqdatac as rq
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# 初始化米筐接口
rq.init('license', 'I6eb8ljE6tv9DWcYa3F0hERxbDdQ3f0RZzgBEnTHuaQlVX56azMDhdh6dYB42IJ7onLu0mAl3A1rRGFVTQuxE4jZcwoByZaySYlNuInciyFGarrHTz24mblwqbrC4RaCvKbkxP-tZ9S7ZjDY8pTNWu4uIslVXYb4XXL9NSwGI58=T7bU8OlDqvS3R5pPNN7s3PsfirJTCFSHPXpm5Ak3n0Dpzaze0NLbHWfZ-JcnlTBz7Oxec6dmkH9X4UB0OT0qxlkHA3pX_muOZI_zgMpCNZFH1wZ-DjeEMXrkqGGBKIo6_rZeaz130Fo1PLRY-rTw71gPmhD1oVg7GVh1kC7SWrk=') 

# 筹码峰因子参数
CHIP_WINDOW = 250  # 筹码留存窗口，进一步精简为30日滚动用于测试
START_DATE = '2015-01-01'  # 缩小时间范围以加快测试
END_DATE = '2025-07-31'

def get_stock_data(start_date, end_date, max_stocks=None):
    """获取股票的历史数据（可选限制股票数量，max_stocks为None时取全部）"""
    print("正在获取股票列表...")
    
    # 获取A股列表
    all_stocks = rq.all_instruments(type='CS', date=None)['order_book_id'].tolist()
    if max_stocks is None:
        test_stocks = all_stocks
    else:
        test_stocks = all_stocks[:max_stocks]
    
    print(f"选择{len(test_stocks)}只股票进行测试")
    print(f"测试股票: {test_stocks[:5]}...")  # 显示前5只股票
    
    print("正在获取股票历史数据...")
    
    # 获取OHLCV数据
    ohlcv_data = rq.get_price(test_stocks, start_date=start_date, end_date=end_date, 
                              frequency='1d', fields=['open', 'high', 'low', 'close', 'volume'],
                              adjust_type='post', skip_suspended=False, market='cn', expect_df=True)
    
    # 获取换手率数据
    turnover_data = rq.get_turnover_rate(test_stocks, start_date=start_date, end_date=end_date, expect_df=True)
    
    # 获取VWAP数据
    vwap_data = rq.get_vwap(test_stocks, start_date=start_date, end_date=end_date, frequency='1d')
    
    print(f"OHLCV数据形状: {ohlcv_data.shape}")
    print(f"换手率数据形状: {turnover_data.shape}")
    print(f"VWAP数据形状: {vwap_data.shape}")
    
    # 重置索引
    ohlcv_data = ohlcv_data.reset_index()
    turnover_data = turnover_data.reset_index()
    vwap_data = vwap_data.reset_index()
    
    print(f"重置索引后OHLCV数据列名: {ohlcv_data.columns.tolist()}")
    print(f"重置索引后换手率数据列名: {turnover_data.columns.tolist()}")
    print(f"重置索引后VWAP数据列名: {vwap_data.columns.tolist()}")
    
    # 检查实际数据结构并重命名列
    print(f"OHLCV数据实际列数: {len(ohlcv_data.columns)}")
    print(f"换手率数据实际列数: {len(turnover_data.columns)}")
    print(f"VWAP数据实际列数: {len(vwap_data.columns)}")
    
    # 根据实际列数重命名
    if len(ohlcv_data.columns) == 7:
        ohlcv_data.columns = ['order_book_id', 'date', 'open', 'high', 'low', 'close', 'volume']
    else:
        print(f"警告：OHLCV数据列数不匹配，实际列数: {len(ohlcv_data.columns)}")
        ohlcv_data.columns = [f'col_{i}' for i in range(len(ohlcv_data.columns))]
    
    if len(turnover_data.columns) == 7:
        turnover_data.columns = ['order_book_id', 'tradedate', 'today', 'week', 'month', 'year', 'current_year']
        # 选择换手率列（通常是today列）
        turnover_data = turnover_data[['order_book_id', 'tradedate', 'today']]
        turnover_data.columns = ['order_book_id', 'date', 'turnover_rate']
    else:
        print(f"警告：换手率数据列数不匹配，实际列数: {len(turnover_data.columns)}")
        turnover_data.columns = [f'col_{i}' for i in range(len(turnover_data.columns))]
    
    if len(vwap_data.columns) == 3:
        vwap_data.columns = ['order_book_id', 'date', 'vwap']
    else:
        print(f"警告：VWAP数据列数不匹配，实际列数: {len(vwap_data.columns)}")
        vwap_data.columns = [f'col_{i}' for i in range(len(vwap_data.columns))]
    
    # 合并数据
    data = pd.merge(ohlcv_data, turnover_data, on=['date', 'order_book_id'], how='inner')
    data = pd.merge(data, vwap_data, on=['date', 'order_book_id'], how='inner')
    
    # 按日期和股票代码排序
    data = data.sort_values(['date', 'order_book_id']).reset_index(drop=True)
    
    # 处理缺失值
    data = data.dropna()
    
    print(f"数据获取完成，共{len(data)}条记录")
    print(f"数据时间范围: {data['date'].min()} 到 {data['date'].max()}")
    print(f"股票数量: {data['order_book_id'].nunique()}")
    
    # 显示数据样本
    print("\n数据样本：")
    print(data.head())
    
    # 检查数据长度问题
    print("\n检查数据长度问题：")
    for stock in data['order_book_id'].unique()[:5]:  # 只检查前5只股票
        stock_data = data[data['order_book_id'] == stock]
        print(f"股票 {stock}: {len(stock_data)} 条记录")
        if len(stock_data) > 0:
            print(f"  日期范围: {stock_data['date'].min()} 到 {stock_data['date'].max()}")
    
    return data

def calculate_single_stock_chips(args):
    """单只股票的筹码留存计算 - 高度向量化版本"""
    stock, stock_data, window = args
    
    if len(stock_data) < window:
        return None
    
    n_days = len(stock_data)
    
    # 获取数据，处理缺失值
    turnover_rates = stock_data['turnover_rate'].fillna(0.01).values
    volumes = stock_data['volume'].values
    vwaps = stock_data['vwap'].values
    
    # 预计算换手率衰减矩阵 - 使用更高效的算法
    # 使用累积乘积的向量化操作
    turnover_matrix = np.ones((n_days, n_days))
    
    # 使用向量化操作填充下三角矩阵
    for i in range(1, n_days):
        # 计算从0到i-1的累积换手率衰减
        if i > 0:
            # 使用cumprod的向量化操作
            decay_factors = np.cumprod(1 - turnover_rates[:i])[::-1]  # 反转顺序
            turnover_matrix[i, :i] = decay_factors
    
    # 计算筹码留存矩阵 - 完全向量化
    chip_retention = np.zeros((n_days, window))
    historical_vwap = np.zeros((n_days, window))
    
    # 当天数据
    chip_retention[:, 0] = volumes
    historical_vwap[:, 0] = vwaps
    
    # 历史数据 - 使用向量化操作
    for k in range(1, min(window, n_days)):
        # 创建索引数组
        valid_indices = np.arange(k, n_days)
        if len(valid_indices) > 0:
            # 获取历史数据
            hist_volumes = volumes[valid_indices - k]
            hist_vwaps = vwaps[valid_indices - k]
            
            # 获取对应的换手率衰减 - 向量化索引
            decay_values = turnover_matrix[valid_indices, valid_indices - k]
            
            # 计算筹码留存
            chip_retention[valid_indices, k] = hist_volumes * decay_values
            historical_vwap[valid_indices, k] = hist_vwaps
    
    # 转换为DataFrame
    chip_df = pd.DataFrame(chip_retention, columns=[f'chip_retention_{i}' for i in range(window)])
    vwap_df = pd.DataFrame(historical_vwap, columns=[f'historical_vwap_{i}' for i in range(window)])
    
    chip_df['date'] = stock_data['date'].values
    chip_df['order_book_id'] = stock
    chip_df['vwap'] = stock_data['vwap'].values
    
    # 合并筹码留存和历史VWAP
    result_df = pd.concat([chip_df, vwap_df], axis=1)
    
    return result_df

def calculate_chip_retention_optimized(df, window=CHIP_WINDOW):
    """优化的筹码留存计算 - 多核并行+完全向量化版本
    按照理论：RSDAmt(T,T-k) = Amt(T-k)*cumprod(1-Turnover(i))(T,T-k+1); k = (0,250]
    使用多核并行和完全向量化操作大幅提高效率
    """
    print("正在计算筹码留存（多核并行+向量化版本）...")
    
    # 准备并行计算参数
    stocks = df['order_book_id'].unique()
    print(f"使用 {cpu_count()} 个CPU核心进行并行计算...")
    
    # 准备数据
    stock_data_list = []
    for stock in stocks:
        stock_data = df[df['order_book_id'] == stock].copy()
        stock_data = stock_data.sort_values('date').reset_index(drop=True)
        stock_data_list.append((stock, stock_data, window))
    
    # 使用多核并行计算
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(calculate_single_stock_chips, stock_data_list),
            total=len(stock_data_list),
            desc="并行计算筹码留存"
        ))
    
    # 过滤掉None结果
    results = [r for r in results if r is not None]
    
    if not results:
        raise ValueError("没有足够的股票数据计算筹码留存")
    
    # 合并所有股票的结果
    final_result = pd.concat(results, ignore_index=True)
    print("筹码留存计算完成")
    return final_result

def calculate_holding_return_optimized(df, chip_df, window=CHIP_WINDOW):
    """优化的筹码收益因子计算 - 超高效向量化版本
    按照理论：holding_ret = sum(vwap(t)*RSDAmt(T,t))/sum(RSDAmt(T,t))
    使用超高效向量化操作提高效率
    """
    print("正在计算筹码收益因子（超高效向量化版本）...")
    
    # 合并数据
    merged_df = pd.merge(df, chip_df, on=['date', 'order_book_id'], how='inner')
    
    # 预提取筹码留存和VWAP列名
    chip_cols = [f'chip_retention_{i}' for i in range(window)]
    vwap_cols = [f'historical_vwap_{i}' for i in range(window)]
    
    # 获取筹码留存和VWAP数据矩阵 - 使用更高效的数据类型
    chip_matrix = merged_df[chip_cols].astype(np.float32).values
    vwap_matrix = merged_df[vwap_cols].astype(np.float32).values
    
    # 超高效向量化计算筹码收益因子
    # 使用numpy的向量化操作，避免Python循环
    
    # 创建掩码，过滤掉无效数据
    valid_mask = (chip_matrix > 0) & np.isfinite(vwap_matrix)
    
    # 使用numpy的einsum进行高效矩阵运算
    # 计算加权成本：sum(vwap(t)*RSDAmt(T,t))
    weighted_cost = np.einsum('ij,ij,ij->i', vwap_matrix, chip_matrix, valid_mask.astype(np.float32))
    
    # 计算总筹码：sum(RSDAmt(T,t))
    total_chips = np.sum(chip_matrix * valid_mask, axis=1)
    
    # 计算平均成本，避免除零
    avg_cost = np.divide(weighted_cost, total_chips, out=np.zeros_like(weighted_cost), where=total_chips > 0)
    
    # 计算筹码收益
    current_prices = merged_df['close'].astype(np.float32).values
    holding_returns = np.divide(current_prices - avg_cost, avg_cost, out=np.zeros_like(current_prices), where=avg_cost > 0)
    
    # 将结果添加到DataFrame
    merged_df['holding_ret'] = holding_returns
    
    print("筹码收益因子计算完成")
    return merged_df

def calculate_market_holding_return(df):
    """计算市场筹码收益 - 超高效向量化版本
    按照理论：mkt_holding_ret为当日市场按照RSDAmt(T,t)为权重对holding_ret加权平均计算的市场平均筹码持有收益
    使用超高效向量化操作提高效率
    """
    print("正在计算市场筹码收益（超高效向量化版本）...")
    
    # 获取筹码留存列名
    chip_cols = [col for col in df.columns if col.startswith('chip_retention_')]
    
    # 使用pandas的groupby和apply进行向量化计算
    def calculate_daily_market_return(group):
        if len(group) == 0:
            return pd.Series({'mkt_holding_ret': np.nan})
        
        # 向量化计算当日所有股票的筹码留存总和
        chip_sums = group[chip_cols].sum(axis=1).values
        holding_rets = group['holding_ret'].values
        
        # 过滤有效数据
        valid_mask = chip_sums > 0
        valid_chips = chip_sums[valid_mask]
        valid_rets = holding_rets[valid_mask]
        
        if len(valid_chips) > 0:
            # 使用筹码留存作为权重计算加权平均
            weighted_avg = np.average(valid_rets, weights=valid_chips)
            return pd.Series({'mkt_holding_ret': weighted_avg})
        else:
            return pd.Series({'mkt_holding_ret': np.nan})
    
    # 按日期分组并应用向量化计算
    market_df = df.groupby('date').apply(calculate_daily_market_return).reset_index()
    
    # 过滤掉无效数据
    market_df = market_df.dropna()
    
    print("市场筹码收益计算完成")
    return market_df

def calculate_adjusted_factor(df, market_df):
    """计算筹码收益调整因子 - 向量化版本
    按照理论：holding_ret_adj = holding_ret * sign(mkt_holding_ret)
    其中mkt_holding_ret代表当日市场的按照RSDAmt(T,t)为权重对holding_ret加权平均计算的市场平均筹码持有收益
    sign()代表负数取-1正数取+1
    使用向量化操作提高效率
    """
    print("正在计算筹码收益调整因子（向量化版本）...")
    
    # 合并市场数据
    result_df = pd.merge(df, market_df, on='date', how='inner')
    
    # 向量化计算调整因子：holding_ret_adj = holding_ret * sign(mkt_holding_ret)
    result_df['holding_ret_adj'] = result_df['holding_ret'] * np.sign(result_df['mkt_holding_ret'])
    
    print("筹码收益调整因子计算完成")
    return result_df

def save_factor_data(df, factor_name):
    """保存因子数据"""
    # 创建因子库目录
    factor_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '因子库')
    if not os.path.exists(factor_dir):
        os.makedirs(factor_dir)
    
    # 检查数据格式并相应处理
    if 'order_book_id' in df.columns:
        # 转换为因子矩阵格式：行为日期，列为股票代码
        factor_matrix = df.pivot(index='date', columns='order_book_id', values=factor_name)
    else:
        # 对于市场数据，直接使用日期作为索引
        factor_matrix = df.set_index('date')[factor_name]
        # 转换为DataFrame格式以保持一致性
        factor_matrix = pd.DataFrame(factor_matrix)
    
    # 保存因子
    factor_path = os.path.join(factor_dir, f'{factor_name}_factor.pkl')
    factor_matrix.to_pickle(factor_path)
    
    print(f"\n{factor_name}因子已保存到: {factor_path}")
    print("保存后的数据格式如下（前5行前5列）：")
    print(factor_matrix.iloc[:5, :5])
    print(f"\n因子矩阵形状: {factor_matrix.shape}")
    print(f"日期范围: {factor_matrix.index.min()} 到 {factor_matrix.index.max()}")
    if 'order_book_id' in df.columns:
        print(f"股票数量: {len(factor_matrix.columns)}")
    else:
        print("市场因子，无股票维度")
    
    return factor_matrix

def main():
    """主函数"""
    print("开始构造筹码峰因子（多核并行+超高效向量化版本）...")
    
    try:
        # 1. 获取股票数据（使用较小数量进行测试，验证优化效果）
        df = get_stock_data(START_DATE, END_DATE, max_stocks=None)  # 先测试50只股票
        
        # 2. 计算筹码留存
        chip_df = calculate_chip_retention_optimized(df, CHIP_WINDOW)
        
        # 3. 计算筹码收益因子
        result_df = calculate_holding_return_optimized(df, chip_df, CHIP_WINDOW)
        
        # 4. 计算市场筹码收益
        market_df = calculate_market_holding_return(result_df)
        
        # 5. 计算筹码收益调整因子
        final_df = calculate_adjusted_factor(result_df, market_df)
        
        # 6. 保存因子数据
        print("\n=== 保存因子数据 ===")
        
        # 保存筹码收益因子
        holding_ret_matrix = save_factor_data(final_df, 'holding_ret')
        
        # 保存筹码收益调整因子
        holding_ret_adj_matrix = save_factor_data(final_df, 'holding_ret_adj')
        
        # 保存市场筹码收益
        market_matrix = save_factor_data(market_df, 'mkt_holding_ret')
        
        print("\n筹码峰因子构造完成！")
        
        # 显示因子统计信息
        print("\n=== 因子统计信息 ===")
        print("筹码收益因子统计：")
        print(final_df['holding_ret'].describe())
        print("\n筹码收益调整因子统计：")
        print(final_df['holding_ret_adj'].describe())
        print("\n市场筹码收益统计：")
        print(market_df['mkt_holding_ret'].describe())
        
    except Exception as e:
        print(f"构造筹码峰因子时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()