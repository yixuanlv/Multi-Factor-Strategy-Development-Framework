
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# 初始化米筐接口
import rqdatac as rq
rq.init('license', 'I6eb8ljE6tv9DWcYa3F0hERxbDdQ3f0RZzgBEnTHuaQlVX56azMDhdh6dYB42IJ7onLu0mAl3A1rRGFVTQuxE4jZcwoByZaySYlNuInciyFGarrHTz24mblwqbrC4RaCvKbkxP-tZ9S7ZjDY8pTNWu4uIslVXYb4XXL9NSwGI58=T7bU8OlDqvS3R5pPNN7s3PsfirJTCFSHPXpm5Ak3n0Dpzaze0NLbHWfZ-JcnlTBz7Oxec6dmkH9X4UB0OT0qxlkHA3pX_muOZI_zgMpCNZFH1wZ-DjeEMXrkqGGBKIo6_rZeaz130Fo1PLRY-rTw71gPmhD1oVg7GVh1kC7SWrk=') 

# 筹码峰因子参数
CHIP_WINDOW = 60  # 筹码留存窗口
START_DATE = '2020-01-01'
END_DATE = '2025-07-31'

def get_stock_data(start_date, end_date, max_stocks=None):
    """获取股票数据"""
    print("正在获取股票列表...")
    
    # 获取A股列表
    all_stocks = rq.all_instruments(type='CS', date=None)['order_book_id'].tolist()
    if max_stocks is None:
        test_stocks = all_stocks
    else:
        test_stocks = all_stocks[:max_stocks]
    
    print(f"选择{len(test_stocks)}只股票进行测试")
    
    # 获取数据
    print("正在获取股票历史数据...")
    ohlcv_data = rq.get_price(test_stocks, start_date=start_date, end_date=end_date, 
                              frequency='1d', fields=['open', 'high', 'low', 'close', 'volume'],
                              adjust_type='post', skip_suspended=False, market='cn', expect_df=True)
    
    turnover_data = rq.get_turnover_rate(test_stocks, start_date=start_date, end_date=end_date, expect_df=True)
    vwap_data = rq.get_vwap(test_stocks, start_date=start_date, end_date=end_date, frequency='1d')
    
    # 重置索引并重命名列
    ohlcv_data = ohlcv_data.reset_index()
    turnover_data = turnover_data.reset_index()
    vwap_data = vwap_data.reset_index()
    
    # 重命名列
    if len(ohlcv_data.columns) == 7:
        ohlcv_data.columns = ['order_book_id', 'date', 'open', 'high', 'low', 'close', 'volume']
    
    if len(turnover_data.columns) == 7:
        turnover_data = turnover_data[['order_book_id', 'tradedate', 'today']]
        turnover_data.columns = ['order_book_id', 'date', 'turnover_rate']
    
    if len(vwap_data.columns) == 3:
        vwap_data.columns = ['order_book_id', 'date', 'vwap']
    
    # 合并数据
    data = pd.merge(ohlcv_data, turnover_data, on=['date', 'order_book_id'], how='inner')
    data = pd.merge(data, vwap_data, on=['date', 'order_book_id'], how='inner')
    
    # 排序并处理缺失值
    data = data.sort_values(['date', 'order_book_id']).reset_index(drop=True)
    data = data.dropna()
    
    print(f"数据获取完成，共{len(data)}条记录")
    return data

def calculate_chip_retention_rolling(df, window=CHIP_WINDOW):
    """使用pandas rolling操作计算筹码留存"""
    print("正在计算筹码留存...")
    
    # 按股票分组处理
    def process_stock(group):
        stock = group['order_book_id'].iloc[0]
        print(f"处理股票: {stock}")
        
        # 计算换手率衰减
        group['turnover_rate'] = group['turnover_rate'].fillna(0.01)
        
        # 使用rolling计算筹码留存
        chip_retention = pd.DataFrame()
        
        for k in range(window):
            if k == 0:
                # 当天数据
                chip_retention[f'chip_retention_{k}'] = group['volume']
                chip_retention[f'historical_vwap_{k}'] = group['vwap']
            else:
                # 历史数据
                # 计算换手率累积衰减
                decay = group['turnover_rate'].rolling(window=k, min_periods=1).apply(
                    lambda x: np.prod(1 - x.iloc[-k:]) if len(x) >= k else 0
                )
                
                # 获取k天前的数据
                hist_volume = group['volume'].shift(k).fillna(0)
                hist_vwap = group['vwap'].shift(k).fillna(group['vwap'])
                
                # 计算筹码留存
                chip_retention[f'chip_retention_{k}'] = hist_volume * decay
                chip_retention[f'historical_vwap_{k}'] = hist_vwap
        
        # 添加日期和股票代码
        chip_retention['date'] = group['date']
        chip_retention['order_book_id'] = stock
        
        return chip_retention
    
    # 按股票分组并应用处理函数
    results = []
    for stock in df['order_book_id'].unique():
        stock_data = df[df['order_book_id'] == stock].copy()
        if len(stock_data) >= window:
            result = process_stock(stock_data)
            results.append(result)
    
    # 合并结果
    final_result = pd.concat(results, ignore_index=True)
    print("筹码留存计算完成")
    return final_result

def calculate_holding_return_rolling(df, chip_df, window=CHIP_WINDOW):
    """使用pandas操作计算筹码收益因子"""
    print("正在计算筹码收益因子...")
    
    # 合并数据
    merged_df = pd.merge(df, chip_df, on=['date', 'order_book_id'], how='inner')
    
    # 获取筹码留存和VWAP列
    chip_cols = [f'chip_retention_{i}' for i in range(window)]
    vwap_cols = [f'historical_vwap_{i}' for i in range(window)]
    
    # 计算加权平均成本
    chip_matrix = merged_df[chip_cols].values
    vwap_matrix = merged_df[vwap_cols].values
    
    # 创建有效数据掩码
    valid_mask = (chip_matrix > 0) & np.isfinite(vwap_matrix)
    
    # 计算加权成本
    weighted_cost = np.sum(vwap_matrix * chip_matrix * valid_mask, axis=1)
    total_chips = np.sum(chip_matrix * valid_mask, axis=1)
    
    # 计算平均成本
    avg_cost = np.divide(weighted_cost, total_chips, out=np.zeros_like(weighted_cost), where=total_chips > 0)
    
    # 计算筹码收益
    current_prices = merged_df['close'].values
    holding_returns = np.divide(current_prices - avg_cost, avg_cost, out=np.zeros_like(current_prices), where=avg_cost > 0)
    
    # 处理异常值
    holding_returns = np.where(np.isnan(holding_returns) | np.isinf(holding_returns), 0, holding_returns)
    
    merged_df['holding_ret'] = holding_returns
    print("筹码收益因子计算完成")
    return merged_df

def calculate_market_holding_return_rolling(df):
    """计算市场筹码收益"""
    print("正在计算市场筹码收益...")
    
    # 获取筹码留存列
    chip_cols = [col for col in df.columns if col.startswith('chip_retention_')]
    
    # 按日期分组计算加权平均
    def calculate_daily_market_return(group):
        if len(group) == 0:
            return pd.Series({'mkt_holding_ret': 0})
        
        # 计算筹码留存总和
        chip_sums = group[chip_cols].sum(axis=1)
        holding_rets = group['holding_ret']
        
        # 过滤有效数据
        valid_mask = (chip_sums > 0) & np.isfinite(holding_rets)
        valid_chips = chip_sums[valid_mask]
        valid_rets = holding_rets[valid_mask]
        
        if len(valid_chips) > 0:
            try:
                weighted_avg = np.average(valid_rets, weights=valid_chips)
                return pd.Series({'mkt_holding_ret': weighted_avg if np.isfinite(weighted_avg) else 0})
            except:
                simple_avg = np.mean(valid_rets)
                return pd.Series({'mkt_holding_ret': simple_avg if np.isfinite(simple_avg) else 0})
        else:
            return pd.Series({'mkt_holding_ret': 0})
    
    # 按日期分组计算
    market_df = df.groupby('date').apply(calculate_daily_market_return).reset_index()
    market_df = market_df.dropna()
    
    print("市场筹码收益计算完成")
    return market_df

def calculate_adjusted_factor_rolling(df, market_df):
    """计算调整因子"""
    print("正在计算调整因子...")
    
    # 合并市场数据
    result_df = pd.merge(df, market_df, on='date', how='inner')
    
    # 计算调整因子
    market_signs = np.sign(result_df['mkt_holding_ret'].fillna(0))
    result_df['holding_ret_adj'] = result_df['holding_ret'] * market_signs
    
    # 处理异常值
    invalid_adj = np.isnan(result_df['holding_ret_adj']) | np.isinf(result_df['holding_ret_adj'])
    result_df.loc[invalid_adj, 'holding_ret_adj'] = 0
    
    print("调整因子计算完成")
    return result_df

def save_factor_data(df, factor_name):
    """保存因子数据"""
    # 创建因子库目录
    factor_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '因子库')
    if not os.path.exists(factor_dir):
        os.makedirs(factor_dir)
    
    # 转换为因子矩阵格式
    if 'order_book_id' in df.columns:
        factor_matrix = df.pivot(index='date', columns='order_book_id', values=factor_name)
    else:
        factor_matrix = df.set_index('date')[factor_name]
        factor_matrix = pd.DataFrame(factor_matrix)
    
    # 保存因子
    factor_path = os.path.join(factor_dir, f'{factor_name}_factor.pkl')
    factor_matrix.to_pickle(factor_path)
    
    print(f"{factor_name}因子已保存到: {factor_path}")
    print(f"因子矩阵形状: {factor_matrix.shape}")
    
    return factor_matrix

def main():
    """主函数"""
    print("开始构造筹码峰因子（精简版本）...")
    
    try:
        # 1. 获取股票数据
        df = get_stock_data(START_DATE, END_DATE, max_stocks=None)
        
        # 2. 计算筹码留存
        chip_df = calculate_chip_retention_rolling(df, CHIP_WINDOW)
        
        # 3. 计算筹码收益因子
        result_df = calculate_holding_return_rolling(df, chip_df, CHIP_WINDOW)
        
        # 4. 计算市场筹码收益
        market_df = calculate_market_holding_return_rolling(result_df)
        
        # 5. 计算调整因子
        final_df = calculate_adjusted_factor_rolling(result_df, market_df)
        
        # 6. 保存因子数据
        print("\n=== 保存因子数据 ===")
        holding_ret_matrix = save_factor_data(final_df, 'holding_ret')
        holding_ret_adj_matrix = save_factor_data(final_df, 'holding_ret_adj')
        market_matrix = save_factor_data(market_df, 'mkt_holding_ret')
        
        print("\n筹码峰因子构造完成！")
        
        # 显示统计信息
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