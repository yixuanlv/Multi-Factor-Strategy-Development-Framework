import pandas as pd
import numpy as np
import os
import sys
import pickle

# ====== 可调参数 ======
RED_GREEN_WINDOW = 20  # 红肥绿瘦因子窗口参数，可根据需要修改

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    

df = pd.read_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '行情数据库', 'data.pkl'))

# 设置显示所有行和列
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print(df.head())
print("\n数据形状：", df.shape)
print("\n列名：", df.columns.tolist())

# 确保数据按日期和股票代码排序
df = df.sort_values(['date', 'order_book_id']).reset_index(drop=True)

# 计算K线实线部分占K线总长度的比例
# 实线长度 = |close - open|
# K线总长度 = high - low
df['body_length'] = np.abs(df['close'] - df['open'])
df['total_length'] = df['high'] - df['low']
df['body_ratio'] = df['body_length'] / df['total_length']

# 处理除零情况，如果总长度为0，则实线比例为0
df['body_ratio'] = df['body_ratio'].fillna(0)

# 判断每日是上涨还是下跌
df['is_up'] = df['close'] > df['open']
df['is_down'] = df['close'] < df['open']

# 定义计算红肥绿瘦因子的函数
def calculate_red_green_factor(group, window=RED_GREEN_WINDOW):
    """
    计算红肥绿瘦因子
    在N日滚动窗口内，计算上涨日和下跌日实线比例的平均值之差
    """
    # 获取数值列
    body_ratio = group['body_ratio'].values
    is_up = group['is_up'].values
    is_down = group['is_down'].values
    
    # 初始化结果数组
    result = np.full(len(group), np.nan)
    
    # 从第window-1个位置开始计算（因为需要window个数据点）
    for i in range(window-1, len(group)):
        # 获取当前窗口的数据
        window_body_ratio = body_ratio[i-window+1:i+1]
        window_is_up = is_up[i-window+1:i+1]
        window_is_down = is_down[i-window+1:i+1]
        
        # 分离上涨日和下跌日
        up_days_ratio = window_body_ratio[window_is_up]
        down_days_ratio = window_body_ratio[window_is_down]
        
        # 计算上涨日实线比例平均值
        up_avg = up_days_ratio.mean() if len(up_days_ratio) > 0 else 0
        
        # 计算下跌日实线比例平均值
        down_avg = down_days_ratio.mean() if len(down_days_ratio) > 0 else 0
        
        # 返回上涨日平均值减去下跌日平均值
        result[i] = up_avg - down_avg  # 注意这里是减法，原来是加法，修正为“上涨日均值-下跌日均值”
    
    return pd.Series(result, index=group.index)

# 按股票分组计算红肥绿瘦因子
factor_col = f'red_green_factor_{RED_GREEN_WINDOW}'

# 创建包含必要列的DataFrame用于计算
calc_df = df[['order_book_id', 'body_ratio', 'is_up', 'is_down']].copy()

# 按股票分组计算因子
df[factor_col] = calc_df.groupby('order_book_id').apply(
    lambda x: calculate_red_green_factor(x, RED_GREEN_WINDOW)
).reset_index(level=0, drop=True)

# 0值替换为nan（最后一步）
df[factor_col] = df[factor_col].replace(0, np.nan)

# 显示结果
print(f"\n=== {RED_GREEN_WINDOW}日红肥绿瘦因子计算结果 ===")
print("包含红肥绿瘦因子的数据前10行：")
print(df[['date', 'order_book_id', 'open', 'close', 'high', 'low', 'body_ratio', 'is_up', 'is_down', factor_col]].head(10))

# 统计信息
print(f"\n=== 红肥绿瘦因子统计信息 ===")
print(f"{RED_GREEN_WINDOW}日红肥绿瘦因子统计：")
print(df[factor_col].describe())

# 显示一些中间计算结果
print(f"\n=== 中间计算结果 ===")
print("实线比例统计：")
print(df['body_ratio'].describe())
print(f"\n上涨日数量: {df['is_up'].sum()}")
print(f"下跌日数量: {df['is_down'].sum()}")
print(f"平盘日数量: {len(df) - df['is_up'].sum() - df['is_down'].sum()}")

# 将df转换为 行为trade_date，列为order_book_id，值为red_green_factor_xxx 的格式
if factor_col in df.columns:
    pivot_df = df.pivot(index='date', columns='order_book_id', values=factor_col)
    factor_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '因子库')
    if not os.path.exists(factor_dir):
        os.makedirs(factor_dir)
    factor_path = os.path.join(factor_dir, f'red_green_factor_{RED_GREEN_WINDOW}_factor.pkl')
    pivot_df.to_pickle(factor_path)
    print(f"\n{RED_GREEN_WINDOW}日红肥绿瘦因子已保存到: {factor_path}")
    print("保存后的数据格式如下（后5行后5列）：")
    print(pivot_df.iloc[-5:, -5:])
    print(f"\n因子矩阵形状: {pivot_df.shape}")
    print(f"日期范围: {pivot_df.index.min()} 到 {pivot_df.index.max()}")
    print(f"股票数量: {len(pivot_df.columns)}")
else:
    print("未生成红肥绿瘦因子，未保存。")
