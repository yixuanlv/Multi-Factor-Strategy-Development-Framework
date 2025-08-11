import pandas as pd
import numpy as np
import os
import sys
import pickle

# ====== 可调参数 ======
VOL_WINDOW = 20  # 波动率窗口参数，可根据需要修改

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

# 计算日收益率 (1日涨跌幅)，明确指定fill_method=None避免警告
df['daily_return'] = df.groupby('order_book_id')['close'].pct_change(fill_method=None)

# 使用更稳定的方法计算滚动标准差
def calculate_rolling_std(group, window=VOL_WINDOW):
    """计算滚动标准差"""
    return group.rolling(window=window, min_periods=window).std()

# 按股票分组计算滚动标准差
std_col = f'std_{VOL_WINDOW}'
df[std_col] = df.groupby('order_book_id')['daily_return'].apply(
    lambda x: x.rolling(window=VOL_WINDOW, min_periods=VOL_WINDOW).std()
).reset_index(level=0, drop=True)

# 年化波动率：乘以sqrt(250)
vol_col = f'volatility_{VOL_WINDOW}'
df[vol_col] = df[std_col] * np.sqrt(250)

# 显示结果
print(f"\n=== {VOL_WINDOW}日收益波动率计算结果 ===")
print("包含波动率因子的数据前10行：")
print(df[['date', 'order_book_id', 'close', 'daily_return', std_col, vol_col]].head(10))

# 统计信息
print(f"\n=== 波动率因子统计信息 ===")
print(f"{VOL_WINDOW}日波动率统计：")
print(df[vol_col].describe())

# 将df转换为 行为trade_date，列为order_book_id，值为volatility_xxx 的格式
if vol_col in df.columns:
    pivot_df = df.pivot(index='date', columns='order_book_id', values=vol_col)
    factor_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '因子库')
    if not os.path.exists(factor_dir):
        os.makedirs(factor_dir)
    factor_path = os.path.join(factor_dir, f'volatility_{VOL_WINDOW}_factor.pkl')
    pivot_df.to_pickle(factor_path)
    print(f"\n{VOL_WINDOW}日收益波动率因子已保存到: {factor_path}")
    print("保存后的数据格式如下（前5行前5列）：")
    print(pivot_df.iloc[:5, :5])
    print(f"\n因子矩阵形状: {pivot_df.shape}")
    print(f"日期范围: {pivot_df.index.min()} 到 {pivot_df.index.max()}")
    print(f"股票数量: {len(pivot_df.columns)}")
else:
    print("未生成波动率因子，未保存。")
