import pandas as pd
import numpy as np
import os
import sys
import pickle

# ====== 可调参数 ======
MOMENTUM_WINDOW = 20  # 日内动量窗口参数，可根据需要修改

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

# 计算日内收益率 (close/open)
df['intraday_return'] = df['close'] / df['open']

# 使用更稳定的方法计算滚动累乘
def calculate_rolling_cumprod(group, window=MOMENTUM_WINDOW):
    """计算滚动累乘"""
    return group.rolling(window=window, min_periods=window).apply(
        lambda x: np.prod(x), raw=True
    )

# 按股票分组计算滚动累乘
momentum_col = f'intraday_momentum_{MOMENTUM_WINDOW}'
df[momentum_col] = df.groupby('order_book_id')['intraday_return'].apply(
    lambda x: x.rolling(window=MOMENTUM_WINDOW, min_periods=MOMENTUM_WINDOW).apply(
        lambda y: np.prod(y), raw=True
    )
).reset_index(level=0, drop=True)

# 显示结果
print(f"\n=== {MOMENTUM_WINDOW}日日内动量因子计算结果 ===")
print("包含日内动量因子的数据前10行：")
print(df[['date', 'order_book_id', 'open', 'close', 'intraday_return', momentum_col]].head(10))

# 统计信息
print(f"\n=== 日内动量因子统计信息 ===")
print(f"{MOMENTUM_WINDOW}日日内动量统计：")
print(df[momentum_col].describe())

# 将df转换为 行为trade_date，列为order_book_id，值为intraday_momentum_xxx 的格式
if momentum_col in df.columns:
    pivot_df = df.pivot(index='date', columns='order_book_id', values=momentum_col)
    factor_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '因子库')
    if not os.path.exists(factor_dir):
        os.makedirs(factor_dir)
    factor_path = os.path.join(factor_dir, f'intraday_momentum_{MOMENTUM_WINDOW}_factor.pkl')
    pivot_df.to_pickle(factor_path)
    print(f"\n{MOMENTUM_WINDOW}日日内动量因子已保存到: {factor_path}")
    print("保存后的数据格式如下（前5行前5列）：")
    print(pivot_df.iloc[:5, :5])
    print(f"\n因子矩阵形状: {pivot_df.shape}")
    print(f"日期范围: {pivot_df.index.min()} 到 {pivot_df.index.max()}")
    print(f"股票数量: {len(pivot_df.columns)}")
else:
    print("未生成日内动量因子，未保存。")
