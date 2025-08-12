import pandas as pd
import numpy as np
import os
import sys
import pickle
import rqdatac as rq

# 初始化rqdatac
ricequant = rq.init('license','FTp2uHaRsHQhI9XoFCwKkJNyYdaZ9f6oMHT4kPr1Pq95n2VHHH_KnxinjLrktuwQ1AuW72X5vSrS01MGqOw8OWRD8W3B_EWHNBWGz5vE2A3cLKxEz25vNYeXIbzDbt5v8crTY1OOkjjypfOnfnItH5r95_8C3ck0QmGHgMyMbTw=U_T-NcVYHMdYw9LovsISVLqco69JY8b6093uhl6PUG4S2gEXQOO_RAuTwjbFEK-GXnzutyBAP-s-3JpXpkjdnfQ96Ypjzl6J1DULxSGzqWQo6LWJwVw8YB125DfT5oSHqk9lUsgriFojSQqG92uKh8HKpZK4fVvNhp3Lv2U410c=')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    

# 读取行情数据
df = pd.read_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '行情数据库', 'data.pkl'))

# 设置显示所有行和列
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("原始数据信息：")
print(df.head())
print("\n数据形状：", df.shape)
print("\n列名：", df.columns.tolist())

# 确保数据按日期和股票代码排序
df = df.sort_values(['date', 'order_book_id']).reset_index(drop=True)

# 获取所有唯一的股票代码
all_stocks = df['order_book_id'].unique().tolist()
print(f"\n总共有 {len(all_stocks)} 只股票")

# 获取股息率TTM因子
print("\n正在获取股息率TTM因子...")
print("时间范围：2010-01-01 到 2025-06-30")

# 你的代码报错的原因是：rq.get_factor 返回的是一个以 MultiIndex（date, order_book_id）为索引的 DataFrame，
# 而你直接赋值给原始 df 的新列，索引对不上，pandas 无法对齐，导致 TypeError。

# 正确做法：先获取股息率ttm因子，重置索引，然后 merge 回原始 df

# 获取全量日期范围
start_date = df['date'].min().strftime('%Y%m%d') if hasattr(df['date'].iloc[0], 'strftime') else str(df['date'].min()).replace('-','')
end_date = df['date'].max().strftime('%Y%m%d') if hasattr(df['date'].iloc[0], 'strftime') else str(df['date'].max()).replace('-','')

dividend_yield = rq.get_factor(
    all_stocks,
    factor=['dividend_yield_ttm'],
    start_date=start_date,
    end_date=end_date
)

# dividend_yield 是 MultiIndex，重置为普通列
dividend_yield = dividend_yield.reset_index()  # 包含 date, order_book_id, dividend_yield_ttm

# 合并回原始 df
df = pd.merge(df, dividend_yield, on=['date', 'order_book_id'], how='left')

# 将所有NaN值填充为0
df = df.fillna(0)

print(df[['date', 'order_book_id', 'dividend_yield_ttm']].head())

if 'dividend_yield_ttm' in df.columns:
    pivot_df = df.pivot(index='date', columns='order_book_id', values='dividend_yield_ttm')
    factor_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '因子库')
    if not os.path.exists(factor_dir):
        os.makedirs(factor_dir)
    factor_path = os.path.join(factor_dir, 'dividend_yield_ttm_factor.pkl')
    pivot_df.to_pickle(factor_path)
    print(f"\n股息率TTM因子已保存到: {factor_path}")
    print("保存后的数据格式如下（前5行前5列）：")
    print(pivot_df.iloc[:5, :5])
    print(f"\n因子矩阵形状: {pivot_df.shape}")
    print(f"日期范围: {pivot_df.index.min()} 到 {pivot_df.index.max()}")
    print(f"股票数量: {len(pivot_df.columns)}")
else:
    print("未生成波动率因子，未保存。")