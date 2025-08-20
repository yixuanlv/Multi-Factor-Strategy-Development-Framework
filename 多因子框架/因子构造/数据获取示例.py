# -*- coding: utf-8 -*-
"""
数据获取示例
演示如何使用rq.get_factor()获取数据
"""

import pickle
import pandas as pd
import numpy as np

def load_factor_names():
    """加载因子名称"""
    try:
        with open(r'C:\Users\9shao\Desktop\github公开项目\Multi-Factor-Strategy-Development-Framework\多因子框架\因子库\米筐因子名称\factor_names.pkl', 'rb') as f:
            factor_names = pickle.load(f)
        print(f"成功加载 {len(factor_names)} 个因子名称")
        return factor_names
    except Exception as e:
        print(f"加载因子名称失败: {e}")
        return None

def get_sample_data():
    """获取示例数据（模拟rq.get_factor()的返回格式）"""
    # 模拟数据格式
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    assets = ['000001.SZ', '000002.SZ', '000858.SZ', '600000.SH', '600036.SH']
    
    # 创建MultiIndex
    index = pd.MultiIndex.from_product([dates, assets], names=['date', 'order_book_id'])
    
    # 模拟各种指标数据
    data = {
        'close': pd.Series(np.random.randn(len(index)), index=index),
        'volume': pd.Series(np.random.randn(len(index)), index=index),
        'revenue_TTM': pd.Series(np.random.randn(len(index)), index=index),
        'net_profit_TTM': pd.Series(np.random.randn(len(index)), index=index),
        'consensus_rating_30': pd.Series(np.random.randn(len(index)), index=index),
        'size': pd.Series(np.random.randn(len(index)), index=index),
        'future_ret': pd.Series(np.random.randn(len(index)), index=index)
    }
    
    return data

def demonstrate_data_structure():
    """演示数据结构"""
    print("=== 数据获取示例 ===")
    
    # 1. 加载因子名称
    factor_names = load_factor_names()
    if factor_names is not None:
        print("\n前10个因子名称:")
        for i, name in enumerate(factor_names.head(10)['factor_name']):
            print(f"{i+1:2d}. {name}")
    
    # 2. 获取示例数据
    print("\n2. 数据格式示例:")
    data = get_sample_data()
    
    for name, series in data.items():
        print(f"  {name}: {series.shape}, 索引: {series.index.names}")
        print(f"    前5行: {series.head()}")
        print()
    
    # 3. 说明rq.get_factor()的使用
    print("3. rq.get_factor()使用说明:")
    print("   rq.get_factor(order_book_ids, factor, start_date, end_date, universe=None, expect_df=True)")
    print("   返回格式: DataFrame with MultiIndex (date, order_book_id)")
    print("   列名: 因子名称")
    
    # 4. 实际使用建议
    print("\n4. 实际使用建议:")
    print("   - 使用conda activate rqplus激活环境")
    print("   - 分批获取因子数据，避免内存溢出")
    print("   - 注意数据频率：日频vs季频")
    print("   - 处理缺失值和异常值")

if __name__ == "__main__":
    demonstrate_data_structure()
