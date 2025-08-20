# -*- coding: utf-8 -*-
"""
股票持仓风格和行业暴露分析
读取回测结果和Barra因子，计算持仓的风格和行业暴露，绘制热力图
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
import sys
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    result = pd.read_pickle(r"C:\Users\9shao\Desktop\github公开项目\Multi-Factor-Strategy-Development-Framework\测试代码\回测结果\测试策略_1(2).pkl")

    # 如果result是字典，尝试提取其中的DataFrame
    if isinstance(result, dict):
        # 常见的DataFrame结果key有'portfolio', 'trades', 'benchmark_portfolio', 'orders'等
        for key, value in result.items():
            if isinstance(value, pd.DataFrame):
                print(f"key: {key}, DataFrame shape: {value.shape}")
        # 例如，提取'portfolio'结果
        if 'stock_positions' in result:
            df = result['stock_positions']
        else:
            # 如果没有'portfolio'，取第一个DataFrame
            for value in result.values():
                if isinstance(value, pd.DataFrame):
                    df = value
                    break
            else:
                raise ValueError("未找到DataFrame类型的数据")
    else:
        # 如果本身就是DataFrame
        df = pd.DataFrame(result)
    # 加载Barra因子
    try:
        barra_path = r"C:\Users\9shao\Desktop\github公开项目\Multi-Factor-Strategy-Development-Framework\多因子框架\因子库\Barra因子\barra.pkl"
        print(f"尝试加载Barra因子: {barra_path}")
        
        if not os.path.exists(barra_path):
            print(f"Barra因子文件不存在: {barra_path}")
            return None, None
        
        with open(barra_path, 'rb') as f:
            barra_data = pickle.load(f)
        print(f"Barra因子数据形状: {barra_data.shape}")
        print(f"Barra因子列名: {list(barra_data.columns)}")
        
    except Exception as e:
        print(f"加载Barra因子失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    return df, barra_data

def process_stock_positions(stock_positions):
    """处理股票持仓数据"""
    print("正在处理股票持仓数据...")
    
    # 检查数据结构
    print(f"持仓数据列名: {list(stock_positions.columns)}")
    print(f"持仓数据索引: {stock_positions.index}")
    print(f"持仓数据前5行:")
    print(stock_positions.head())
    
    # 如果持仓数据是MultiIndex，需要重新整理
    if stock_positions.index.nlevels > 1:
        # 假设索引是 (date, order_book_id)
        stock_positions = stock_positions.reset_index()
        print("重置索引后的列名:", list(stock_positions.columns))
    
    # 查找日期和股票代码列
    date_col = None
    stock_col = None
    weight_col = None
    
    for col in stock_positions.columns:
        col_lower = str(col).lower()
        if 'date' in col_lower or 'time' in col_lower:
            date_col = col
        elif 'stock' in col_lower or 'code' in col_lower or 'order_book' in col_lower:
            stock_col = col
        elif 'weight' in col_lower or 'position' in col_lower or 'ratio' in col_lower:
            weight_col = col
    
    if date_col is None or stock_col is None:
        print("警告: 无法识别日期或股票代码列，尝试使用前几列...")
        if len(stock_positions.columns) >= 2:
            date_col = stock_positions.columns[0]
            stock_col = stock_positions.columns[1]
            if len(stock_positions.columns) >= 3:
                weight_col = stock_positions.columns[2]
    
    print(f"使用列: 日期={date_col}, 股票={stock_col}, 权重={weight_col}")
    
    # 如果没有权重列，假设等权重
    if weight_col is None:
        print("未找到权重列，使用等权重计算")
        # 按日期分组，每只股票等权重
        stock_positions = stock_positions.groupby(date_col).apply(
            lambda x: x.assign(weight=1.0/len(x))
        ).reset_index(drop=True)
        weight_col = 'weight'
    
    # 确保日期格式正确
    try:
        stock_positions[date_col] = pd.to_datetime(stock_positions[date_col])
    except Exception as e:
        print(f"日期转换失败: {e}")
        print(f"日期列示例值: {stock_positions[date_col].head()}")
        return None
    
    # 选择需要的列
    processed_positions = stock_positions[[date_col, stock_col, weight_col]].copy()
    processed_positions.columns = ['date', 'order_book_id', 'weight']
    
    print(f"处理后的持仓数据形状: {processed_positions.shape}")
    print(f"日期范围: {processed_positions['date'].min()} 到 {processed_positions['date'].max()}")
    print(f"股票数量: {processed_positions['order_book_id'].nunique()}")
    
    return processed_positions

def process_barra_data(barra_data):
    """处理Barra因子数据"""
    print("正在处理Barra因子数据...")
    
    # 确保日期列是datetime类型
    barra_data['date'] = pd.to_datetime(barra_data['date'])
    
    # 分离风格因子和行业因子
    style_factors = ['size', 'non_linear_size', 'momentum', 'liquidity', 'book_to_price', 
                     'leverage', 'growth', 'earnings_yield', 'beta', 'residual_volatility', 'comovement']
    
    # 检查哪些风格因子存在
    available_style_factors = [col for col in style_factors if col in barra_data.columns]
    print(f"可用的风格因子: {available_style_factors}")
    
    # 行业因子是除了风格因子和基础列之外的所有列
    industry_factors = [col for col in barra_data.columns if col not in style_factors + ['date', 'order_book_id']]
    print(f"可用的行业因子: {industry_factors}")
    
    # 设置索引
    barra_data = barra_data.set_index(['date', 'order_book_id'])
    
    return barra_data, available_style_factors, industry_factors

def calculate_exposures(positions, barra_data, style_factors, industry_factors):
    """计算持仓的风格和行业暴露"""
    print("正在计算持仓暴露...")
    
    # 合并持仓和Barra因子数据
    merged_data = positions.merge(
        barra_data.reset_index(), 
        on=['date', 'order_book_id'], 
        how='inner'
    )
    
    print(f"合并后数据形状: {merged_data.shape}")
    
    # 按日期分组计算加权暴露
    exposures = []
    
    for date, group in merged_data.groupby('date'):
        if len(group) == 0:
            continue
            
        # 计算风格因子暴露
        style_exposure = {}
        for factor in style_factors:
            if factor in group.columns:
                # 加权平均
                weighted_exposure = np.average(group[factor], weights=group['weight'])
                style_exposure[factor] = weighted_exposure
        
        # 计算行业因子暴露
        industry_exposure = {}
        for factor in industry_factors:
            if factor in group.columns:
                # 加权平均
                weighted_exposure = np.average(group[factor], weights=group['weight'])
                industry_exposure[factor] = weighted_exposure
        
        exposures.append({
            'date': date,
            **style_exposure,
            **industry_exposure
        })
    
    exposures_df = pd.DataFrame(exposures)
    exposures_df = exposures_df.set_index('date').sort_index()
    
    print(f"暴露数据形状: {exposures_df.shape}")
    print(f"暴露数据列: {list(exposures_df.columns)}")
    
    return exposures_df

def resample_to_quarterly(exposures_df):
    """将日度数据重采样为季度数据"""
    print("正在重采样为季度数据...")
    
    # 按季度重采样，取平均值
    quarterly_exposures = exposures_df.resample('Q').mean()
    
    print(f"季度暴露数据形状: {quarterly_exposures.shape}")
    print(f"季度范围: {quarterly_exposures.index[0]} 到 {quarterly_exposures.index[-1]}")
    
    return quarterly_exposures

def plot_heatmaps(quarterly_exposures, style_factors, industry_factors):
    """绘制风格和行业暴露热力图"""
    print("正在绘制热力图...")
    
    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # 1. 风格因子暴露热力图
    if style_factors:
        style_data = quarterly_exposures[style_factors].T
        
        # 绘制风格因子热力图
        sns.heatmap(style_data, 
                    ax=axes[0],
                    cmap='RdBu_r', 
                    center=0,
                    annot=True, 
                    fmt='.3f',
                    cbar_kws={'label': '风格因子暴露'},
                    xticklabels=[d.strftime('%Y-%m') for d in style_data.columns],
                    yticklabels=style_data.index)
        
        axes[0].set_title('持仓风格因子暴露热力图（季度平均）', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('时间（季度）')
        axes[0].set_ylabel('风格因子')
    
    # 2. 行业因子暴露热力图
    if industry_factors:
        industry_data = quarterly_exposures[industry_factors].T
        
        # 绘制行业因子热力图
        sns.heatmap(industry_data, 
                    ax=axes[1],
                    cmap='RdBu_r', 
                    center=0,
                    annot=True, 
                    fmt='.3f',
                    cbar_kws={'label': '行业因子暴露'},
                    xticklabels=[d.strftime('%Y-%m') for d in industry_data.columns],
                    yticklabels=industry_data.index)
        
        axes[1].set_title('持仓行业因子暴露热力图（季度平均）', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('时间（季度）')
        axes[1].set_ylabel('行业因子')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = r"C:\Users\9shao\Desktop\github公开项目\Multi-Factor-Strategy-Development-Framework\测试代码\回测结果\持仓风格行业暴露热力图.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"热力图已保存到: {output_path}")
    
    plt.show()
    
    return fig

def main():
    """主函数"""
    print("=== 股票持仓风格和行业暴露分析 ===")
    
    # 1. 加载数据
    stock_positions, barra_data = load_data()
    if stock_positions is None or barra_data is None:
        print("数据加载失败，程序退出")
        return
    
    # 2. 处理股票持仓数据
    processed_positions = process_stock_positions(stock_positions)
    if processed_positions is None:
        print("股票持仓数据处理失败，程序退出")
        return
    
    # 3. 处理Barra因子数据
    barra_data, style_factors, industry_factors = process_barra_data(barra_data)
    
    # 4. 计算持仓暴露
    exposures_df = calculate_exposures(processed_positions, barra_data, style_factors, industry_factors)
    
    # 5. 重采样为季度数据
    quarterly_exposures = resample_to_quarterly(exposures_df)
    
    # 6. 绘制热力图
    fig = plot_heatmaps(quarterly_exposures, style_factors, industry_factors)
    
    # 7. 输出统计信息
    print("\n=== 暴露统计信息 ===")
    print(f"风格因子暴露统计:")
    if style_factors:
        style_stats = quarterly_exposures[style_factors].describe()
        print(style_stats)
    
    print(f"\n行业因子暴露统计:")
    if industry_factors:
        industry_stats = quarterly_exposures[industry_factors].describe()
        print(industry_stats)
    
    print("\n=== 分析完成 ===")

if __name__ == "__main__":
    main()
