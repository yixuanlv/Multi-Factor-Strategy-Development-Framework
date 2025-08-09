


import pandas as pd
import numpy as np
import pickle
import os
from corr_linear_analysis import CollinearityAnalyzer, analyze_collinearity

work_dir = os.path.dirname(os.path.abspath(__file__))
print(work_dir)

def load_data(factor_names):
    """加载行情数据和多个因子数据"""
    print("正在加载数据...")
    
    # 加载行情数据
    data_path = os.path.join(work_dir, "../行情数据库/data.pkl")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    print(f"行情数据加载完成，数据形状: {data.shape}")
    
    # 加载多个因子数据
    factors_data = {}
    for factor_name in factor_names:
        factor_path = os.path.join(work_dir, f"../因子库/{factor_name}.pkl")
        try:
            with open(factor_path, 'rb') as f:
                factor_data = pickle.load(f)
            factors_data[factor_name] = factor_data
            print(f"因子 {factor_name} 数据加载完成，数据形状: {factor_data.shape}")
        except FileNotFoundError:
            print(f"警告：因子 {factor_name} 数据文件不存在，跳过")
            continue
    
    if not factors_data:
        raise ValueError("没有找到任何有效的因子数据文件")
    
    print(f"成功加载 {len(factors_data)} 个因子数据")
    return data, factors_data

def prepare_data(data, factors_data):
    """准备分析所需的数据格式"""
    print("正在准备数据...")
    
    # 处理行情数据
    data_reset = None
    if 'close' not in data.columns:
        if isinstance(data.index, pd.MultiIndex):
            data_reset = data.reset_index()
            if len(data_reset.columns) == 3:
                data_reset.columns = ['date', 'order_book_id', 'close']
            else:
                print("数据格式需要调整，请检查数据结构")
                return None, None
        else:
            print("请检查行情数据格式，需要包含close列")
            return None, None
    else:
        if isinstance(data.index, pd.MultiIndex):
            data_reset = data.reset_index()
        else:
            data_reset = data.copy()
    
    # 准备多个因子数据
    factors_data_ready = {}
    for factor_name, factor_data in factors_data.items():
        factor_reset = None
        if isinstance(factor_data.index, pd.MultiIndex):
            factor_reset = factor_data.reset_index()
            if len(factor_reset.columns) == 3:
                factor_reset.columns = ['date', 'order_book_id', 'factor_value']
            else:
                print(f"因子 {factor_name} 数据格式需要调整，请检查数据结构")
                continue
        elif isinstance(factor_data, pd.DataFrame) and not isinstance(factor_data.index, pd.MultiIndex):
            factor_reset = factor_data.stack().reset_index()
            factor_reset.columns = ['date', 'order_book_id', 'factor_value']
        else:
            print(f"因子 {factor_name} 数据格式错误，跳过")
            continue
        
        factors_data_ready[factor_name] = factor_reset
    
    if not factors_data_ready:
        raise ValueError("没有有效的因子数据")
    
    print(f"数据准备完成")
    print(f"行情数据: {data_reset.shape}")
    print(f"因子数据数量: {len(factors_data_ready)}")
    
    return data_reset, factors_data_ready

def main(factor_names=None):
    """主函数"""
    if factor_names is None:
        factor_names = ['beta', 'book_to_price', 'earnings_yield', 
                       'growth', 'leverage', 'liquidity', 'momentum', 
                       'residual_volatility', 'size', 'str']
    
    print("=" * 60)
    print(f"多因子共线性分析")
    print(f"分析因子: {', '.join(factor_names)}")
    print("=" * 60)
    
    try:
        # 加载数据
        data, factors_data = load_data(factor_names)
        
        # 准备数据
        returns_data, factors_data_ready = prepare_data(data, factors_data)
        
        if returns_data is None or not factors_data_ready:
            print("数据准备失败，请检查数据格式")
            return
        
        # 显示数据基本信息
        print("\n数据基本信息:")
        print(f"时间范围: {returns_data['date'].min()} 到 {returns_data['date'].max()}")
        print(f"股票数量: {returns_data['order_book_id'].nunique()}")
        print(f"有效因子数量: {len(factors_data_ready)}")
        
        # 进行共线性分析
        print("\n开始多因子共线性分析...")
        
        # 准备图片保存路径
        image_save_path = f"../测试结果/多因子共线性分析_分析结果.png"
        
        results = analyze_collinearity(
            factors_data=factors_data_ready,
            returns_data=returns_data,
            rebalance_period=1,
            save_path=image_save_path,
            n_jobs=4,  # 使用默认的并行线程数（CPU核心数-1）
            use_parallel=False  # 使用串行计算以避免内存问题
        )
        
        print("\n共线性分析完成！")
        
        # 保存结果
        output_dir = f"../测试结果/多因子共线性分析结果"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存beta序列数据
        beta_df = results['beta_df']
        beta_df.to_csv(f"{output_dir}/多因子beta序列.csv")
        
        # 保存因子收益率相关性矩阵
        beta_corr = results['beta_corr']
        beta_corr.to_csv(f"{output_dir}/多因子收益率相关性矩阵.csv")
        
        # 保存截面因子值相关性均值矩阵
        factor_corr_mean = results['factor_corr_mean']
        factor_corr_mean.to_csv(f"{output_dir}/多因子截面相关性均值矩阵.csv")
        
        # 保存累积相关性序列
        cum_corr_cumsum = results['cum_corr_cumsum']
        cum_corr_cumsum.to_csv(f"{output_dir}/多因子累积相关性序列.csv")
        
        # 保存原始相关性序列
        cum_corr_df = results['cum_corr_df']
        cum_corr_df.to_csv(f"{output_dir}/多因子相关性序列.csv")
        
        print(f"\n结果已保存到: {output_dir}")
        print(f"图表已保存到: {image_save_path}")
        
        # 打印简要总结
        print("\n" + "="*60)
        print("多因子共线性分析简要总结")
        print("="*60)
        
        # 因子收益率相关性排名
        if not beta_corr.empty:
            print("\n因子收益率相关性分析:")
            # 计算平均相关性（排除对角线）
            avg_corr = []
            for factor in beta_corr.index:
                corr_values = beta_corr.loc[factor].drop(factor)
                if len(corr_values) > 0:
                    avg_corr.append((factor, corr_values.mean()))
            
            avg_corr.sort(key=lambda x: x[1], reverse=True)
            print("因子收益率平均相关性排名:")
            for i, (factor, corr) in enumerate(avg_corr, 1):
                print(f"{i:2d}. {factor}: {corr:.4f}")
        
        # 截面因子值相关性排名
        if not factor_corr_mean.empty:
            print("\n截面因子值相关性分析:")
            # 计算平均相关性（排除对角线）
            avg_corr = []
            for factor in factor_corr_mean.index:
                corr_values = factor_corr_mean.loc[factor].drop(factor)
                if len(corr_values) > 0:
                    avg_corr.append((factor, corr_values.mean()))
            
            avg_corr.sort(key=lambda x: x[1], reverse=True)
            print("截面因子值平均相关性排名:")
            for i, (factor, corr) in enumerate(avg_corr, 1):
                print(f"{i:2d}. {factor}: {corr:.4f}")
        
        # 因子对相关性统计
        if not cum_corr_df.empty:
            print("\n因子对相关性统计:")
            pair_stats = []
            for col in cum_corr_df.columns:
                corr_series = cum_corr_df[col].dropna()
                if len(corr_series) > 0:
                    pair_stats.append((col, corr_series.mean(), corr_series.std()))
            
            pair_stats.sort(key=lambda x: x[1], reverse=True)
            print("因子对平均相关性排名:")
            for i, (pair, mean_corr, std_corr) in enumerate(pair_stats, 1):
                print(f"{i:2d}. {pair}: {mean_corr:.4f} ± {std_corr:.4f}")
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()  # 不写就是分析所有因子