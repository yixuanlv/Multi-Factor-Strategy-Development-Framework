import pandas as pd
import numpy as np
import pickle
import os
from multi_factor_analysis import MultiFactorAnalyzer
import sys
work_dir = os.path.dirname(os.path.abspath(__file__))
print(f"工作目录: {work_dir}")

def get_all_factor_names():
    """自动获取因子库下所有pkl文件名作为因子名"""
    factor_dir = os.path.join(work_dir, "../因子库")
    if not os.path.exists(factor_dir):
        print(f"因子库文件夹不存在: {factor_dir}")
        return []
    
    factor_names = [os.path.splitext(fname)[0] for fname in os.listdir(factor_dir) 
                   if fname.endswith('.pkl')]
    print(f"找到 {len(factor_names)} 个因子文件")
    return factor_names

def load_data(factor_names):
    """加载行情数据和多个因子数据"""
    print("正在加载数据...")
    
    # 加载行情数据
    data_path = os.path.join(work_dir, "../行情数据库/data.pkl")
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        print(f"行情数据加载完成，数据形状: {data.shape}")
    except Exception as e:
        print(f"无法加载行情数据: {e}")
        raise
    
    # 加载因子数据
    factors_data = {}
    for factor_name in factor_names:
        factor_path = os.path.join(work_dir, f"../因子库/{factor_name}.pkl")
        try:
            with open(factor_path, 'rb') as f:
                factor_data = pickle.load(f)
            factors_data[factor_name] = factor_data
            print(f"因子 {factor_name} 数据加载完成，数据形状: {factor_data.shape}")
        except Exception as e:
            print(f"警告：因子 {factor_name} 数据加载失败: {e}")
            continue
    
    if not factors_data:
        raise ValueError("没有找到任何有效的因子数据文件")
    
    print(f"成功加载 {len(factors_data)} 个因子数据")
    return data, factors_data

def prepare_data(data, factors_data):
    """准备分析所需的数据格式"""
    print("正在准备数据...")
    
    # 处理行情数据
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
        data_reset = data.reset_index() if isinstance(data.index, pd.MultiIndex) else data.copy()
    
    # 准备因子数据
    factors_data_ready = {}
    for factor_name, factor_data in factors_data.items():
        if isinstance(factor_data.index, pd.MultiIndex):
            factor_reset = factor_data.reset_index()
            if len(factor_reset.columns) == 3:
                factor_reset.columns = ['date', 'order_book_id', 'factor_value']
            else:
                print(f"因子 {factor_name} 数据格式需要调整，跳过")
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
    
    print(f"数据准备完成，行情数据: {data_reset.shape}，因子数据数量: {len(factors_data_ready)}")
    return data_reset, factors_data_ready

def main(analysis_type='both', factor_names=None):
    """主函数 - 支持多因子测试和共线性分析"""
    if factor_names is None:
        factor_names = get_all_factor_names()
    
    print("=" * 60)
    print(f"多因子综合分析 - 分析类型: {analysis_type}")
    print(f"分析因子: {', '.join(factor_names)}")
    print("=" * 60)
    
    try:
        # 加载和准备数据
        data, factors_data = load_data(factor_names)
        returns_data, factors_data_ready = prepare_data(data, factors_data)
        
        if returns_data is None or not factors_data_ready:
            print("数据准备失败，请检查数据格式")
            return
        
        # 显示数据基本信息
        print(f"\n数据基本信息:")
        print(f"时间范围: {returns_data['date'].min()} 到 {returns_data['date'].max()}")
        print(f"股票数量: {returns_data['order_book_id'].nunique()}")
        print(f"有效因子数量: {len(factors_data_ready)}")
        
        # 创建分析器并执行分析
        analyzer = MultiFactorAnalyzer(factors_data_ready, returns_data, rebalance_period=1)
        
        if analysis_type in ['both', 'testing']:
            print("\n开始多因子集中测试分析...")
            output_dir = os.path.join(work_dir, "../../测试结果/多因子集中测试结果")
            os.makedirs(output_dir, exist_ok=True)
            
            testing_results = analyzer.generate_comprehensive_report(
                n_groups=10, method='spearman', 
                save_path=os.path.join(output_dir, "多因子集中测试_分析结果.html")
            )
            save_testing_results(testing_results, output_dir)
            print(f"多因子测试结果已保存到: {output_dir}")
        
        if analysis_type in ['both', 'collinearity']:
            print("\n开始多因子共线性分析...")
            output_dir = os.path.join(work_dir, "../../测试结果/多因子共线性分析")
            os.makedirs(output_dir, exist_ok=True)
            
            collinearity_results = analyzer.generate_collinearity_report(
                save_path=os.path.join(output_dir, "多因子共线性分析_分析结果.html")
            )
            save_collinearity_results(collinearity_results, output_dir)
            print(f"共线性分析结果已保存到: {output_dir}")
        
        print("\n所有分析完成！")
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

def save_testing_results(results, output_dir):
    """保存多因子测试结果"""
    # 保存IC数据
    results['ic_df'].to_csv(f"{output_dir}/多因子IC序列.csv")
    results['cum_ic_df'].to_csv(f"{output_dir}/多因子累计IC序列.csv")
    results['stats_df'].to_csv(f"{output_dir}/多因子统计指标.csv")
    
    # 保存分组收益率数据
    for factor_name, group_data in results['group_returns_data'].items():
        group_data['group_returns'].to_csv(f"{output_dir}/{factor_name}_分组收益率.csv")
    
    # 保存多空组合数据
    for factor_name, ls_data in results['long_short_data'].items():
        ls_data['ls_1_10'].to_csv(f"{output_dir}/{factor_name}_第1层倒1层多空收益率.csv")
        ls_data['ls_2_9'].to_csv(f"{output_dir}/{factor_name}_第2层倒2层多空收益率.csv")
    
    # 保存多头超额收益率数据
    for factor_name, excess_data in results['long_excess_data'].items():
        excess_data['excess_1'].to_csv(f"{output_dir}/{factor_name}_第1层多头超额收益率.csv")
        excess_data['excess_2'].to_csv(f"{output_dir}/{factor_name}_第2层多头超额收益率.csv")
    
    # 保存性能统计指标汇总
    performance_summary = []
    for factor_name, stats in results['performance_stats'].items():
        summary = {
            '因子名称': factor_name,
            '第1层倒1层多空年化收益': stats['ls_1_10']['annual_return'],
            '第1层倒1层多空夏普比率': stats['ls_1_10']['sharpe_ratio'],
            '第2层倒2层多空年化收益': stats['ls_2_9']['annual_return'],
            '第2层倒2层多空夏普比率': stats['ls_2_9']['sharpe_ratio'],
            '第1层多头超额年化收益': stats['excess_1']['annual_return'],
            '第1层多头超额夏普比率': stats['excess_1']['sharpe_ratio'],
            '第2层多头超额年化收益': stats['excess_2']['annual_return'],
            '第2层多头超额夏普比率': stats['excess_2']['sharpe_ratio']
        }
        performance_summary.append(summary)
    
    pd.DataFrame(performance_summary).to_csv(f"{output_dir}/多因子性能对比汇总.csv", index=False)

def save_collinearity_results(results, output_dir):
    """保存共线性分析结果"""
    results['beta_df'].to_csv(os.path.join(output_dir, "多因子beta序列.csv"))
    results['beta_corr'].to_csv(os.path.join(output_dir, "多因子收益率相关性矩阵.csv"))
    results['factor_corr_mean'].to_csv(os.path.join(output_dir, "多因子截面相关性均值矩阵.csv"))
    results['cum_corr_cumsum'].to_csv(os.path.join(output_dir, "多因子累积相关性序列.csv"))
    results['cum_corr_df'].to_csv(os.path.join(output_dir, "多因子相关性序列.csv"))

if __name__ == "__main__":
    
    
    # 解析命令行参数
    analysis_type = 'testing'
    factor_list = None

    # 执行分析
    if factor_list:
        print(f"分析指定因子: {factor_list}")
        main(analysis_type, factor_list)
    else:
        print("分析所有可用因子")
        main(analysis_type)
