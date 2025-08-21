import pandas as pd
import numpy as np
import pickle
import os
from multi_factor_analysis import MultiFactorAnalyzer, analyze_multiple_factors
from numpy_compat import safe_pickle_load

work_dir = os.path.dirname(os.path.abspath(__file__))
print(work_dir)

def load_data(factor_names):
    """加载行情数据和多个因子数据"""
    print("正在加载数据...")
    
    # 加载行情数据
    data_path = os.path.join(work_dir, "../行情数据库/data.pkl")
    try:
        data = safe_pickle_load(data_path)
        print(f"行情数据加载完成，数据形状: {data.shape}")
    except Exception as e:
        print(f"无法加载行情数据: {e}")
        raise
    
    # 加载多个因子数据
    factors_data = {}
    for factor_name in factor_names:
        factor_path = os.path.join(work_dir, f"../因子库/{factor_name}.pkl")
        try:
            factor_data = safe_pickle_load(factor_path)
            factors_data[factor_name] = factor_data
            print(f"因子 {factor_name} 数据加载完成，数据形状: {factor_data.shape}")
        except FileNotFoundError:
            print(f"警告：因子 {factor_name} 数据文件不存在，跳过")
            continue
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
        # 自动读取指定目录下所有pkl后缀且大小在50MB~500MB之间的文件名作为因子名
        import os
        factor_dir = r"C:\Users\9shao\Desktop\github公开项目\Multi-Factor-Strategy-Development-Framework\多因子框架\因子库"
        factor_names = []
        for fname in os.listdir(factor_dir):
            if fname.endswith('.pkl'):
                fpath = os.path.join(factor_dir, fname)
                try:
                    fsize = os.path.getsize(fpath)
                    if 50 * 1024 * 1024 < fsize < 500 * 1024 * 1024:
                        factor_names.append(os.path.splitext(fname)[0])
                except Exception as e:
                    print(f"文件 {fname} 读取大小出错: {e}")
        if not factor_names:
            print("未找到符合条件的因子文件")
            return
    print("=" * 60)
    print(f"多因子集中测试分析")
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
        
        # 进行多因子分析
        print("\n开始多因子集中测试分析...")
        
        # 所有输出文件都放在 测试结果/多因子集中测试结果 目录下
        output_dir = os.path.join(work_dir, "../测试结果/多因子集中测试结果")
        os.makedirs(output_dir, exist_ok=True)
        html_save_path = os.path.join(output_dir, "多因子集中测试_分析结果.html")
        
        results = analyze_multiple_factors(
            factors_data=factors_data_ready,
            returns_data=returns_data,
            n_groups=10,
            method='spearman',
            rebalance_period=1,
            save_path=html_save_path
        )
        
        print("\n多因子分析完成！")
        
        # 保存IC数据
        ic_df = results['ic_df']
        ic_df.to_csv(f"{output_dir}/多因子IC序列.csv")
        
        # 保存累计IC数据
        cum_ic_df = results['cum_ic_df']
        cum_ic_df.to_csv(f"{output_dir}/多因子累计IC序列.csv")
        
        # 保存统计指标
        stats_df = results['stats_df']
        stats_df.to_csv(f"{output_dir}/多因子统计指标.csv")
        
        # 保存分组收益率数据
        group_returns_data = results['group_returns_data']
        for factor_name, group_data in group_returns_data.items():
            group_returns = group_data['group_returns']
            group_returns.to_csv(f"{output_dir}/{factor_name}_分组收益率.csv")
        
        # 保存多空组合数据
        long_short_data = results['long_short_data']
        for factor_name, ls_data in long_short_data.items():
            ls_1_10 = ls_data['ls_1_10']
            ls_1_10.to_csv(f"{output_dir}/{factor_name}_第1层倒1层多空收益率.csv")
            
            ls_2_9 = ls_data['ls_2_9']
            ls_2_9.to_csv(f"{output_dir}/{factor_name}_第2层倒2层多空收益率.csv")
        
        # 保存多头超额收益率数据
        long_excess_data = results['long_excess_data']
        for factor_name, excess_data in long_excess_data.items():
            excess_1 = excess_data['excess_1']
            excess_1.to_csv(f"{output_dir}/{factor_name}_第1层多头超额收益率.csv")
            
            excess_2 = excess_data['excess_2']
            excess_2.to_csv(f"{output_dir}/{factor_name}_第2层多头超额收益率.csv")
        
        # 保存性能统计指标
        performance_stats = results['performance_stats']
        performance_summary = []
        for factor_name, stats in performance_stats.items():
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
        
        performance_df = pd.DataFrame(performance_summary)
        performance_df.to_csv(f"{output_dir}/多因子性能对比汇总.csv", index=False)
        
        print(f"\n结果已保存到: {output_dir}")
        print(f"HTML图表已保存到: {html_save_path}")
        
        # 打印简要总结
        print("\n" + "="*60)
        print("多因子分析简要总结")
        print("="*60)
        
        # IC均值排名
        ic_ranking = stats_df['IC_mean'].sort_values(ascending=False)
        print("\nIC均值排名:")
        for i, (factor, ic) in enumerate(ic_ranking.items(), 1):
            print(f"{i:2d}. {factor}: {ic:.4f}")
        
        # 多空组合年化收益排名
        ls_returns = []
        for factor_name, stats in performance_stats.items():
            ls_returns.append((factor_name, stats['ls_1_10']['annual_return']))
        ls_returns.sort(key=lambda x: x[1], reverse=True)
        
        print("\n第1层倒1层多空组合年化收益排名:")
        for i, (factor, ret) in enumerate(ls_returns, 1):
            print(f"{i:2d}. {factor}: {ret:.2%}")
        
        # 多头超额收益排名
        excess_returns = []
        for factor_name, stats in performance_stats.items():
            excess_returns.append((factor_name, stats['excess_1']['annual_return']))
        excess_returns.sort(key=lambda x: x[1], reverse=True)
        
        print("\n第1层多头超额年化收益排名:")
        for i, (factor, ret) in enumerate(excess_returns, 1):
            print(f"{i:2d}. {factor}: {ret:.2%}")
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

def analyze_specific_factors(factor_list):
    """分析指定的因子列表"""
    print(f"分析指定因子: {factor_list}")
    main(factor_list)

def analyze_all_factors():
    """分析所有可用因子"""
    print("分析所有可用因子")
    main()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 如果提供了命令行参数，使用指定的因子
        factor_list = sys.argv[1:]
        analyze_specific_factors(factor_list)
    else:
        # 否则分析所有因子
        analyze_all_factors()
