import os
import pandas as pd
import numpy as np
from factor_combination import FactorCombiner

work_dir = os.path.dirname(os.path.abspath(__file__))
print(f"工作目录: {work_dir}")



def load_data():
    """加载因子数据和价格数据（均为pkl文件）"""
    data = {}

    # 价格数据路径
    price_file = r"C:\Users\9shao\Desktop\github公开项目\Multi-Factor-Strategy-Development-Framework\多因子框架\行情数据库\data.pkl"
    try:
        prices = pd.read_pickle(price_file)
        data['prices'] = prices
        
    except Exception as e:
        print(f"加载价格数据时出错: {e}")
        return None

    # 因子数据目录
    factor_dir = r"C:\Users\9shao\Desktop\github公开项目\Multi-Factor-Strategy-Development-Framework\多因子框架\因子库"
    if not os.path.exists(factor_dir):
        print(f"因子库目录不存在: {factor_dir}")
        return None

    factor_files = [f for f in os.listdir(factor_dir) if f.endswith('.pkl')]
    if not factor_files:
        print("未找到任何因子pkl文件")
        return None

    for fname in factor_files:
        factor_name = os.path.splitext(fname)[0]
        factor_path = os.path.join(factor_dir, fname)
        try:
            df = pd.read_pickle(factor_path)
            data[factor_name] = df
            
        except Exception as e:
            print(f"加载因子 {factor_name} 时出错: {e}")

    if len(data) <= 1:
        print("未能成功加载任何因子数据")
        return None

    return data

def run_factor_combination_analysis(factor_names=None, N_values=None, methods=None, rebalance_period=1, save_combined_factors=True):
    """运行因子复合分析"""
    print("=" * 60)
    print("因子复合分析")
    print("=" * 60)
    
    # 加载数据
    print("\n1. 加载数据...")
    data = load_data()
    
    if not data:
        print("   ✗ 数据加载失败")
        return None
    
    # 分离因子数据和价格数据
    all_factors = {k: v for k, v in data.items() if k != 'prices'}
    prices = data['prices']
    
    if not all_factors:
        print("   ✗ 没有可用的因子数据")
        return None
    
    # 处理factor_names参数
    if factor_names is None:
        # 如果未指定factor_names，则使用所有可用的因子
        factor_names = list(all_factors.keys())
        print(f"未指定因子名，使用所有可用因子: {factor_names}")
    else:
        # 检查指定的因子是否都存在
        available_factors = list(all_factors.keys())
        missing_factors = [f for f in factor_names if f not in available_factors]
        if missing_factors:
            print(f"警告: 以下因子不存在: {missing_factors}")
            print(f"可用因子: {available_factors}")
            return None
        
        # 只保留指定的因子
        all_factors = {k: v for k, v in all_factors.items() if k in factor_names}
        print(f"使用指定因子: {factor_names}")
    
    if N_values is None:
        N_values = [30, 60, 120]
    if methods is None:
        methods = ['univariate', 'multivariate', 'rank_ic']
    
    print(f"分析因子: {factor_names}")
    print(f"滚动窗口: {N_values}")
    print(f"权重方法: {methods}")
    print(f"调仓周期: {rebalance_period}")
    
    # 使用筛选后的因子数据
    factors = all_factors
    
    # 运行分析
    print("\n2. 运行因子复合分析...")
    results = {}
    combined_factors_data = {}  # 存储复合因子数据
    
    for method in methods:
        print(f"\n   分析方法: {method}")
        for N in N_values:
            print(f"     滚动窗口 N={N}...")
            
            try:
                # 创建因子复合器
                combiner = FactorCombiner(
                    factors=factors,
                    prices=prices,
                    rebalance_period=rebalance_period
                )
                
                # 运行分析
                result = combiner.build(method=method, N=N)
                results[f"{method}_N{N}"] = result
                
                # 保存复合因子数据
                if save_combined_factors:
                    combined_factors_data[f"{method}_N{N}"] = {
                        'combined_factor': result.combined_factor,
                        'weight_history': result.weight_history,
                        'method': method,
                        'N': N,
                        'rebalance_period': rebalance_period,
                        'factor_names': list(factors.keys()),
                        'summary': result.summary
                    }
                
                print(f"        ✓ 完成 - IC均值: {result.summary['ic_mean']:.4f}, 夏普: {result.summary['sharpe_ratio']:.2f}")
                
            except Exception as e:
                print(f"        ✗ 失败: {e}")
                continue
    
    # 保存复合因子到pkl文件
    if save_combined_factors and combined_factors_data:
        print("\n3. 保存复合因子到pkl文件...")
        try:
            # 创建复合因子保存目录
            combined_factor_dir = os.path.join(work_dir, "../因子库/复合因子库")
            os.makedirs(combined_factor_dir, exist_ok=True)
            
            # 保存每个方法的复合因子
            for key, data in combined_factors_data.items():
                pkl_filename = f"combined_factor_{key}.pkl"
                pkl_path = os.path.join(combined_factor_dir, pkl_filename)
                
                # 保存到pkl文件
                pd.to_pickle(data, pkl_path)
            print(f"   ✓ 所有复合因子已保存到目录: {combined_factor_dir}")
            
        except Exception as e:
            print(f"   ✗ 复合因子保存失败: {e}")
    
    # 生成报告
    if results:
        print("\n4. 生成分析报告...")
        try:
            html_path = os.path.join(work_dir, "因子复合分析报告.html")
            combiner.render_report(results, html_path)
            print(f"   ✓ 报告已生成: {html_path}")
        except Exception as e:
            print(f"   ✗ 报告生成失败: {e}")
    
    print("\n" + "=" * 60)
    print("因子复合分析完成")
    print("=" * 60)
    
    return results, combined_factors_data

def main(factor_names=None, N_values=None, methods=None, rebalance_period=1, save_combined_factors=True):
    """主函数 - 运行因子复合分析"""
    try:
        print("开始执行因子复合分析...")
        result = run_factor_combination_analysis(
            factor_names=factor_names,
            N_values=N_values,
            methods=methods,
            rebalance_period=rebalance_period,
            save_combined_factors=save_combined_factors
        )
        print("因子复合分析执行完成！")
        return result
    except Exception as e:
        import traceback
        print(f"程序执行过程中发生错误: {e}")
        print("详细错误信息:")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # 示例：运行因子复合分析，设置调仓周期为5天
    # 方式1：指定特定因子（取消注释并修改因子名）
    # main(
    #     factor_names=['factor_1', 'factor_2'],  # 修改为实际存在的因子名
    #     N_values=[20, 60, 120], 
    #     methods=['univariate', 'multivariate', 'rank_ic'],
    #     rebalance_period=5,  # 每5个交易日调仓一次
    #     save_combined_factors=True  # 保存复合因子到pkl文件
    # )
    

    try:
        print("程序开始执行...")
        main(
            factor_names=['beta', 'size', 'momentum'],  # None表示使用所有可用因子
            N_values=[20, 60, 120], 
            methods=['univariate', 'multivariate', 'rank_ic'],
            rebalance_period=5,  # 每5个交易日调仓一次
            save_combined_factors=True  # 保存复合因子到pkl文件
        )
        print("程序执行完成！")
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        import traceback
        print(f"程序执行失败: {e}")
        print("详细错误信息:")
        traceback.print_exc()
