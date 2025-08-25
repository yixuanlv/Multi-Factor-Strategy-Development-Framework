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

def run_factor_combination_analysis(factor_names=None, N_values=None, methods=None):
    """运行因子复合分析"""
    print("=" * 60)
    print("因子复合分析")
    print("=" * 60)
    
    # 如果未指定factor_names，则自动读取指定目录下所有pkl文件名作为因子名
    if factor_names is None:
        factor_dir = r"C:\Users\9shao\Desktop\github公开项目\Multi-Factor-Strategy-Development-Framework\多因子框架\因子库"
        factor_names = [os.path.splitext(f)[0] for f in os.listdir(factor_dir) if f.endswith('.pkl')]
        print(f"自动读取因子库目录，获得因子名: {factor_names}")
    if N_values is None:
        N_values = [30, 60, 120]
    if methods is None:
        methods = ['univariate', 'multivariate', 'rank_ic']
    
    print(f"分析因子: {factor_names}")
    print(f"滚动窗口: {N_values}")
    print(f"权重方法: {methods}")
    
    # 加载数据
    print("\n1. 加载数据...")
    data = load_data()
    
    if not data:
        print("   ✗ 数据加载失败")
        return None
    
    # 分离因子数据和价格数据
    factors = {k: v for k, v in data.items() if k != 'prices'}
    prices = data['prices']
    
    if not factors:
        print("   ✗ 没有可用的因子数据")
        return None
    
    # 运行分析
    print("\n2. 运行因子复合分析...")
    results = {}
    
    for method in methods:
        print(f"\n   分析方法: {method}")
        for N in N_values:
            print(f"     滚动窗口 N={N}...")
            
            try:
                # 创建因子复合器
                combiner = FactorCombiner(
                    factors=factors,
                    prices=prices,
                    rebalance_period=1
                )
                
                # 运行分析
                result = combiner.build(method=method, N=N)
                results[f"{method}_N{N}"] = result
                
                print(f"        ✓ 完成 - IC均值: {result.summary['ic_mean']:.4f}, 夏普: {result.summary['sharpe_ratio']:.2f}")
                
            except Exception as e:
                print(f"        ✗ 失败: {e}")
                continue
    
    # 生成报告
    if results:
        print("\n3. 生成分析报告...")
        try:
            html_path = os.path.join(work_dir, "因子复合分析报告.html")
            combiner.render_report(results, html_path)
            print(f"   ✓ 报告已生成: {html_path}")
        except Exception as e:
            print(f"   ✗ 报告生成失败: {e}")
    
    print("\n" + "=" * 60)
    print("因子复合分析完成")
    print("=" * 60)
    
    return results

def main(factor_names=None, N_values=None, methods=None):
    """主函数 - 运行因子复合分析"""
    return run_factor_combination_analysis(
        factor_names=factor_names,
        N_values=N_values,
        methods=methods
    )

if __name__ == "__main__":

    main(factor_names=['beta', 'size', 'momentum'], N_values=[60], methods=['univariate'])
