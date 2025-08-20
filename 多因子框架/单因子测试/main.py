import pandas as pd
import numpy as np
import pickle
import os
import importlib.util
import sys
import single_factor_analysis


SingleFactorAnalyzer = single_factor_analysis.SingleFactorAnalyzer
analyze_single_factor = single_factor_analysis.analyze_single_factor
work_dir = os.path.dirname(os.path.abspath(__file__))
def load_data(factor_name):
    """加载行情数据和因子数据"""

    
    # 加载行情数据
    data_path = os.path.join(work_dir, "../行情数据库/data.pkl")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    
    # 加载因子数据
    factor_path = os.path.join(work_dir, f"../因子库/{factor_name}.pkl")
    with open(factor_path, 'rb') as f:
        factor_data = pickle.load(f)

    
    # 加载Barra因子数据
    barra_path = os.path.join(work_dir, "../因子库/Barra因子/barra.pkl")
    barra_data = None
    try:
        with open(barra_path, 'rb') as f:
            barra_data = pickle.load(f)

    except Exception as e:
        pass
    
    return data, factor_data, barra_data

def prepare_data(data, factor_data):
    """准备分析所需的数据格式"""

    
    # 处理行情数据
    data_reset = None
    if 'close' not in data.columns:
        # 如果数据是宽格式，需要转换为长格式
        if isinstance(data.index, pd.MultiIndex):
            # 假设索引是(date, order_book_id)
            data_reset = data.reset_index()
            if len(data_reset.columns) == 3:  # date, order_book_id, close
                data_reset.columns = ['date', 'order_book_id', 'close']
            else:
                # 需要进一步处理
                return None, None
        else:
            return None, None
    else:
        # 数据已经有close列，检查是否需要重置索引
        if isinstance(data.index, pd.MultiIndex):
            data_reset = data.reset_index()
        else:
            data_reset = data.copy()
    
    # 添加必要的过滤标志列（如果不存在）
    required_flags = ['limit_up_flag', 'limit_down_flag', 'ST', 'suspended']
    for flag in required_flags:
        if flag not in data_reset.columns:

            data_reset[flag] = 0  # 默认值设为0（不过滤）
    
    # 准备因子数据
    factor_reset = None
    if isinstance(factor_data.index, pd.MultiIndex):
        # 已经是长格式，直接重命名列
        factor_reset = factor_data.reset_index()
        if len(factor_reset.columns) == 3:
            factor_reset.columns = ['date', 'order_book_id', 'factor_value']
        else:
            return None, None
    elif isinstance(factor_data, pd.DataFrame) and not isinstance(factor_data.index, pd.MultiIndex):
        # 宽格式，行为日期，列为股票代码
        factor_reset = factor_data.stack().reset_index()
        factor_reset.columns = ['date', 'order_book_id', 'factor_value']
    else:
        return None, None
    

    
    return data_reset, factor_reset

def run_single_factor_analysis(factor_name,rebalance_period):  
    """执行单因子分析并保存结果"""
    print("=" * 60)
    print(f"单因子分析 - {factor_name} 因子")
    print("=" * 60)
    try:
        # 创建测试结果主文件夹
        test_results_dir = os.path.join(work_dir, "../测试结果")
        os.makedirs(test_results_dir, exist_ok=True)

        # 为当前因子创建独立的子文件夹
        factor_results_dir = os.path.join(test_results_dir, factor_name)
        os.makedirs(factor_results_dir, exist_ok=True)

        # 加载数据
        data, factor_data, barra_data = load_data(factor_name)

        # 准备数据
        returns_data, factor_data_ready = prepare_data(data, factor_data)

        if returns_data is None or factor_data_ready is None:
            print("数据准备失败，请检查数据格式")
            return



        # 准备图片保存路径
        image_save_path = os.path.join(factor_results_dir, f"{factor_name}_full_analysis.html")

        results = analyze_single_factor(
            factor_data=factor_data_ready,
            returns_data=returns_data,
            factor_name=factor_name,
            n_groups=10,
            method='spearman',
            rebalance_period=rebalance_period,
            save_path=image_save_path,
            enable_stock_filter=True,
            barra_data=barra_data
        )



        # 保存结果到因子专用文件夹
        output_dir = factor_results_dir

        # 保存IC统计
        ic_stats = results['ic_stats']
        ic_df = pd.DataFrame([ic_stats])
        ic_df.to_csv(os.path.join(output_dir, f"{factor_name}_ic_stats.csv"), index=False)

        # 保存多空组合统计
        long_short_stats = results['long_short_stats']
        ls_stats_df = pd.DataFrame([long_short_stats])
        ls_stats_df.to_csv(os.path.join(output_dir, f"{factor_name}_long_short_stats.csv"), index=False)

        # 保存分组收益率时间序列
        group_returns = results['group_returns']['group_returns']
        group_returns.to_csv(os.path.join(output_dir, f"{factor_name}_group_returns.csv"))

        # 保存累计收益率时间序列
        cumulative_returns = results['cumulative_returns']
        cumulative_returns.to_csv(os.path.join(output_dir, f"{factor_name}_cumulative_returns.csv"))

        # 保存多空组合收益率时间序列
        long_short_returns = results['long_short_returns']['long_short_returns']
        long_short_returns.to_csv(os.path.join(output_dir, f"{factor_name}_long_short_returns.csv"))

        # 保存多空组合累计收益率时间序列
        cumulative_ls_returns = results['long_short_returns']['cumulative_ls_returns']
        cumulative_ls_returns.to_csv(os.path.join(output_dir, f"{factor_name}_cumulative_ls_returns.csv"))

        # 打印结果摘要
        print(f"\n=== {factor_name} 因子分析结果摘要 ===")
        print(f"IC统计:")
        print(f"  IC均值: {ic_stats['IC_mean']:.4f}")
        print(f"  IC标准差: {ic_stats['IC_std']:.4f}")
        print(f"  ICIR: {ic_stats['ICIR']:.4f}")
        print(f"  IC正比例: {ic_stats['IC_positive_ratio']:.2%}")
        print(f"  IC偏度: {ic_stats['IC_skew']:.4f}")
        print(f"  IC峰度: {ic_stats['IC_kurtosis']:.4f}")
        print(f"  t值: {ic_stats['IC_tvalue']:.4f}")
        print(f"  p值: {ic_stats['IC_pvalue']:.4g}")
        
        print(f"\n多空组合统计:")
        print(f"  平均收益: {long_short_stats['mean_return']:.4f}")
        print(f"  收益率标准差: {long_short_stats['std_return']:.4f}")
        print(f"  夏普比率: {long_short_stats['sharpe_ratio']:.4f}")
        print(f"  胜率: {long_short_stats['win_rate']:.2%}")
        print(f"  最大回撤: {long_short_stats['max_drawdown']:.2%}")
        print(f"  总收益: {long_short_stats['total_return']:.2%}")



    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

def main(factor_name,rebalance_period):
    """主函数，支持批量分析所有因子"""
    if factor_name is None:
        factor_dir = os.path.join(work_dir, "../因子库")
        all_files = os.listdir(factor_dir)
        factor_files = [f for f in all_files if f.endswith('.pkl')]
        if not factor_files:
            return
        for f in factor_files:
            factor_name_single = os.path.splitext(f)[0]
            run_single_factor_analysis(factor_name_single,rebalance_period)
    else:
        run_single_factor_analysis(factor_name,rebalance_period)

if __name__ == "__main__":
    main(factor_name='residual_5day_all_factor',rebalance_period = 1)
