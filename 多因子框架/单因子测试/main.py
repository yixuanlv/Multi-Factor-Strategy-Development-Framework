import pandas as pd
import numpy as np
import pickle
import os
from single_factor_analysis import SingleFactorAnalyzer, analyze_single_factor
work_dir = os.path.dirname(os.path.abspath(__file__))
print(work_dir)
def load_data(factor_name):
    """加载行情数据和因子数据"""
    print("正在加载数据...")
    
    # 加载行情数据
    data_path = os.path.join(work_dir, "../行情数据库/data.pkl")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    print(f"行情数据加载完成，数据形状: {data.shape}")
    print(f"行情数据列名: {data.columns.tolist()[:10]}")
    
    # 加载因子数据
    factor_path = os.path.join(work_dir, f"../因子库/{factor_name}.pkl")
    with open(factor_path, 'rb') as f:
        factor_data = pickle.load(f)
    print(f"因子数据加载完成，数据形状: {factor_data.shape}")
    print(f"因子数据列名: {factor_data.columns.tolist()[:10]}")
    
    return data, factor_data

def prepare_data(data, factor_data):
    """准备分析所需的数据格式"""
    print("正在准备数据...")
    
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
                print("数据格式需要调整，请检查数据结构")
                return None, None
        else:
            print("请检查行情数据格式，需要包含close列")
            return None, None
    else:
        # 数据已经有close列，检查是否需要重置索引
        if isinstance(data.index, pd.MultiIndex):
            data_reset = data.reset_index()
        else:
            data_reset = data.copy()
    
    # 准备因子数据
    factor_reset = None
    if isinstance(factor_data.index, pd.MultiIndex):
        # 已经是长格式，直接重命名列
        factor_reset = factor_data.reset_index()
        if len(factor_reset.columns) == 3:
            factor_reset.columns = ['date', 'order_book_id', 'factor_value']
        else:
            print("因子数据格式需要调整，请检查数据结构")
            return None, None
    elif isinstance(factor_data, pd.DataFrame) and not isinstance(factor_data.index, pd.MultiIndex):
        # 宽格式，行为日期，列为股票代码
        factor_reset = factor_data.stack().reset_index()
        factor_reset.columns = ['date', 'order_book_id', 'factor_value']
    else:
        print("请检查因子数据格式")
        return None, None
    
    print(f"数据准备完成")
    print(f"行情数据: {data_reset.shape}")
    print(f"因子数据: {factor_reset.shape}")
    
    return data_reset, factor_reset

def main(factor_name):
    """主函数"""
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
        data, factor_data = load_data(factor_name)
        
        # 准备数据
        returns_data, factor_data_ready = prepare_data(data, factor_data)
        
        if returns_data is None or factor_data_ready is None:
            print("数据准备失败，请检查数据格式")
            return
        
        # 显示数据基本信息
        print("\n数据基本信息:")
        print(f"时间范围: {returns_data['date'].min()} 到 {returns_data['date'].max()}")
        print(f"股票数量: {returns_data['order_book_id'].nunique()}")
        print(f"数据点数量: {len(returns_data)}")
        
        # 进行单因子分析
        print("\n开始单因子分析...")
        
        # 准备图片保存路径
        image_save_path = os.path.join(factor_results_dir, f"{factor_name}_full_analysis.png")
        
        results = analyze_single_factor(
            factor_data=factor_data_ready,
            returns_data=returns_data,
            factor_name=factor_name,
            n_groups=10,
            method='spearman',
            rebalance_period=1,
            save_path=image_save_path
        )
        
        print("\n分析完成！")
        
        # 保存结果到因子专用文件夹
        output_dir = factor_results_dir
        
        # 保存IC统计
        ic_stats = results['ic_stats']
        ic_df = pd.DataFrame([ic_stats])
        ic_df.to_csv(os.path.join(output_dir, f"{factor_name}_ic_stats.csv"), index=False)
        
        # 保存多空组合统计
        ls_stats = results['long_short_stats']
        ls_df = pd.DataFrame([ls_stats])
        ls_df.to_csv(os.path.join(output_dir, f"{factor_name}_long_short_stats.csv"), index=False)
        
        # 保存分组收益率
        group_returns = results['group_returns']['group_returns']
        group_returns.to_csv(os.path.join(output_dir, f"{factor_name}_group_returns.csv"))
        
        # 保存累计收益率
        cumulative_returns = results['cumulative_returns']
        cumulative_returns.to_csv(os.path.join(output_dir, f"{factor_name}_cumulative_returns.csv"))
        
        # 保存多空收益率
        ls_returns = results['long_short_returns']['long_short_returns']
        ls_returns.to_csv(os.path.join(output_dir, f"{factor_name}_long_short_returns.csv"))
        
        print(f"\n结果已保存到: {output_dir}")
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main( factor_name='volatility_250_factor')
