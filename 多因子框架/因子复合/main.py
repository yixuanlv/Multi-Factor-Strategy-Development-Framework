import os
from factor_combination import run_complete_factor_analysis

work_dir = os.path.dirname(os.path.abspath(__file__))
print(f"工作目录: {work_dir}")

def main(factor_names=None, N_values=None, M_values=None, methods=None, start_year=2024):
    """主函数 - 使用统一的因子复合分析接口"""
    print("=" * 60)
    print("因子复合分析启动")
    print("=" * 60)
    
    # 使用factor_combination.py中的统一分析函数
    results = run_complete_factor_analysis(
        work_dir=work_dir,
        factor_names=factor_names,
        N_values=N_values,
        M_values=M_values,
        methods=methods,
        rebalance_period=1,
        start_year=start_year
    )
    
    return results

if __name__ == "__main__":
    # 可以自定义参数
    # main(factor_names=['beta', 'size', 'momentum'], N_values=[30, 60], M_values=[30, 60])
    # main(methods=['univariate', 'ic'])  # 只分析一元回归和IC加权方法
    # main(methods=['ranking'])  # 只分析排序加权方法
    
    # 测试运行 - 只分析多元回归加权方法，使用24年以后的数据
    main(
        factor_names=None,
        methods=['multivariate'],
        
    )
    
    # 使用默认参数，分析所有方法
    # main()
