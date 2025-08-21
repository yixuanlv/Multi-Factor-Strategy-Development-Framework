import pandas as pd

# 读取回测结果
result = pd.read_pickle(r"C:\Users\9shao\Desktop\github公开项目\Multi-Factor-Strategy-Development-Framework\纯多头_优化版.pkl")

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

df