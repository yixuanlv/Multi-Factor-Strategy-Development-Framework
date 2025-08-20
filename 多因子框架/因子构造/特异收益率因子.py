import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 添加路径以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def calc_residuals_single_date(df_chunk: pd.DataFrame, date: str, used_factors: list) -> pd.DataFrame:
    """
    计算单个日期的残差
    """
    sub = df_chunk[df_chunk['date'] == date].copy()
    sub = sub.dropna(subset=['ret'] + used_factors)
    
    if len(sub) < len(used_factors) + 2:
        return pd.DataFrame()  # 数据太少返回空DataFrame
        
    X = sub[used_factors]
    y = sub['ret']
    
    
    try:
        model = sm.OLS(y, X).fit()
        sub['resid'] = model.resid
        return sub[['order_book_id', 'date', 'resid']]
    except Exception as e:
        print(f"回归异常，日期{date}，跳过。异常信息：{e}")
        return pd.DataFrame()

def calc_residuals(data: pd.DataFrame,
                   barra: pd.DataFrame,
                   n: int = 5,
                   factor_type: str = "all",
                   chunk_size: int = 100,
                   max_workers: int = 4) -> pd.DataFrame:
    """
    计算n日累计收益率残差（多线程分块处理，节省内存）
    Parameters
    ----------
    data : pd.DataFrame
        必须包含 ['order_book_id', 'date', 'close']
    barra : pd.DataFrame
        必须包含 ['order_book_id', 'date'] + 一系列因子列
    n : int
        滚动残差累计天数
    factor_type : str
        'style'   : 只用风格因子
        'industry': 只用行业因子
        'all'     : 全部因子
    chunk_size : int
        每次处理的日期数量，默认100，防止内存溢出
    max_workers : int
        最大线程数，默认4
    Returns
    -------
    pd.DataFrame
        ['order_book_id', 'date', 'resid_sum']，记录每只股票在每一天的n日累计残差
    """

    # === 1. 获取所有日期，分块处理 ===
    data_dates = set(data['date'].unique())
    barra_dates = set(barra['date'].unique())
    all_dates = sorted(list(data_dates & barra_dates))
    
    # === 2. 按因子类型选择因子列 ===
    factor_cols = [c for c in barra.columns if c not in ['order_book_id', 'date']]
    style_factors = [c for c in factor_cols if c.startswith("STYLE_")]
    industry_factors = [c for c in factor_cols if c.startswith("INDUSTRY_")]
    
    if factor_type == "style":
        used_factors = style_factors
    elif factor_type == "industry":
        used_factors = industry_factors
    else:
        used_factors = factor_cols
    
    print(f"使用的因子数量: {len(used_factors)}")
    print(f"总日期数量: {len(all_dates)}")
    print(f"使用线程数: {max_workers}")
    
    # === 3. 多线程分块处理 ===
    residuals_list = []
    
    for i in tqdm(range(0, len(all_dates), chunk_size), desc="分块横截面回归"):
        date_chunk = all_dates[i:i+chunk_size]
        
        # 取出当前块的data和barra
        data_chunk = data[data['date'].isin(date_chunk)].copy()
        barra_chunk = barra[barra['date'].isin(date_chunk)].copy()
        
        if data_chunk.empty or barra_chunk.empty:
            continue
            
        # 合并数据
        df = pd.merge(data_chunk, barra_chunk, on=['order_book_id', 'date'], how='inner')
        if df.empty:
            continue
            
        df = df.sort_values(['order_book_id', 'date'])
        
        # 计算每日收益率
        df['ret'] = df.groupby('order_book_id')['close'].pct_change()
        
        # 多线程处理每个日期
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_date = {
                executor.submit(calc_residuals_single_date, df, date, used_factors): date 
                for date in date_chunk
            }
            
            # 收集结果
            for future in as_completed(future_to_date):
                result = future.result()
                if not result.empty:
                    residuals_list.append(result)
        
        # 清理内存
        del data_chunk, barra_chunk, df

    if not residuals_list:
        print("没有有效的残差结果，返回空DataFrame")
        return pd.DataFrame(columns=['order_book_id', 'date', 'resid_sum'])

    resid_df = pd.concat(residuals_list, axis=0)
    resid_df = resid_df.sort_values(['order_book_id', 'date'])

    # === 4. n日残差累计 ===
    resid_df['resid_sum'] = resid_df.groupby('order_book_id')['resid'].transform(
        lambda x: x.rolling(n, min_periods=1).sum()
    )

    return resid_df[['order_book_id', 'date', 'resid_sum']]


def save_factor_to_pkl(resid_df: pd.DataFrame, n: int, factor_type: str):
    """
    将残差因子保存为pkl格式，仿照N日_日内动量.py的保存方式
    """
    # 将数据转换为 行为date，列为order_book_id，值为resid_sum 的格式
    pivot_df = resid_df.pivot(index='date', columns='order_book_id', values='resid_sum')
    
    # 创建因子库目录
    factor_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '因子库')
    if not os.path.exists(factor_dir):
        os.makedirs(factor_dir)
    
    # 保存因子
    factor_name = f"residual_{n}day_{factor_type}_factor"
    factor_path = os.path.join(factor_dir, f'{factor_name}.pkl')
    pivot_df.to_pickle(factor_path)
    
    print(f"\n{n}日{factor_type}残差因子已保存到: {factor_path}")
    print("保存后的数据格式如下（前5行前5列）：")
    print(pivot_df.iloc[:5, :5])
    print(f"\n因子矩阵形状: {pivot_df.shape}")
    print(f"日期范围: {pivot_df.index.min()} 到 {pivot_df.index.max()}")
    print(f"股票数量: {len(pivot_df.columns)}")
    
    return factor_path


if __name__ == "__main__":
    # 建议在命令行/终端先 conda activate rqplus
    print("正在加载数据...")
    data = pd.read_pickle(r"C:\Users\9shao\Desktop\github公开项目\Multi-Factor-Strategy-Development-Framework\多因子框架\行情数据库\data.pkl")
    barra = pd.read_pickle(r"C:\Users\9shao\Desktop\github公开项目\Multi-Factor-Strategy-Development-Framework\多因子框架\因子库\Barra因子\barra.pkl")
    
    print(f"数据形状: data={data.shape}, barra={barra.shape}")
    print(f"内存使用: data={data.memory_usage(deep=True).sum() / 1024**3:.2f} GB, barra={barra.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
    
    # 设置显示选项
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    # 可根据内存情况调整chunk_size和max_workers
    resid_result = calc_residuals(
        data, barra, 
        n=5, 
        factor_type="all", 
        chunk_size=20, 
        max_workers=4
    )
    
    print("\n=== 残差因子计算结果 ===")
    print("包含残差因子的数据前10行：")
    print(resid_result.head(10))
    
    # 统计信息
    print(f"\n=== 残差因子统计信息 ===")
    print("5日累计残差统计：")
    print(resid_result['resid_sum'].describe())
    
    # 保存因子
    if not resid_result.empty:
        factor_path = save_factor_to_pkl(resid_result, n=5, factor_type="all")
        print(f"\n因子已成功保存到: {factor_path}")
    else:
        print("残差结果为空，未保存因子。")
