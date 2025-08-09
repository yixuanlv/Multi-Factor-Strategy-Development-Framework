import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
from typing import Dict, List, Tuple, Optional
import os
from tqdm import tqdm
from itertools import combinations
import multiprocessing as mp
from joblib import Parallel, delayed
import functools
import time
import psutil

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 获取CPU核心数，用于并行计算
N_JOBS = max(1, mp.cpu_count() - 1)  # 保留一个核心给系统

def timing_decorator(func):
    """性能监控装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        result = func(*args, **kwargs)
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        elapsed_time = end_time - start_time
        memory_used = end_memory - start_memory
        print(f"{func.__name__} 执行时间: {elapsed_time:.2f} 秒, 内存使用: {memory_used:.2f} MB")
        return result
    return wrapper

class CollinearityAnalyzer:
    """
    多因子共线性分析工具类
    输入：多个因子数据框的字典，收益率数据框
    处理：共线性分析
    输出：多因子共线性分析报表
    """

    def __init__(self, factors_data: Dict[str, pd.DataFrame], returns_data: pd.DataFrame, 
                 rebalance_period: int = 1, n_jobs: Optional[int] = None, use_parallel: bool = True):
        self.factors_data = factors_data
        self.returns_data = returns_data
        self.rebalance_period = rebalance_period
        self.factor_names = list(factors_data.keys())
        self.use_parallel = use_parallel
        if n_jobs is None:
            self.n_jobs = N_JOBS
        else:
            self.n_jobs = min(n_jobs, mp.cpu_count())
        if self.use_parallel:
            print(f"共线性分析器初始化完成，使用 {self.n_jobs} 个核心进行并行计算")
        else:
            print(f"共线性分析器初始化完成，使用串行计算模式")
        print(f"系统总核心数: {mp.cpu_count()}")
        self.validate_results = False  # 可以设置为True来启用验证
        self._align_data()
        
    def _align_data(self):
        print("正在对齐多因子数据...")
        for factor_name, factor_data in self.factors_data.items():
            factor_data['date'] = pd.to_datetime(factor_data['date'])
        self.returns_data['date'] = pd.to_datetime(self.returns_data['date'])
        if 'close' not in self.returns_data.columns:
            raise ValueError("returns_data中必须包含'close'列")
        returns_data = self.returns_data.copy()
        returns_data.sort_values(['order_book_id', 'date'], inplace=True)
        returns_data['return'] = returns_data.groupby('order_book_id')['close'].pct_change().shift(-1)
        self.aligned_factors = {}
        common_dates = None
        common_stocks = None
        for factor_name, factor_data in self.factors_data.items():
            merged_data = pd.merge(
                factor_data[['date', 'order_book_id', 'factor_value']],
                returns_data[['date', 'order_book_id', 'return']],
                on=['date', 'order_book_id'],
                how='inner'
            )
            merged_data = merged_data.dropna()
            if len(merged_data) == 0:
                print(f"警告：因子 {factor_name} 没有有效数据")
                continue
            self.aligned_factors[factor_name] = merged_data
            if common_dates is None:
                common_dates = set(merged_data['date'].unique())
                common_stocks = set(merged_data['order_book_id'].unique())
            else:
                common_dates = common_dates.intersection(set(merged_data['date'].unique()))
                common_stocks = common_stocks.intersection(set(merged_data['order_book_id'].unique()))
        if not self.aligned_factors:
            raise ValueError("没有有效的因子数据")
        print(f"数据对齐完成：{len(common_dates)}个日期，{len(common_stocks)}只股票")
        print(f"有效因子数量：{len(self.aligned_factors)}")
        
    def _calculate_beta_for_date(self, date_data_tuple):
        date, date_data = date_data_tuple
        try:
            valid_data = date_data[['factor_value', 'future_return']].dropna()
            if len(valid_data) < 10:
                return date, np.nan
            X = valid_data['factor_value'].values.reshape(-1, 1)
            y = valid_data['future_return'].values
            reg = LinearRegression()
            reg.fit(X, y)
            return date, reg.coef_[0]
        except Exception as e:
            return date, np.nan

    def calculate_factor_beta(self, factor_name: str) -> pd.Series:
        factor_data = self.aligned_factors[factor_name].copy()
        factor_data['future_return'] = factor_data.groupby('order_book_id')['return'].shift(-self.rebalance_period)
        unique_dates = list(factor_data['date'].unique())
        date_data_list = []
        for date in unique_dates:
            date_data = factor_data[factor_data['date'] == date]
            date_data_list.append((date, date_data))
        if self.use_parallel:
            n_jobs_actual = min(self.n_jobs, 16)
            print(f"正在并行计算 {factor_name} 的beta序列，使用 {n_jobs_actual} 个核心...")
            results = Parallel(n_jobs=n_jobs_actual, verbose=0, batch_size=10)(
                delayed(self._calculate_beta_for_date)(date_data_tuple) 
                for date_data_tuple in tqdm(date_data_list, desc=f"计算{factor_name}的beta序列")
            )
        else:
            print(f"正在串行计算 {factor_name} 的beta序列...")
            results = []
            for date_data_tuple in tqdm(date_data_list, desc=f"计算{factor_name}的beta序列"):
                result = self._calculate_beta_for_date(date_data_tuple)
                results.append(result)
        beta_series = pd.Series(index=unique_dates, dtype=float)
        for date, beta_value in results:
            beta_series[date] = beta_value
        self._validate_beta_calculation(factor_name, beta_series)
        return beta_series

    def _validate_beta_calculation(self, factor_name: str, parallel_result: pd.Series) -> bool:
        if not self.validate_results:
            return True
        print(f"正在验证 {factor_name} 的beta计算结果...")
        factor_data = self.aligned_factors[factor_name].copy()
        factor_data['future_return'] = factor_data.groupby('order_book_id')['return'].shift(-self.rebalance_period)
        serial_result = pd.Series(index=factor_data['date'].unique(), dtype=float)
        for date in factor_data['date'].unique():
            date_data = factor_data[factor_data['date'] == date]
            valid_data = date_data[['factor_value', 'future_return']].dropna()
            if len(valid_data) < 10:
                serial_result[date] = np.nan
                continue
            try:
                X = valid_data['factor_value'].values.reshape(-1, 1)
                y = valid_data['future_return'].values
                reg = LinearRegression()
                reg.fit(X, y)
                serial_result[date] = reg.coef_[0]
            except:
                serial_result[date] = np.nan
        is_valid = parallel_result.equals(serial_result)
        if is_valid:
            print(f"✓ {factor_name} beta计算结果验证通过")
        else:
            print(f"✗ {factor_name} beta计算结果验证失败")
            print(f"  并行结果: {parallel_result.head()}")
            print(f"  串行结果: {serial_result.head()}")
        return is_valid

    def calculate_all_factors_beta(self) -> pd.DataFrame:
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        print("正在计算所有因子的beta序列...")
        factor_names = list(self.aligned_factors.keys())
        
        # 检查数据规模，小规模数据使用串行计算
        total_data_size = sum(len(data) for data in self.aligned_factors.values())
        if total_data_size < 500:  # 小于1万条数据使用串行
            print(f"数据规模较小({total_data_size}条)，使用串行计算...")
            self.use_parallel = False
        
        if self.use_parallel:
            print(f"正在并行计算 {len(factor_names)} 个因子的beta序列，使用 {self.n_jobs} 个核心...")
            # 避免嵌套并行，内层使用串行
            temp_use_parallel = self.use_parallel
            self.use_parallel = False
            results = Parallel(n_jobs=self.n_jobs, verbose=0, batch_size=1)(
                delayed(self.calculate_factor_beta)(factor_name) 
                for factor_name in tqdm(factor_names, desc="计算所有因子的beta序列")
            )
            self.use_parallel = temp_use_parallel
        else:
            print(f"正在串行计算 {len(factor_names)} 个因子的beta序列...")
            results = []
            for factor_name in tqdm(factor_names, desc="计算所有因子的beta序列"):
                beta_series = self.calculate_factor_beta(factor_name)
                results.append(beta_series)
        beta_data = {}
        for factor_name, beta_series in zip(factor_names, results):
            beta_data[factor_name] = beta_series
        beta_df = pd.DataFrame(beta_data)
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        elapsed_time = end_time - start_time
        memory_used = end_memory - start_memory
        print(f"calculate_all_factors_beta 执行时间: {elapsed_time:.2f} 秒, 内存使用: {memory_used:.2f} MB")
        return beta_df

    def calculate_beta_correlation(self, beta_df: pd.DataFrame) -> pd.DataFrame:
        print("正在计算因子收益率相关性矩阵...")
        beta_corr = beta_df.corr(method='pearson')
        return beta_corr

    def calculate_factor_correlation_for_date(self, date: pd.Timestamp) -> pd.DataFrame:
        factor_data_for_date = {}
        for factor_name, factor_data in self.aligned_factors.items():
            date_data = factor_data[factor_data['date'] == date]
            if len(date_data) > 0:
                # 处理重复的order_book_id，取第一个值或平均值
                stock_factor_map = date_data.groupby('order_book_id')['factor_value'].first()
                factor_data_for_date[factor_name] = stock_factor_map
        if len(factor_data_for_date) < 2:
            return pd.DataFrame()
        all_factors_df = pd.DataFrame(factor_data_for_date)
        factor_corr = all_factors_df.corr(method='pearson')
        return factor_corr

    def calculate_factor_correlation_series(self) -> Dict[pd.Timestamp, pd.DataFrame]:
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        print("正在计算截面因子值相关性矩阵序列...")
        all_dates = set()
        for factor_data in self.aligned_factors.values():
            all_dates.update(factor_data['date'].unique())
        common_dates = sorted(list(all_dates))
        
        # 检查数据规模，小规模数据使用串行计算
        if len(common_dates) < 100:  # 小于100个日期使用串行
            print(f"日期数量较少({len(common_dates)}个)，使用串行计算...")
            self.use_parallel = False
        
        if self.use_parallel:
            print(f"正在并行计算 {len(common_dates)} 个日期的相关性矩阵，使用 {self.n_jobs} 个核心...")
            results = Parallel(n_jobs=self.n_jobs, verbose=0, batch_size=max(1, len(common_dates)//self.n_jobs))(
                delayed(self.calculate_factor_correlation_for_date)(date) 
                for date in tqdm(common_dates, desc="计算截面相关性")
            )
        else:
            print(f"正在串行计算 {len(common_dates)} 个日期的相关性矩阵...")
            results = []
            for date in tqdm(common_dates, desc="计算截面相关性"):
                factor_corr = self.calculate_factor_correlation_for_date(date)
                results.append(factor_corr)
        factor_corr_series = {}
        for date, factor_corr in zip(common_dates, results):
            if not factor_corr.empty:
                factor_corr_series[date] = factor_corr
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        elapsed_time = end_time - start_time
        memory_used = end_memory - start_memory
        print(f"calculate_factor_correlation_series 执行时间: {elapsed_time:.2f} 秒, 内存使用: {memory_used:.2f} MB")
        return factor_corr_series

    def calculate_factor_correlation_mean(self, factor_corr_series: Dict[pd.Timestamp, pd.DataFrame]) -> pd.DataFrame:
        print("正在计算截面因子值相关性矩阵均值...")
        if not factor_corr_series:
            return pd.DataFrame()
        corr_matrices = list(factor_corr_series.values())
        factor_corr_mean = pd.concat(corr_matrices).groupby(level=0).mean()
        return factor_corr_mean

    def _calculate_pair_correlation(self, factor_pair_data):
        factor1, factor2, factor_corr_series = factor_pair_data
        pair_name = f"{factor1}_{factor2}"
        corr_series = []
        dates = []
        for date, corr_matrix in factor_corr_series.items():
            if factor1 in corr_matrix.index and factor2 in corr_matrix.columns:
                corr_value = corr_matrix.loc[factor1, factor2]
                if not pd.isna(corr_value):
                    corr_series.append(corr_value)
                    dates.append(date)
        if corr_series:
            return pair_name, pd.Series(corr_series, index=dates)
        else:
            return pair_name, None

    def calculate_pairwise_correlation_series(self, factor_corr_series: Dict[pd.Timestamp, pd.DataFrame]) -> pd.DataFrame:
        print("正在计算两两配对因子的相关性序列...")
        if not factor_corr_series:
            return pd.DataFrame()
        factor_pairs = list(combinations(self.factor_names, 2))
        pair_data_list = [(factor1, factor2, factor_corr_series) for factor1, factor2 in factor_pairs]
        if self.use_parallel:
            print(f"正在并行计算 {len(factor_pairs)} 个因子对的相关性序列，使用 {self.n_jobs} 个核心...")
            results = Parallel(n_jobs=self.n_jobs, verbose=0, batch_size=5)(
                delayed(self._calculate_pair_correlation)(pair_data) 
                for pair_data in tqdm(pair_data_list, desc="计算因子对相关性序列")
            )
        else:
            print(f"正在串行计算 {len(factor_pairs)} 个因子对的相关性序列...")
            results = []
            for pair_data in tqdm(pair_data_list, desc="计算因子对相关性序列"):
                result = self._calculate_pair_correlation(pair_data)
                results.append(result)
        pair_corr_data = {}
        for pair_name, corr_series in results:
            if corr_series is not None:
                pair_corr_data[pair_name] = corr_series
        if pair_corr_data:
            cum_corr_df = pd.DataFrame(pair_corr_data)
            return cum_corr_df
        else:
            return pd.DataFrame()

    def calculate_cumulative_correlation(self, corr_df: pd.DataFrame) -> pd.DataFrame:
        return corr_df.cumsum()

    def generate_collinearity_report(self, save_path: Optional[str] = None) -> Dict:
        print("=" * 60)
        print("多因子共线性分析报告")
        print(f"调仓周期: {self.rebalance_period}")
        print("=" * 60)
        beta_df = self.calculate_all_factors_beta()
        beta_corr = self.calculate_beta_correlation(beta_df)
        factor_corr_series = self.calculate_factor_correlation_series()
        factor_corr_mean = self.calculate_factor_correlation_mean(factor_corr_series)
        cum_corr_df = self.calculate_pairwise_correlation_series(factor_corr_series)
        cum_corr_cumsum = self.calculate_cumulative_correlation(cum_corr_df)
        self.plot_collinearity_analysis(
            beta_corr, factor_corr_mean, cum_corr_cumsum, save_path=save_path
        )
        self.print_collinearity_report(beta_corr, factor_corr_mean, cum_corr_df)
        return {
            'beta_df': beta_df,
            'beta_corr': beta_corr,
            'factor_corr_series': factor_corr_series,
            'factor_corr_mean': factor_corr_mean,
            'cum_corr_df': cum_corr_df,
            'cum_corr_cumsum': cum_corr_cumsum
        }

    def plot_collinearity_analysis(self, beta_corr: pd.DataFrame, factor_corr_mean: pd.DataFrame,
                                    cum_corr_cumsum: pd.DataFrame, save_path: Optional[str] = None):
        """绘制共线性分析图表"""
        print("正在生成共线性分析图表...")

        # 动态调整画布大小，尤其是相关性矩阵较大时
        n_factors = max(
            beta_corr.shape[0] if beta_corr is not None else 0,
            factor_corr_mean.shape[0] if factor_corr_mean is not None else 0,
            2
        )
        # 基础宽高
        base_w, base_h = 24, 20
        # 每多一个因子，宽高增加
        extra = max(n_factors - 8, 0)
        fig_w = base_w + extra * 2.5
        # 画布高度大幅增加，避免遮挡
        fig_h = base_h + extra * 10

        # 只保留两个子图：因子收益率相关性热力图、截面因子值相关性均值热力图
        fig, axes = plt.subplots(1, 2, figsize=(fig_w, base_h))

        # Sheet1: 因子收益率相关性热力图
        ax1 = axes[0]
        if not beta_corr.empty:
            sns.heatmap(
                beta_corr,
                annot=True,
                fmt='.4f',
                cmap='RdYlBu_r',
                center=0,
                ax=ax1,
                cbar_kws={'label': '相关系数'},
                annot_kws={"size": 14} if n_factors <= 10 else {"size": 10}
            )
            ax1.set_title('因子收益率相关性矩阵', fontsize=18, pad=24)
            ax1.tick_params(axis='x', labelsize=14)
            ax1.tick_params(axis='y', labelsize=14)
        else:
            ax1.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax1.transAxes, fontsize=16)
            ax1.set_title('因子收益率相关性矩阵', fontsize=18, pad=24)

        # Sheet2: 截面因子值相关性均值热力图
        ax2 = axes[1]
        if not factor_corr_mean.empty:
            sns.heatmap(
                factor_corr_mean,
                annot=True,
                fmt='.4f',
                cmap='RdYlBu_r',
                center=0,
                ax=ax2,
                cbar_kws={'label': '相关系数'},
                annot_kws={"size": 14} if n_factors <= 10 else {"size": 10}
            )
            ax2.set_title('截面因子值相关性均值矩阵', fontsize=18, pad=24)
            ax2.tick_params(axis='x', labelsize=14)
            ax2.tick_params(axis='y', labelsize=14)
        else:
            ax2.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax2.transAxes, fontsize=16)
            ax2.set_title('截面因子值相关性均值矩阵', fontsize=18, pad=24)

        plt.tight_layout(rect=[0, 0, 1, 0.98])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"共线性分析图表已保存到: {save_path}")

        plt.show()

    def print_collinearity_report(self, beta_corr: pd.DataFrame, factor_corr_mean: pd.DataFrame, 
                                cum_corr_df: pd.DataFrame):
        print("\n" + "="*60)
        print("多因子共线性分析详细报告")
        print("="*60)
        print("\n1. 因子收益率相关性矩阵:")
        if not beta_corr.empty:
            print(beta_corr.round(4))
        else:
            print("无有效数据")
        print("\n2. 截面因子值相关性均值矩阵:")
        if not factor_corr_mean.empty:
            print(factor_corr_mean.round(4))
        else:
            print("无有效数据")
        print("\n3. 因子对相关性统计:")
        if not cum_corr_df.empty:
            for col in tqdm(list(cum_corr_df.columns), desc="打印因子对相关性统计"):
                corr_series = cum_corr_df[col].dropna()
                if len(corr_series) > 0:
                    print(f"\n{col}:")
                    print(f"  均值: {corr_series.mean():.4f}")
                    print(f"  标准差: {corr_series.std():.4f}")
                    print(f"  最小值: {corr_series.min():.4f}")
                    print(f"  最大值: {corr_series.max():.4f}")
                    print(f"  观测数: {len(corr_series)}")
        else:
            print("无有效数据")


def analyze_collinearity(factors_data: Dict[str, pd.DataFrame],
                        returns_data: pd.DataFrame,
                        rebalance_period: int = 1,
                        save_path: Optional[str] = None,
                        n_jobs: Optional[int] = None,
                        use_parallel: bool = True) -> Dict:
    analyzer = CollinearityAnalyzer(factors_data, returns_data, rebalance_period, n_jobs, use_parallel)
    return analyzer.generate_collinearity_report(save_path)