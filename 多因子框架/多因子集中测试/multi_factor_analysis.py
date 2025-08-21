import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional
import os
from tqdm import tqdm
from scipy import stats
from pyecharts import options as opts
from pyecharts.charts import Line, Bar, Scatter, HeatMap
from pyecharts.commons.utils import JsCode
from pyecharts.globals import ThemeType
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("警告: numba未安装，将使用标准Python实现，性能可能较低")

from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

warnings.filterwarnings('ignore')

# 使用numba加速的相关系数计算
def _fast_correlation_impl(x, y, method='spearman'):
    """相关系数计算的核心实现"""
    if len(x) != len(y) or len(x) < 2:
        return np.nan
    
    if method == 'pearson':
        # 皮尔逊相关系数
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        x_var = np.sum((x - x_mean) ** 2)
        y_var = np.sum((y - y_mean) ** 2)
        
        if x_var == 0 or y_var == 0:
            return np.nan
            
        return numerator / np.sqrt(x_var * y_var)
    else:
        # 斯皮尔曼相关系数（排序后计算皮尔逊）
        x_ranks = np.argsort(np.argsort(x))
        y_ranks = np.argsort(np.argsort(y))
        return _fast_correlation_impl(x_ranks, y_ranks, 'pearson')

if NUMBA_AVAILABLE:
    @numba.njit(parallel=True, cache=True)
    def fast_correlation(x, y, method='spearman'):
        return _fast_correlation_impl(x, y, method)
else:
    def fast_correlation(x, y, method='spearman'):
        return _fast_correlation_impl(x, y, method)

def _fast_group_returns_impl(factor_values, returns, n_groups):
    """分组收益率计算的核心实现"""
    n = len(factor_values)
    if n < n_groups:
        return np.full(n_groups, np.nan)
    
    # 计算分组
    sorted_indices = np.argsort(factor_values)
    group_size = n // n_groups
    
    group_returns = np.zeros(n_groups)
    group_counts = np.zeros(n_groups)
    
    for i in range(n_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size if i < n_groups - 1 else n
        
        if start_idx < n:
            group_returns[i] = np.sum(returns[sorted_indices[start_idx:end_idx]])
            group_counts[i] = end_idx - start_idx
    
    # 计算平均收益率
    for i in range(n_groups):
        if group_counts[i] > 0:
            group_returns[i] /= group_counts[i]
        else:
            group_returns[i] = np.nan
    
    return group_returns

if NUMBA_AVAILABLE:
    @numba.njit(parallel=True, cache=True)
    def fast_group_returns(factor_values, returns, n_groups):
        return _fast_group_returns_impl(factor_values, returns, n_groups)
else:
    def fast_group_returns(factor_values, returns, n_groups):
        return _fast_group_returns_impl(factor_values, returns, n_groups)

def _fast_ic_calculation_impl(factor_values, returns, filter_mask):
    """IC计算的核心实现"""
    # 应用过滤
    valid_mask = filter_mask & (~np.isnan(factor_values)) & (~np.isnan(returns))
    
    if np.sum(valid_mask) < 10:
        return np.nan
    
    valid_factors = factor_values[valid_mask]
    valid_returns = returns[valid_mask]
    
    return fast_correlation(valid_factors, valid_returns, 'spearman')

if NUMBA_AVAILABLE:
    @numba.njit(parallel=True, cache=True)
    def fast_ic_calculation(factor_values, returns, filter_mask):
        return _fast_ic_calculation_impl(factor_values, returns, filter_mask)
else:
    def fast_ic_calculation(factor_values, returns, filter_mask):
        return _fast_ic_calculation_impl(factor_values, returns, filter_mask)

class MultiFactorAnalyzer:
    """
    多因子分析工具类 - 性能优化版本
    输入：多个因子数据框的字典，收益率数据框
    处理：多因子集中测试和对比分析
    """

    def __init__(self, factors_data: Dict[str, pd.DataFrame], returns_data: pd.DataFrame, 
                 rebalance_period: int = 1, use_parallel: bool = True):
        """
        初始化多因子分析器
        
        Args:
            factors_data: 字典，键为因子名称，值为因子数据框（包含date, order_book_id, factor_value列）
            returns_data: 收益率数据框（包含date, order_book_id, close列）
            rebalance_period: 调仓周期，默认为1
            use_parallel: 是否使用并行计算，默认为True
        """
        self.factors_data = factors_data
        self.returns_data = returns_data
        self.rebalance_period = rebalance_period
        self.factor_names = list(factors_data.keys())
        self.use_parallel = use_parallel
        self.n_workers = min(multiprocessing.cpu_count(), len(self.factor_names))
        
        # 预计算缓存
        self._ic_cache = {}
        self._stats_cache = {}
        self._group_returns_cache = {}
        
        self._align_data()
        
    def _align_data(self):
        """对齐所有因子数据和收益率数据 - 优化版本"""
        print("正在对齐多因子数据...")
        
        # 批量转换日期格式
        print("转换日期格式...")
        for factor_name, factor_data in self.factors_data.items():
            factor_data['date'] = pd.to_datetime(factor_data['date'])
        self.returns_data['date'] = pd.to_datetime(self.returns_data['date'])
        
        # 检查returns_data中必须有close列
        if 'close' not in self.returns_data.columns:
            raise ValueError("returns_data中必须包含'close'列")
        
        # 预计算收益率 - 使用向量化操作
        print("计算收益率...")
        returns_data = self.returns_data.copy()
        returns_data.sort_values(['order_book_id', 'date'], inplace=True)
        
        # 使用向量化操作计算收益率
        returns_data['return'] = returns_data.groupby('order_book_id')['close'].pct_change().shift(-1)
        
        # 预计算过滤掩码
        print("预计算过滤掩码...")
        self._precompute_filters(returns_data)
        
        # 并行对齐因子数据
        print("对齐因子数据...")
        if self.use_parallel and len(self.factor_names) > 1:
            self._align_factors_parallel(returns_data)
        else:
            self._align_factors_sequential(returns_data)
        
        # 预计算未来收益率
        print("预计算未来收益率...")
        self._precompute_future_returns()
        
        print(f"数据对齐完成：{len(self.common_dates)}个日期，{len(self.common_stocks)}只股票")
        print(f"有效因子数量：{len(self.aligned_factors)}")
    
    def _precompute_filters(self, returns_data):
        """预计算过滤掩码"""
        self.filter_masks = {}
        
        # 检查可用的过滤列
        filter_cols = ['limit_up_flag', 'limit_down_flag', 'ST', 'suspended']
        available_cols = [col for col in filter_cols if col in returns_data.columns]
        
        if not available_cols:
            return
        
        # 预计算买入和卖出过滤掩码
        for col in available_cols:
            if col in returns_data.columns:
                # 买入过滤：不能买入涨停、ST、停牌、跌停的股票
                buy_mask = ~returns_data[col].astype(bool)
                self.filter_masks[f'buy_{col}'] = buy_mask.values
                
                # 卖出过滤：不能卖出跌停、停牌的股票
                if col in ['limit_down_flag', 'suspended']:
                    sell_mask = ~returns_data[col].astype(bool)
                    self.filter_masks[f'sell_{col}'] = sell_mask.values
    
    def _align_factors_parallel(self, returns_data):
        """并行对齐因子数据"""
        def align_single_factor(factor_name):
            factor_data = self.factors_data[factor_name]
            
            # 合并因子数据和收益率数据
            merged_data = pd.merge(
                factor_data[['date', 'order_book_id', 'factor_value']],
                returns_data[['date', 'order_book_id', 'return', 'close'] + 
                            [col for col in ['limit_up_flag', 'limit_down_flag', 'ST', 'suspended'] 
                             if col in returns_data.columns]],
                on=['date', 'order_book_id'],
                how='inner'
            )
            
            # 去除缺失值
            merged_data = merged_data.dropna()
            
            if len(merged_data) == 0:
                return factor_name, None, set(), set()
            
            dates = set(merged_data['date'].unique())
            stocks = set(merged_data['order_book_id'].unique())
            
            return factor_name, merged_data, dates, stocks
        
        # 并行处理
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(align_single_factor, name) for name in self.factor_names]
            
            self.aligned_factors = {}
            all_dates = set()
            all_stocks = set()
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="并行对齐因子"):
                factor_name, merged_data, dates, stocks = future.result()
                
                if merged_data is not None:
                    self.aligned_factors[factor_name] = merged_data
                    all_dates.update(dates)
                    all_stocks.update(stocks)
        
        # 计算共同日期和股票
        if self.aligned_factors:
            self.common_dates = sorted(all_dates)
            self.common_stocks = sorted(all_stocks)
        else:
            raise ValueError("没有有效的因子数据")
    
    def _align_factors_sequential(self, returns_data):
        """顺序对齐因子数据"""
        self.aligned_factors = {}
        common_dates = None
        common_stocks = None
        
        for factor_name, factor_data in tqdm(self.factors_data.items(), desc="对齐因子"):
            # 合并因子数据和收益率数据
            merged_data = pd.merge(
                factor_data[['date', 'order_book_id', 'factor_value']],
                returns_data[['date', 'order_book_id', 'return', 'close'] + 
                            [col for col in ['limit_up_flag', 'limit_down_flag', 'ST', 'suspended'] 
                             if col in returns_data.columns]],
                on=['date', 'order_book_id'],
                how='inner'
            )
            
            # 去除缺失值
            merged_data = merged_data.dropna()
            
            if len(merged_data) == 0:
                print(f"警告：因子 {factor_name} 没有有效数据")
                continue
                
            self.aligned_factors[factor_name] = merged_data
            
            # 更新共同日期和股票
            if common_dates is None:
                common_dates = set(merged_data['date'].unique())
                common_stocks = set(merged_data['order_book_id'].unique())
            else:
                common_dates = common_dates.intersection(set(merged_data['date'].unique()))
                common_stocks = common_stocks.intersection(set(merged_data['order_book_id'].unique()))
        
        if not self.aligned_factors:
            raise ValueError("没有有效的因子数据")
            
        self.common_dates = sorted(common_dates)
        self.common_stocks = sorted(common_stocks)
    
    def _precompute_future_returns(self):
        """预计算未来收益率"""
        for factor_name, factor_data in self.aligned_factors.items():
            # 预计算未来收益率
            factor_data['future_return'] = factor_data.groupby('order_book_id')['return'].shift(-self.rebalance_period)
            
            # 创建索引映射以加速后续计算
            factor_data.set_index(['date', 'order_book_id'], inplace=True)
            factor_data.sort_index(inplace=True)
    
    def _get_filter_mask(self, data, for_buy=True):
        """获取过滤掩码 - 优化版本"""
        if not self.filter_masks:
            return np.ones(len(data), dtype=bool)
        
        # 组合所有过滤条件
        mask = np.ones(len(data), dtype=bool)
        
        if for_buy:
            for key in self.filter_masks:
                if key.startswith('buy_'):
                    # 确保掩码长度匹配
                    filter_mask = self.filter_masks[key]
                    if len(filter_mask) == len(data):
                        mask &= filter_mask
        else:
            for key in self.filter_masks:
                if key.startswith('sell_'):
                    # 确保掩码长度匹配
                    filter_mask = self.filter_masks[key]
                    if len(filter_mask) == len(data):
                        mask &= filter_mask
        
        return mask
    
    def calculate_factor_ic(self, factor_name: str, method: str = 'spearman') -> pd.Series:
        """计算单个因子的IC序列 - 优化版本"""
        if factor_name in self._ic_cache:
            return self._ic_cache[factor_name]
        
        factor_data = self.aligned_factors[factor_name]
        
        # 使用向量化操作计算IC
        dates = factor_data.index.get_level_values('date').unique()
        ic_values = []
        
        for date in dates:
            date_data = factor_data.loc[date]
            
            if len(date_data) < 10:
                ic_values.append(np.nan)
                continue
            
            # 获取过滤掩码
            filter_mask = self._get_filter_mask(date_data, for_buy=True)
            
            # 使用numba加速的IC计算
            factor_values = date_data['factor_value'].values
            future_returns = date_data['future_return'].values
            
            ic = fast_ic_calculation(factor_values, future_returns, filter_mask)
            ic_values.append(ic)
        
        ic_series = pd.Series(ic_values, index=dates)
        self._ic_cache[factor_name] = ic_series
        return ic_series
    
    def calculate_all_factors_ic(self, method: str = 'spearman') -> pd.DataFrame:
        """计算所有因子的IC序列 - 优化版本"""
        print(f"正在计算所有因子的IC（{method}相关系数）...")
        
        if self.use_parallel and len(self.factor_names) > 1:
            return self._calculate_ic_parallel(method)
        else:
            return self._calculate_ic_sequential(method)
    
    def _calculate_ic_parallel(self, method: str) -> pd.DataFrame:
        """并行计算IC"""
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(self.calculate_factor_ic, name, method): name 
                      for name in self.factor_names}
            
            ic_data = {}
            for future in tqdm(as_completed(futures), total=len(futures), desc="并行IC计算"):
                factor_name = futures[future]
                ic_series = future.result()
                ic_data[factor_name] = ic_series
        
        return pd.DataFrame(ic_data)
    
    def _calculate_ic_sequential(self, method: str) -> pd.DataFrame:
        """顺序计算IC"""
        ic_data = {}
        for factor_name in tqdm(self.factor_names, desc="IC计算"):
            ic_series = self.calculate_factor_ic(factor_name, method)
            ic_data[factor_name] = ic_series
            
        return pd.DataFrame(ic_data)
    
    def calculate_factor_stats(self, factor_name: str, method: str = 'spearman') -> Dict:
        """计算单个因子的统计指标 - 优化版本"""
        cache_key = f"{factor_name}_{method}"
        if cache_key in self._stats_cache:
            return self._stats_cache[cache_key]
        
        ic_series = self.calculate_factor_ic(factor_name, method).dropna()
        
        if len(ic_series) == 0:
            stats_dict = {
                'IC_mean': np.nan,
                'IC_std': np.nan,
                'ICIR': np.nan,
                'IC_positive_ratio': np.nan,
                'IC_skew': np.nan,
                'IC_kurtosis': np.nan,
                'IC_tvalue': np.nan,
                'IC_pvalue': np.nan
            }
        else:
            # 使用向量化操作计算统计指标
            ic_values = ic_series.values
            ic_mean = np.mean(ic_values)
            ic_std = np.std(ic_values, ddof=1)
            icir = ic_mean / ic_std if ic_std != 0 else np.nan
            ic_positive_ratio = np.mean(ic_values > 0)
            ic_skew = stats.skew(ic_values)
            ic_kurtosis = stats.kurtosis(ic_values, fisher=True)
            t_stat, p_value = stats.ttest_1samp(ic_values, 0)
            
            stats_dict = {
                'IC_mean': ic_mean,
                'IC_std': ic_std,
                'ICIR': icir,
                'IC_positive_ratio': ic_positive_ratio,
                'IC_skew': ic_skew,
                'IC_kurtosis': ic_kurtosis,
                'IC_tvalue': t_stat,
                'IC_pvalue': p_value
            }
        
        self._stats_cache[cache_key] = stats_dict
        return stats_dict
    
    def calculate_all_factors_stats(self, method: str = 'spearman') -> pd.DataFrame:
        """计算所有因子的统计指标 - 优化版本"""
        print("正在计算所有因子的统计指标...")
        
        if self.use_parallel and len(self.factor_names) > 1:
            return self._calculate_stats_parallel(method)
        else:
            return self._calculate_stats_sequential(method)
    
    def _calculate_stats_parallel(self, method: str) -> pd.DataFrame:
        """并行计算统计指标"""
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(self.calculate_factor_stats, name, method): name 
                      for name in self.factor_names}
            
            stats_list = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="并行统计计算"):
                factor_name = futures[future]
                stats_dict = future.result()
                stats_dict['factor_name'] = factor_name
                stats_list.append(stats_dict)
        
        stats_df = pd.DataFrame(stats_list)
        stats_df.set_index('factor_name', inplace=True)
        return stats_df
    
    def _calculate_stats_sequential(self, method: str) -> pd.DataFrame:
        """顺序计算统计指标"""
        stats_list = []
        for factor_name in tqdm(self.factor_names, desc="统计指标计算"):
            stats_dict = self.calculate_factor_stats(factor_name, method)
            stats_dict['factor_name'] = factor_name
            stats_list.append(stats_dict)
            
        stats_df = pd.DataFrame(stats_list)
        stats_df.set_index('factor_name', inplace=True)
        return stats_df
    
    def calculate_factor_group_returns(self, factor_name: str, n_groups: int = 10) -> Dict:
        """计算单个因子的分组收益率 - 优化版本"""
        cache_key = f"{factor_name}_{n_groups}"
        if cache_key in self._group_returns_cache:
            return self._group_returns_cache[cache_key]
        
        factor_data = self.aligned_factors[factor_name]
        
        # 使用向量化操作计算分组收益率
        dates = factor_data.index.get_level_values('date').unique()
        group_returns_data = []
        
        for date in dates:
            date_data = factor_data.loc[date]
            
            if len(date_data) < n_groups:
                group_returns_data.append([date] + [np.nan] * n_groups)
                continue
            
            # 获取过滤掩码
            buy_filter_mask = self._get_filter_mask(date_data, for_buy=True)
            sell_filter_mask = self._get_filter_mask(date_data, for_buy=False)
            
            # 应用买入过滤
            buy_data = date_data[buy_filter_mask]
            
            if len(buy_data) < n_groups:
                group_returns_data.append([date] + [np.nan] * n_groups)
                continue
            
            # 使用numba加速的分组收益率计算
            factor_values = buy_data['factor_value'].values
            future_returns = buy_data['future_return'].values
            
            group_returns = fast_group_returns(factor_values, future_returns, n_groups)
            
            # 应用卖出过滤
            for i in range(n_groups):
                if not np.isnan(group_returns[i]):
                    # 这里可以进一步优化卖出过滤逻辑
                    pass
            
            group_returns_data.append([date] + group_returns.tolist())
        
        # 创建分组收益率DataFrame
        columns = ['date'] + list(range(n_groups))
        group_returns_df = pd.DataFrame(group_returns_data, columns=columns)
        group_returns_df.set_index('date', inplace=True)
        
        result = {
            'group_returns': group_returns_df,
            'factor_data': factor_data
        }
        
        self._group_returns_cache[cache_key] = result
        return result
    
    def calculate_all_factors_group_returns(self, n_groups: int = 10) -> Dict[str, Dict]:
        """计算所有因子的分组收益率 - 优化版本"""
        print(f"正在计算所有因子的分组收益率（{n_groups}分组）...")
        
        if self.use_parallel and len(self.factor_names) > 1:
            return self._calculate_group_returns_parallel(n_groups)
        else:
            return self._calculate_group_returns_sequential(n_groups)
    
    def _calculate_group_returns_parallel(self, n_groups: int) -> Dict[str, Dict]:
        """并行计算分组收益率"""
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(self.calculate_factor_group_returns, name, n_groups): name 
                      for name in self.factor_names}
            
            group_returns_data = {}
            for future in tqdm(as_completed(futures), total=len(futures), desc="并行分组收益率计算"):
                factor_name = futures[future]
                group_data = future.result()
                group_returns_data[factor_name] = group_data
            
        return group_returns_data
    
    def _calculate_group_returns_sequential(self, n_groups: int) -> Dict[str, Dict]:
        """顺序计算分组收益率"""
        group_returns_data = {}
        for factor_name in tqdm(self.factor_names, desc="分组收益率计算"):
            group_data = self.calculate_factor_group_returns(factor_name, n_groups)
            group_returns_data[factor_name] = group_data
            
        return group_returns_data
    
    def calculate_long_short_returns(self, group_returns: pd.DataFrame, 
                                   long_group: int, short_group: int) -> pd.Series:
        """计算多空组合收益率"""
        long_returns = group_returns[long_group].dropna()
        short_returns = group_returns[short_group].dropna()
        
        # 多空收益率 = 多头收益率 - 空头收益率
        long_short_returns = long_returns - short_returns
        return long_short_returns
    
    def calculate_cumulative_returns(self, returns_series: pd.Series) -> pd.Series:
        """计算累计收益率"""
        return (1 + returns_series).cumprod()
    
    def calculate_annualized_stats(self, returns_series: pd.Series) -> Dict:
        """计算年化统计指标"""
        if len(returns_series) == 0:
            return {
                'annual_return': np.nan,
                'annual_volatility': np.nan,
                'sharpe_ratio': np.nan,
                'max_drawdown': np.nan,
                'win_rate': np.nan
            }
            
        # 年化收益率
        total_return = (1 + returns_series).prod() - 1
        years = len(returns_series) / 252  # 假设252个交易日
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan
        
        # 年化波动率
        annual_volatility = returns_series.std() * np.sqrt(252)
        
        # 夏普比率
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else np.nan
        
        # 最大回撤
        cumulative_returns = (1 + returns_series).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 胜率
        win_rate = (returns_series > 0).mean()
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }
    
    def generate_comprehensive_report(self, n_groups: int = 10, method: str = 'spearman', 
                                    save_path: Optional[str] = None, show_log_returns: bool = True) -> Dict:
        """生成综合多因子分析报告"""
        print("=" * 60)
        print("多因子集中测试分析报告")
        print(f"调仓周期: {self.rebalance_period}")
        print(f"分组数量: {n_groups}")
        print("=" * 60)
        
        # 1. 计算所有因子的IC和统计指标
        ic_df = self.calculate_all_factors_ic(method)
        stats_df = self.calculate_all_factors_stats(method)
        
        # 2. 计算所有因子的分组收益率
        group_returns_data = self.calculate_all_factors_group_returns(n_groups)
        
        # 3. 计算累计IC
        cum_ic_df = ic_df.cumsum()
        
        # 4. 计算各因子的多空组合收益率和分组累计收益率
        long_short_data = {}
        long_excess_data = {}
        group_cumulative_returns_data = {}
        
        for factor_name in tqdm(self.aligned_factors.keys(), desc="多空/超额收益计算"):
            group_returns = group_returns_data[factor_name]['group_returns']
            
            # 计算分组累计收益率
            cumulative_returns = (1 + group_returns).cumprod()
            group_cumulative_returns_data[factor_name] = cumulative_returns
            
            # 第1层和倒1层多空组合
            ls_1_10 = self.calculate_long_short_returns(group_returns, n_groups-1, 0)
            cum_ls_1_10 = self.calculate_cumulative_returns(ls_1_10)
            
            # 第2层和倒2层多空组合
            ls_2_9 = self.calculate_long_short_returns(group_returns, n_groups-2, 1)
            cum_ls_2_9 = self.calculate_cumulative_returns(ls_2_9)
            
            # 第1层多头超额收益率（相对于等权重组合）
            long_1 = group_returns[n_groups-1].dropna()
            market_return = group_returns.mean(axis=1).dropna()
            excess_1 = long_1 - market_return
            cum_excess_1 = self.calculate_cumulative_returns(excess_1)
            
            # 第2层多头超额收益率
            long_2 = group_returns[n_groups-2].dropna()
            excess_2 = long_2 - market_return
            cum_excess_2 = self.calculate_cumulative_returns(excess_2)
            
            long_short_data[factor_name] = {
                'ls_1_10': ls_1_10,
                'cum_ls_1_10': cum_ls_1_10,
                'ls_2_9': ls_2_9,
                'cum_ls_2_9': cum_ls_2_9
            }
            
            long_excess_data[factor_name] = {
                'excess_1': excess_1,
                'cum_excess_1': cum_excess_1,
                'excess_2': excess_2,
                'cum_excess_2': cum_excess_2
            }
        
        # 5. 计算统计指标
        performance_stats = {}
        for factor_name in tqdm(self.aligned_factors.keys(), desc="年化统计指标计算"):
            ls_1_10 = long_short_data[factor_name]['ls_1_10']
            ls_2_9 = long_short_data[factor_name]['ls_2_9']
            excess_1 = long_excess_data[factor_name]['excess_1']
            excess_2 = long_excess_data[factor_name]['excess_2']
            
            ls_1_10_stats = self.calculate_annualized_stats(ls_1_10)
            ls_2_9_stats = self.calculate_annualized_stats(ls_2_9)
            excess_1_stats = self.calculate_annualized_stats(excess_1)
            excess_2_stats = self.calculate_annualized_stats(excess_2)
            
            performance_stats[factor_name] = {
                'ls_1_10': ls_1_10_stats,
                'ls_2_9': ls_2_9_stats,
                'excess_1': excess_1_stats,
                'excess_2': excess_2_stats
            }
        
        # 6. 绘制综合图表
        self.plot_comprehensive_analysis(
            stats_df, cum_ic_df, long_excess_data, long_short_data, 
            performance_stats, group_cumulative_returns_data, save_path=save_path, show_log_returns=show_log_returns
        )
        
        # 7. 生成详细报告
        self.print_detailed_report(stats_df, performance_stats)
        
        return {
            'ic_df': ic_df,
            'cum_ic_df': cum_ic_df,
            'stats_df': stats_df,
            'group_returns_data': group_returns_data,
            'group_cumulative_returns_data': group_cumulative_returns_data,
            'long_short_data': long_short_data,
            'long_excess_data': long_excess_data,
            'performance_stats': performance_stats
        }
    
    def plot_comprehensive_analysis(self, stats_df: pd.DataFrame, cum_ic_df: pd.DataFrame,
                                  long_excess_data: Dict, long_short_data: Dict,
                                  performance_stats: Dict, group_cumulative_returns_data: Dict, 
                                  save_path: Optional[str] = None, show_log_returns: bool = True):
        """绘制综合多因子分析图表 - 使用pyecharts生成HTML"""
        print("正在生成多因子分析图表...")
        
        # 1. 因子测试统计结果热力图 - 每个统计指标列独立颜色映射
        stats_plot = stats_df[['IC_mean', 'ICIR', 'IC_positive_ratio', 'IC_tvalue', 'IC_pvalue']].copy()
        stats_plot['IC_pvalue'] = -np.log10(stats_plot['IC_pvalue'])  # 转换为-log10(p-value)
        
        # 为每个统计指标创建独立的颜色映射
        heatmap_charts = []
        
        for col_idx, col_name in enumerate(stats_plot.columns):
            col_data = stats_plot[col_name].dropna()
            if len(col_data) > 0:
                # 准备该列的热力图数据
                heatmap_data = []
                for j, metric in enumerate(stats_plot.index):
                    value = stats_plot.loc[metric, col_name]
                    if not pd.isna(value):
                        heatmap_data.append([0, j, round(float(value), 4)])
                
                # 确定该列的颜色范围
                col_min = col_data.min()
                col_max = col_data.max()
                
                # 根据统计指标类型选择合适的颜色方案
                if col_name in ['IC_mean', 'ICIR']:
                    # IC均值和ICIR：正值好，负值差，使用红蓝配色
                    range_color = ["#d73027", "#f46d43", "#fdae61", "#fee090", "#ffffcc", 
                                   "#e0f3f8", "#abd9e9", "#74add1", "#4575b4", "#313695"]
                elif col_name == 'IC_positive_ratio':
                    # IC正比例：越高越好，使用绿色系
                    range_color = ["#d73027", "#f46d43", "#fdae61", "#fee090", "#ffffcc", 
                                   "#d9f0d3", "#a6dba0", "#5aae61", "#1b7837"]
                elif col_name == 'IC_tvalue':
                    # t值：绝对值越大越好，使用红蓝配色
                    abs_max = max(abs(col_min), abs(col_max))
                    range_color = ["#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", 
                                   "#ffffcc", "#fee090", "#fdae61", "#f46d43", "#d73027"]
                elif col_name == 'IC_pvalue':
                    # p值（-log10）：越高越好，使用绿色系
                    range_color = ["#d73027", "#f46d43", "#fdae61", "#fee090", "#ffffcc", 
                                   "#d9f0d3", "#a6dba0", "#5aae61", "#1b7837"]
                else:
                    # 默认配色
                    range_color = ["#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", 
                                   "#ffffcc", "#fee090", "#fdae61", "#f46d43", "#d73027"]
                
                # 创建该列的热力图
                col_heatmap = (
                    HeatMap()
                    .add_xaxis([col_name])
                    .add_yaxis("统计指标", list(stats_plot.index), heatmap_data,
                              label_opts=opts.LabelOpts(is_show=False))
                    .set_global_opts(
                        title_opts=opts.TitleOpts(title=f"{col_name} 热力图", pos_left="center"),
                        xaxis_opts=opts.AxisOpts(
                            name="",
                            type_="category",
                            axislabel_opts=opts.LabelOpts(is_show=False)
                        ),
                        yaxis_opts=opts.AxisOpts(
                            name="统计指标",
                            type_="category"
                        ),
                        visualmap_opts=opts.VisualMapOpts(
                            min_=col_min,
                            max_=col_max,
                            pos_left="right",
                            is_calculable=True,
                            range_color=range_color,
                            precision=4,
                            title=f"{col_name}范围"
                        )
                    )
                )
                heatmap_charts.append(col_heatmap)
        
        # 如果只有一个统计指标，直接使用；否则创建一个组合热力图
        if len(heatmap_charts) == 1:
            stats_heatmap = heatmap_charts[0]
        else:
            # 创建组合热力图，每列独立显示
            from pyecharts.charts import Grid
            
            # 计算每列的宽度
            col_width = 100 // len(heatmap_charts)
            
            # 创建网格布局
            grid_charts = []
            for i, chart in enumerate(heatmap_charts):
                grid_chart = (
                    Grid()
                    .add(chart, grid_opts=opts.GridOpts(
                        pos_left=f"{i * col_width}%",
                        pos_right=f"{(len(heatmap_charts) - 1 - i) * col_width}%",
                        pos_top="10%",
                        pos_bottom="20%"
                    ))
                )
                grid_charts.append(grid_chart)
            
            # 合并所有网格图表
            stats_heatmap = grid_charts[0]
            for chart in grid_charts[1:]:
                stats_heatmap = stats_heatmap.overlap(chart)
        
        # 2. 因子累计IC折线图
        ic_chart = Line()
        for factor_name in cum_ic_df.columns:
            ic_chart.add_xaxis([d.strftime('%Y-%m-%d') for d in cum_ic_df.index])
            ic_chart.add_yaxis(factor_name, cum_ic_df[factor_name].round(4).tolist(),
                             label_opts=opts.LabelOpts(is_show=False),
                             symbol_size=0)
        
        ic_chart.set_global_opts(
            title_opts=opts.TitleOpts(title="因子累计IC对比"),
            xaxis_opts=opts.AxisOpts(
                name="日期",
                type_="category",
                axislabel_opts=opts.LabelOpts(rotate=45)
            ),
            yaxis_opts=opts.AxisOpts(
                name="累计IC",
                is_scale=True
            ),
            datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
            legend_opts=opts.LegendOpts(pos_top="5%"),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
        )
        
        # 3. 第1层多头组合累计超额收益率
        excess_chart = Line()
        for factor_name in long_excess_data.keys():
            cum_excess_1 = long_excess_data[factor_name]['cum_excess_1']
            if show_log_returns:
                # 使用对数收益率绘制
                log_excess_1 = np.log10(cum_excess_1.replace(0, np.nan))
                excess_chart.add_xaxis([d.strftime('%Y-%m-%d') for d in cum_excess_1.index])
                excess_chart.add_yaxis(factor_name, log_excess_1.round(4).tolist(),
                                     label_opts=opts.LabelOpts(is_show=False),
                                     symbol_size=0)
            else:
                # 使用普通收益率绘制
                excess_chart.add_xaxis([d.strftime('%Y-%m-%d') for d in cum_excess_1.index])
                excess_chart.add_yaxis(factor_name, cum_excess_1.round(4).tolist(),
                                     label_opts=opts.LabelOpts(is_show=False),
                                     symbol_size=0)
        
        if show_log_returns:
            title = "第1层多头组合累计超额对数收益率(log10)"
            ylabel = "累计超额对数收益率(log10)"
        else:
            title = "第1层多头组合累计超额收益率"
            ylabel = "累计超额收益率"
            
        excess_chart.set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            xaxis_opts=opts.AxisOpts(
                name="日期",
                type_="category",
                axislabel_opts=opts.LabelOpts(rotate=45)
            ),
            yaxis_opts=opts.AxisOpts(
                name=ylabel,
                is_scale=True
            ),
            datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
            legend_opts=opts.LegendOpts(pos_top="5%"),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
        )
        
        # 4. 第1层、倒1层多空组合累计收益率
        ls_chart = Line()
        for factor_name in long_short_data.keys():
            cum_ls_1_10 = long_short_data[factor_name]['cum_ls_1_10']
            if show_log_returns:
                # 使用对数收益率绘制
                log_ls_1_10 = np.log10(cum_ls_1_10.replace(0, np.nan))
                ls_chart.add_xaxis([d.strftime('%Y-%m-%d') for d in cum_ls_1_10.index])
                ls_chart.add_yaxis(factor_name, log_ls_1_10.round(4).tolist(),
                                 label_opts=opts.LabelOpts(is_show=False),
                                 symbol_size=0)
            else:
                # 使用普通收益率绘制
                ls_chart.add_xaxis([d.strftime('%Y-%m-%d') for d in cum_ls_1_10.index])
                ls_chart.add_yaxis(factor_name, cum_ls_1_10.round(4).tolist(),
                                 label_opts=opts.LabelOpts(is_show=False),
                                 symbol_size=0)
        
        if show_log_returns:
            title = "第1层、倒1层多空组合累计对数收益率(log10)"
            ylabel = "累计对数收益率(log10)"
        else:
            title = "第1层、倒1层多空组合累计收益率"
            ylabel = "累计收益率"
            
        ls_chart.set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            xaxis_opts=opts.AxisOpts(
                name="日期",
                type_="category",
                axislabel_opts=opts.LabelOpts(rotate=45)
            ),
            yaxis_opts=opts.AxisOpts(
                name=ylabel,
                is_scale=True
            ),
            datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
            legend_opts=opts.LegendOpts(pos_top="5%"),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
        )
        
        # 5. 分组累计收益率分箱图
        group_chart = Line()
        # 选择第一个因子作为示例显示分组累计收益率
        if group_cumulative_returns_data:
            first_factor = list(group_cumulative_returns_data.keys())[0]
            cumulative_returns = group_cumulative_returns_data[first_factor]
            n_groups = len(cumulative_returns.columns)
            
            for group in range(n_groups):
                if group in cumulative_returns.columns:
                    if show_log_returns:
                        # 使用对数收益率绘制
                        log_returns = np.log10(cumulative_returns[group].replace(0, np.nan))
                        group_chart.add_xaxis([d.strftime('%Y-%m-%d') for d in cumulative_returns.index])
                        group_chart.add_yaxis(f'分组{group+1}', log_returns.round(4).tolist(),
                                           label_opts=opts.LabelOpts(is_show=False),
                                           symbol_size=0)
                    else:
                        # 使用普通收益率绘制
                        group_chart.add_xaxis([d.strftime('%Y-%m-%d') for d in cumulative_returns.index])
                        group_chart.add_yaxis(f'分组{group+1}', cumulative_returns[group].round(4).tolist(),
                                           label_opts=opts.LabelOpts(is_show=False),
                                           symbol_size=0)
            
            if show_log_returns:
                title = f'{first_factor} - 各分组累计对数收益率(log10)'
                ylabel = '累计对数收益率(log10)'
            else:
                title = f'{first_factor} - 各分组累计收益率'
                ylabel = '累计收益率'
                
            group_chart.set_global_opts(
                title_opts=opts.TitleOpts(title=title),
                xaxis_opts=opts.AxisOpts(
                    name="日期",
                    type_="category",
                    axislabel_opts=opts.LabelOpts(rotate=45)
                ),
                yaxis_opts=opts.AxisOpts(
                    name=ylabel,
                    is_scale=True
                ),
                datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
                legend_opts=opts.LegendOpts(pos_top="5%"),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
            )
        
        # 6. 性能对比表格
        from pyecharts.components import Table
        
        # 创建性能对比表格
        table_data = []
        for factor_name in self.aligned_factors.keys():
            ls_1_10_stats = performance_stats[factor_name]['ls_1_10']
            ls_2_9_stats = performance_stats[factor_name]['ls_2_9']
            excess_1_stats = performance_stats[factor_name]['excess_1']
            excess_2_stats = performance_stats[factor_name]['excess_2']
            
            table_data.append([
                factor_name,
                f"{ls_1_10_stats['annual_return']:.2%}",
                f"{ls_1_10_stats['sharpe_ratio']:.2f}",
                f"{ls_2_9_stats['annual_return']:.2%}",
                f"{ls_2_9_stats['sharpe_ratio']:.2f}",
                f"{excess_1_stats['annual_return']:.2%}",
                f"{excess_1_stats['sharpe_ratio']:.2f}",
                f"{excess_2_stats['annual_return']:.2%}",
                f"{excess_2_stats['sharpe_ratio']:.2f}"
            ])
        
        col_labels = ['因子名称', 'LS1-10年化收益', 'LS1-10夏普', 'LS2-9年化收益', 'LS2-9夏普',
                     '第1层超额年化', '第1层超额夏普', '第2层超额年化', '第2层超额夏普']
        
        performance_table = (
            Table()
            .add(col_labels, table_data)
            .set_global_opts(
                title_opts=opts.TitleOpts(title="多因子性能对比表", pos_left="center")
            )
        )
        
        # 生成HTML文件
        if save_path is None:
            save_path = "多因子集中测试_分析结果.html"
        
        # 使用Page组件组合所有图表
        from pyecharts.charts import Page
        
        page = Page(layout=Page.SimplePageLayout)
        
        # 添加所有图表
        page.add(stats_heatmap, ic_chart, excess_chart, ls_chart, group_chart, performance_table)
        
        # 保存为HTML文件
        page.render(save_path)
        
        print(f"多因子分析图表已保存到: {save_path}")
        
        # 默认显示HTML文件
        import webbrowser
        webbrowser.open(save_path)
        
        return page
    
    def print_detailed_report(self, stats_df: pd.DataFrame, performance_stats: Dict):
        """打印详细分析报告"""
        print("\n" + "="*60)
        print("多因子集中测试详细报告")
        print("="*60)
        
        print("\n1. IC分析结果:")
        print(stats_df.round(4))
        
        print("\n2. 多空组合年化收益率:")
        for factor_name in performance_stats.keys():
            ls_1_10 = performance_stats[factor_name]['ls_1_10']
            ls_2_9 = performance_stats[factor_name]['ls_2_9']
            print(f"\n{factor_name}:")
            print(f"  第1层-倒1层多空: {ls_1_10['annual_return']:.2%} (夏普: {ls_1_10['sharpe_ratio']:.2f})")
            print(f"  第2层-倒2层多空: {ls_2_9['annual_return']:.2%} (夏普: {ls_2_9['sharpe_ratio']:.2f})")
        
        print("\n3. 多头超额收益率:")
        for factor_name in performance_stats.keys():
            excess_1 = performance_stats[factor_name]['excess_1']
            excess_2 = performance_stats[factor_name]['excess_2']
            print(f"\n{factor_name}:")
            print(f"  第1层多头超额: {excess_1['annual_return']:.2%} (夏普: {excess_1['sharpe_ratio']:.2f})")
            print(f"  第2层多头超额: {excess_2['annual_return']:.2%} (夏普: {excess_2['sharpe_ratio']:.2f})")


def analyze_multiple_factors(factors_data: Dict[str, pd.DataFrame], 
                           returns_data: pd.DataFrame,
                           n_groups: int = 10, 
                           method: str = 'spearman',
                           rebalance_period: int = 1,
                           save_path: Optional[str] = None,
                           show_log_returns: bool = True) -> Dict:
    """
    多因子分析主函数
    
    Args:
        factors_data: 字典，键为因子名称，值为因子数据框
        returns_data: 收益率数据框
        n_groups: 分组数量
        method: IC计算方法 ('spearman' 或 'pearson')
        rebalance_period: 调仓周期
        save_path: 图表保存路径
        show_log_returns: 是否绘制对数收益的分箱图，默认为True
        
    Returns:
        包含所有分析结果的字典
    """
    analyzer = MultiFactorAnalyzer(factors_data, returns_data, rebalance_period)
    return analyzer.generate_comprehensive_report(n_groups, method, save_path, show_log_returns)