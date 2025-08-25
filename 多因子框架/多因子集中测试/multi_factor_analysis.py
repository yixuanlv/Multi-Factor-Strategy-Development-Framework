import pandas as pd
import numpy as np
import warnings
from typing import Dict, Optional
from tqdm import tqdm
from scipy import stats
from pyecharts import options as opts
from pyecharts.charts import Line, HeatMap, Page
from pyecharts.components import Table
from sklearn.linear_model import LinearRegression
from itertools import combinations

warnings.filterwarnings('ignore')

# 简化的相关系数计算
def fast_correlation(x, y, method='spearman'):
    """相关系数计算"""
    if len(x) != len(y) or len(x) < 2:
        return np.nan
    
    if method == 'pearson':
        x_mean, y_mean = np.mean(x), np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        x_var, y_var = np.sum((x - x_mean) ** 2), np.sum((y - y_mean) ** 2)
        return numerator / np.sqrt(x_var * y_var) if x_var != 0 and y_var != 0 else np.nan
    else:
        x_ranks, y_ranks = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        return fast_correlation(x_ranks, y_ranks, 'pearson')

def fast_group_returns(factor_values, returns, n_groups):
    """分组收益率计算"""
    n = len(factor_values)
    if n < n_groups:
        return np.full(n_groups, np.nan)
    
    sorted_indices = np.argsort(factor_values)
    group_size = n // n_groups
    group_returns = np.zeros(n_groups)
    
    for i in range(n_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size if i < n_groups - 1 else n
        if start_idx < n:
            group_returns[i] = np.mean(returns[sorted_indices[start_idx:end_idx]])
        else:
            group_returns[i] = np.nan
    
    return group_returns

def fast_ic_calculation(factor_values, returns, filter_mask):
    """IC计算"""
    valid_mask = filter_mask & (~np.isnan(factor_values)) & (~np.isnan(returns))
    if np.sum(valid_mask) < 10:
        return np.nan
    
    valid_factors, valid_returns = factor_values[valid_mask], returns[valid_mask]
    return fast_correlation(valid_factors, valid_returns, 'spearman')

class MultiFactorAnalyzer:
    """
    多因子分析工具类 - 集成多因子测试和相关性分析功能
    输入：多个因子数据框的字典，收益率数据框
    处理：多因子集中测试、因子相关性分析
    输出：综合多因子分析报表
    """

    def __init__(self, factors_data: Dict[str, pd.DataFrame], returns_data: pd.DataFrame, 
                 rebalance_period: int = 1):
        """初始化多因子分析器"""
        self.factors_data = factors_data
        self.returns_data = returns_data
        self.rebalance_period = rebalance_period
        self.factor_names = list(factors_data.keys())
        
        # 缓存
        self._ic_cache = {}
        self._stats_cache = {}
        self._group_returns_cache = {}
        self._correlation_cache = {}
        
        self._align_data()
        
    def _align_data(self):
        """对齐所有因子数据和收益率数据"""
        print("正在对齐多因子数据...")
        
        # 转换日期格式
        for factor_data in self.factors_data.values():
            factor_data['date'] = pd.to_datetime(factor_data['date'])
        self.returns_data['date'] = pd.to_datetime(self.returns_data['date'])
        
        if 'close' not in self.returns_data.columns:
            raise ValueError("returns_data中必须包含'close'列")
        
        # 计算收益率
        returns_data = self.returns_data.copy()
        returns_data.sort_values(['order_book_id', 'date'], inplace=True)
        returns_data['return'] = returns_data.groupby('order_book_id')['close'].pct_change().shift(-1)
        
        self._precompute_filters(returns_data)
        self._align_factors(returns_data)
        self._precompute_future_returns()
        
        print(f"数据对齐完成：{len(self.common_dates)}个日期，{len(self.common_stocks)}只股票")
    
    def _precompute_filters(self, returns_data):
        """预计算过滤掩码"""
        self.filter_masks = {}
        filter_cols = ['limit_up_flag', 'limit_down_flag', 'ST', 'suspended']
        available_cols = [col for col in filter_cols if col in returns_data.columns]
        
        for col in available_cols:
                buy_mask = ~returns_data[col].astype(bool)
                self.filter_masks[f'buy_{col}'] = buy_mask.values
                
    def _align_factors(self, returns_data):
        """对齐因子数据"""
        self.aligned_factors = {}
        common_dates = None
        common_stocks = None
        
        for factor_name, factor_data in tqdm(self.factors_data.items(), desc="对齐因子"):
            # 合并数据
            filter_cols = [col for col in ['limit_up_flag', 'limit_down_flag', 'ST', 'suspended'] 
                          if col in returns_data.columns]
            merged_data = pd.merge(
                factor_data[['date', 'order_book_id', 'factor_value']],
                returns_data[['date', 'order_book_id', 'return', 'close'] + filter_cols],
                on=['date', 'order_book_id'], how='inner'
            ).dropna()
            
            if len(merged_data) == 0:
                print(f"警告：因子 {factor_name} 没有有效数据")
                continue
                
            self.aligned_factors[factor_name] = merged_data
            
            # 更新共同日期和股票
            if common_dates is None:
                common_dates = set(merged_data['date'].unique())
                common_stocks = set(merged_data['order_book_id'].unique())
            else:
                common_dates &= set(merged_data['date'].unique())
                common_stocks &= set(merged_data['order_book_id'].unique())
        
        if not self.aligned_factors:
            raise ValueError("没有有效的因子数据")
            
        self.common_dates = sorted(common_dates)
        self.common_stocks = sorted(common_stocks)
    
    def _precompute_future_returns(self):
        """预计算未来收益率"""
        for factor_data in self.aligned_factors.values():
            factor_data['future_return'] = factor_data.groupby('order_book_id')['return'].shift(-1)
            factor_data.set_index(['date', 'order_book_id'], inplace=True)
            factor_data.sort_index(inplace=True)
    
    def _get_filter_mask(self, data, for_buy=True):
        """获取过滤掩码"""
        if not self.filter_masks:
            return np.ones(len(data), dtype=bool)
        
        mask = np.ones(len(data), dtype=bool)
        for key, filter_mask in self.filter_masks.items():
            if key.startswith('buy_') and len(filter_mask) == len(data):
                        mask &= filter_mask
        return mask
    
    def calculate_factor_ic(self, factor_name: str, method: str = 'spearman') -> pd.Series:
        """计算单个因子的IC序列"""
        if factor_name in self._ic_cache:
            return self._ic_cache[factor_name]
        
        factor_data = self.aligned_factors[factor_name]
        dates = factor_data.index.get_level_values('date').unique()
        ic_values = []
        
        for date in dates:
            date_data = factor_data.loc[date]
            if len(date_data) < 10:
                ic_values.append(np.nan)
                continue
            
            filter_mask = self._get_filter_mask(date_data, for_buy=True)
            factor_values = date_data['factor_value'].values
            future_returns = date_data['future_return'].values
            
            ic = fast_ic_calculation(factor_values, future_returns, filter_mask)
            ic_values.append(ic)
        
        ic_series = pd.Series(ic_values, index=dates)
        self._ic_cache[factor_name] = ic_series
        return ic_series
    
    def calculate_all_factors_ic(self, method: str = 'spearman') -> pd.DataFrame:
        """计算所有因子的IC序列"""
        print(f"正在计算所有因子的IC（{method}相关系数）...")
        
        ic_data = {}
        for factor_name in tqdm(self.factor_names, desc="IC计算"):
            ic_series = self.calculate_factor_ic(factor_name, method)
            ic_data[factor_name] = ic_series
            
        return pd.DataFrame(ic_data)
    
    def calculate_factor_stats(self, factor_name: str, method: str = 'spearman') -> Dict:
        """计算单个因子的统计指标"""
        cache_key = f"{factor_name}_{method}"
        if cache_key in self._stats_cache:
            return self._stats_cache[cache_key]
        
        ic_series = self.calculate_factor_ic(factor_name, method).dropna()
        
        if len(ic_series) == 0:
            stats_dict = {col: np.nan for col in ['IC_mean', 'IC_std', 'ICIR', 'IC_positive_ratio', 
                                                 'IC_skew', 'IC_kurtosis', 'IC_tvalue', 'IC_pvalue']}
        else:
            ic_values = ic_series.values
            ic_mean, ic_std = np.mean(ic_values), np.std(ic_values, ddof=1)
            icir = ic_mean / ic_std if ic_std != 0 else np.nan
            ic_positive_ratio = np.mean(ic_values > 0)
            ic_skew = stats.skew(ic_values)
            ic_kurtosis = stats.kurtosis(ic_values, fisher=True)
            t_stat, p_value = stats.ttest_1samp(ic_values, 0)
            
            stats_dict = {
                'IC_mean': ic_mean, 'IC_std': ic_std, 'ICIR': icir, 'IC_positive_ratio': ic_positive_ratio,
                'IC_skew': ic_skew, 'IC_kurtosis': ic_kurtosis, 'IC_tvalue': t_stat, 'IC_pvalue': p_value
            }
        
        self._stats_cache[cache_key] = stats_dict
        return stats_dict
    
    def calculate_all_factors_stats(self, method: str = 'spearman') -> pd.DataFrame:
        """计算所有因子的统计指标"""
        print("正在计算所有因子的统计指标...")
        
        stats_list = []
        for factor_name in tqdm(self.factor_names, desc="统计指标计算"):
            stats_dict = self.calculate_factor_stats(factor_name, method)
            stats_dict['factor_name'] = factor_name
            stats_list.append(stats_dict)
            
        stats_df = pd.DataFrame(stats_list)
        stats_df.set_index('factor_name', inplace=True)
        return stats_df
    
    def _make_groups_on_rebalance_day(self, df_day, n_groups):
        """单个调仓日分组，返回 ['rb_date','order_book_id','group']"""
        df = df_day.dropna(subset=['factor_value']).copy()
        df = df[~df['limit_up_flag'].astype(bool) if 'limit_up_flag' in df.columns else True]
        df = df[~df['ST'].astype(bool) if 'ST' in df.columns else True]
        df = df[~df['suspended'].astype(bool) if 'suspended' in df.columns else True]
        
        if len(df) < n_groups:
            return pd.DataFrame(columns=['rb_date', 'order_book_id', 'group'])

        ranks = df['factor_value'].rank(method='first', ascending=True)
        group_size = max(1, len(ranks) // n_groups)
        groups = ((ranks - 1) // group_size).clip(upper=n_groups - 1).astype(int)

        return pd.DataFrame({
            'rb_date': df_day['date'].iloc[0],
            'order_book_id': df.loc[ranks.index, 'order_book_id'].values,
            'group': groups.values
        })
    
    def calculate_factor_group_returns(self, factor_name: str, n_groups: int = 10) -> Dict:
        """计算单个因子的分组收益率 - 使用改进的持仓权重计算"""
        cache_key = f"{factor_name}_{n_groups}"
        if cache_key in self._group_returns_cache:
            return self._group_returns_cache[cache_key]
        
        factor_data = self.aligned_factors[factor_name].reset_index()
        factor_data = factor_data.sort_values(['order_book_id', 'date'])
        
        # 调仓日分组
        rebalance_dates = factor_data['date'].unique()[::self.rebalance_period]
        df_rb = factor_data[factor_data['date'].isin(rebalance_dates)].copy()
        
        group_labels_list = []
        for d, g in df_rb.groupby('date'):
            res = self._make_groups_on_rebalance_day(g, n_groups)
            if len(res) > 0:
                group_labels_list.append(res)
        
        if not group_labels_list:
            return {'group_returns': pd.DataFrame(), 'factor_data': factor_data}
        
        group_labels_rebalance = pd.concat(group_labels_list, ignore_index=True)
        
        # 每个交易日映射到最近调仓日
        all_dates = sorted(factor_data['date'].unique())
        dates_df = pd.DataFrame({'date': all_dates})
        rb_df = pd.DataFrame({'rb_date': rebalance_dates})
        map_df = pd.merge_asof(dates_df, rb_df, left_on='date', right_on='rb_date', direction='backward')
        positions_daily = map_df.merge(group_labels_rebalance, how='left', on='rb_date')
        positions_daily = positions_daily[['date', 'rb_date', 'order_book_id', 'group']]
        
        # 合并未来收益
        pr = positions_daily.merge(
            factor_data[['date', 'order_book_id', 'future_return', 'return']],
            on=['date', 'order_book_id'], how='left'
        )
        pr = pr.dropna(subset=['group'])
        pr['group'] = pr['group'].astype(int)
        
        # 计算持仓权重
        counts = pr.groupby(['rb_date', 'group'])['order_book_id'].nunique().rename('n').reset_index()
        pr = pr.merge(counts, on=['rb_date', 'group'], how='left')
        pr['start_weight'] = 1.0 / pr['n']
        
        # 计算累计收益和持仓权重
        pr['gross'] = pr['return'].fillna(0) + 1.0
        pr['stock_cum'] = pr.groupby(['rb_date', 'group', 'order_book_id'])['gross'].cumprod()
        pr['holding_weight'] = np.where(
            pr['rb_date'] == pr['date'],
            pr['start_weight'],
            pr['start_weight'] * pr['stock_cum']
        )
        pr['holding_weight'] = pr.groupby(['date', 'group'])['holding_weight'].transform(
            lambda x: x / x.sum() if x.sum() != 0 else 0
        )
        
        # 计算分组每日收益率
        group_returns = pr.groupby(['date', 'group']).apply(
            lambda x: np.average(x['future_return'], weights=x['holding_weight'])
        ).reset_index()
        group_returns.columns = ['date', 'group', 'group_return']
        
        # 转换为宽表格式
        group_daily_returns = group_returns.pivot(index='date', columns='group', values='group_return').fillna(0.0)
        group_daily_returns = group_daily_returns.reindex(sorted(group_daily_returns.columns), axis=1)
        
        # 计算累计净值
        group_cum_nav = (group_daily_returns + 1).cumprod()
        
        # 确保所有日期都有数据
        group_daily_returns = group_daily_returns.reindex(all_dates, method='ffill').fillna(0.0)
        group_cum_nav = group_cum_nav.reindex(all_dates, method='ffill')
        
        result = {
            'group_returns': group_daily_returns,
            'group_cum_nav': group_cum_nav,
            'factor_data': factor_data
        }
        self._group_returns_cache[cache_key] = result
        return result
    
    def calculate_all_factors_group_returns(self, n_groups: int = 10) -> Dict[str, Dict]:
        """计算所有因子的分组收益率"""
        print(f"正在计算所有因子的分组收益率（{n_groups}分组）...")
        
        group_returns_data = {}
        for factor_name in tqdm(self.factor_names, desc="分组收益率计算"):
            group_data = self.calculate_factor_group_returns(factor_name, n_groups)
            group_returns_data[factor_name] = group_data
            
        return group_returns_data
    
    # ==================== 因子相关性分析方法 ====================
    
    def _filter_tradable_stocks(self, data, for_buy=True):
        """过滤可交易的股票"""
        filtered_data = data.copy()
        filter_cols = ['limit_up_flag', 'ST', 'suspended']
        
        for col in filter_cols:
            if col in filtered_data.columns:
                filtered_data = filtered_data[~filtered_data[col].astype(bool)]
        
        return filtered_data
    
    def calculate_factor_correlation_matrix(self) -> pd.DataFrame:
        """计算因子值相关性矩阵 - 简化版本，只计算一个相关性矩阵"""
        if 'factor_correlation' in self._correlation_cache:
            return self._correlation_cache['factor_correlation']
        
        print("正在计算因子值相关性矩阵...")
        
        # 构建因子值矩阵
        factor_values_matrix = pd.DataFrame()
        
        for factor_name in tqdm(self.factor_names, desc="构建因子值矩阵"):
            factor_data = self.aligned_factors[factor_name].reset_index()
            
            # 获取每个因子的截面因子值，按日期平均
            factor_pivot = factor_data.pivot(index='date', columns='order_book_id', values='factor_value')
            # 计算每个日期所有股票因子值的均值，作为该日期的因子代表值
            factor_values_matrix[factor_name] = factor_pivot.mean(axis=1)
        
        # 计算因子值之间的相关系数矩阵
        correlation_matrix = factor_values_matrix.corr(method='pearson')
        
        self._correlation_cache['factor_correlation'] = correlation_matrix
        return correlation_matrix

    # ==================== 性能计算方法 ====================
    
    def calculate_long_short_returns(self, group_returns: pd.DataFrame, long_group: int, short_group: int) -> pd.Series:
        """计算多空组合收益率"""
        return group_returns[long_group].dropna() - group_returns[short_group].dropna()
    
    def calculate_cumulative_returns(self, returns_series: pd.Series) -> pd.Series:
        """计算累计收益率"""
        return (1 + returns_series).cumprod()
    
    def calculate_annualized_stats(self, returns_series: pd.Series) -> Dict:
        """计算年化统计指标"""
        if len(returns_series) == 0:
            return {col: np.nan for col in ['annual_return', 'annual_volatility', 'sharpe_ratio', 'max_drawdown', 'win_rate']}
            
        total_return = (1 + returns_series).prod() - 1
        years = len(returns_series) / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan
        annual_volatility = returns_series.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else np.nan
        
        cumulative_returns = (1 + returns_series).cumprod()
        running_max = cumulative_returns.expanding().max()
        max_drawdown = ((cumulative_returns - running_max) / running_max).min()
        win_rate = (returns_series > 0).mean()
        
        return {
            'annual_return': annual_return, 'annual_volatility': annual_volatility, 'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown, 'win_rate': win_rate
        }
    
    # ==================== 报告生成方法 ====================
    
    def generate_comprehensive_report(self, n_groups: int = 10, method: str = 'spearman', 
                                    save_path: Optional[str] = None, show_log_returns: bool = True) -> Dict:
        """生成综合多因子分析报告"""
        print("=" * 60)
        print("多因子集中测试分析报告")
        print(f"调仓周期: {self.rebalance_period}, 分组数量: {n_groups}")
        print("=" * 60)
        
        # 计算基础数据
        ic_df = self.calculate_all_factors_ic(method)
        stats_df = self.calculate_all_factors_stats(method)
        group_returns_data = self.calculate_all_factors_group_returns(n_groups)
        cum_ic_df = ic_df.cumsum()
        
        # 计算多空组合和超额收益率
        long_short_data, long_excess_data, group_cumulative_returns_data = {}, {}, {}
        
        for factor_name in tqdm(self.aligned_factors.keys(), desc="多空/超额收益计算"):
            group_returns = group_returns_data[factor_name]['group_returns']
            cumulative_returns = (1 + group_returns).cumprod()
            group_cumulative_returns_data[factor_name] = cumulative_returns
            
            # 多空组合
            ls_1_10 = self.calculate_long_short_returns(group_returns, n_groups-1, 0)
            ls_2_9 = self.calculate_long_short_returns(group_returns, n_groups-2, 1)
            
            # 超额收益率
            market_return = group_returns.mean(axis=1).dropna()
            excess_1 = group_returns[n_groups-1].dropna() - market_return
            excess_2 = group_returns[n_groups-2].dropna() - market_return
            
            long_short_data[factor_name] = {
                'ls_1_10': ls_1_10, 'cum_ls_1_10': self.calculate_cumulative_returns(ls_1_10),
                'ls_2_9': ls_2_9, 'cum_ls_2_9': self.calculate_cumulative_returns(ls_2_9)
            }
            
            long_excess_data[factor_name] = {
                'excess_1': excess_1, 'cum_excess_1': self.calculate_cumulative_returns(excess_1),
                'excess_2': excess_2, 'cum_excess_2': self.calculate_cumulative_returns(excess_2)
            }
        
        # 计算年化统计指标
        performance_stats = {}
        for factor_name in tqdm(self.aligned_factors.keys(), desc="年化统计指标计算"):
            ls_1_10, ls_2_9 = long_short_data[factor_name]['ls_1_10'], long_short_data[factor_name]['ls_2_9']
            excess_1, excess_2 = long_excess_data[factor_name]['excess_1'], long_excess_data[factor_name]['excess_2']
            
            performance_stats[factor_name] = {
                'ls_1_10': self.calculate_annualized_stats(ls_1_10),
                'ls_2_9': self.calculate_annualized_stats(ls_2_9),
                'excess_1': self.calculate_annualized_stats(excess_1),
                'excess_2': self.calculate_annualized_stats(excess_2)
            }
        
        # 绘制图表和生成报告
        self.plot_comprehensive_analysis(stats_df, cum_ic_df, long_excess_data, long_short_data, 
                                       performance_stats, group_cumulative_returns_data, save_path, show_log_returns)
        self.print_detailed_report(stats_df, performance_stats)
        
        return {
            'ic_df': ic_df, 'cum_ic_df': cum_ic_df, 'stats_df': stats_df,
            'group_returns_data': group_returns_data, 'group_cumulative_returns_data': group_cumulative_returns_data,
            'long_short_data': long_short_data, 'long_excess_data': long_excess_data, 'performance_stats': performance_stats
        }
    

    
    # ==================== 图表绘制方法 ====================
    
    def plot_comprehensive_analysis(self, stats_df: pd.DataFrame, cum_ic_df: pd.DataFrame,
                                  long_excess_data: Dict, long_short_data: Dict,
                                  performance_stats: Dict, group_cumulative_returns_data: Dict, 
                                  save_path: Optional[str] = None, show_log_returns: bool = True):
        """绘制综合多因子分析图表 - 使用pyecharts生成HTML"""
        print("正在生成多因子分析图表...")
        
        # 1. 因子测试统计结果表格 - 直接显示数据
        stats_plot = stats_df[['IC_mean', 'ICIR', 'IC_positive_ratio', 'IC_tvalue', 'IC_pvalue']].copy()
        stats_plot['IC_pvalue'] = -np.log10(stats_plot['IC_pvalue'])
        
        # 准备表格数据
        table_data = []
        for factor_name in stats_plot.index:
            row_data = [factor_name]
            for col_name in stats_plot.columns:
                value = stats_plot.loc[factor_name, col_name]
                if not pd.isna(value):
                    row_data.append(f"{float(value):.4f}")
                else:
                    row_data.append("N/A")
            table_data.append(row_data)
        
        # 创建统计指标表格
        col_labels = ['因子名称'] + list(stats_plot.columns)
        stats_table = (
            Table()
            .add(col_labels, table_data)
            .set_global_opts(
                title_opts=opts.TitleOpts(title="因子测试统计结果表", pos_left="center")
            )
        )
        
        # 2. 因子相关系数矩阵热力图
        # 使用简化的相关性矩阵计算方法
        factor_corr_matrix = self.calculate_factor_correlation_matrix()
        
        factor_corr_heatmap_data = []
        for i, factor1 in enumerate(factor_corr_matrix.index):
            for j, factor2 in enumerate(factor_corr_matrix.columns):
                value = factor_corr_matrix.loc[factor1, factor2]
                if not pd.isna(value):
                    factor_corr_heatmap_data.append([j, i, round(float(value), 4)])
        
        factor_corr_heatmap = (
            HeatMap()
            .add_xaxis(list(factor_corr_matrix.columns))
            .add_yaxis("因子", list(factor_corr_matrix.index), factor_corr_heatmap_data,
                      label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(
                title_opts=opts.TitleOpts(title="因子值相关系数矩阵", pos_left="center"),
                xaxis_opts=opts.AxisOpts(name="因子", type_="category"),
                yaxis_opts=opts.AxisOpts(name="因子", type_="category"),
                visualmap_opts=opts.VisualMapOpts(
                    min_=factor_corr_matrix.values.min(), max_=factor_corr_matrix.values.max(),
                    pos_left="right", is_calculable=True, precision=4,
                    range_color=["#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", 
                               "#ffffcc", "#fee090", "#fdae61", "#f46d43", "#d73027"]
                )
            )
        )
        
        # 3. 因子累计IC折线图
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
        
        # 4. 第1层多头组合累计超额收益率
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
        
        # 5. 第1层、倒1层多空组合累计收益率
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
        
        # 6. 所有因子的分组累计收益率分箱图
        group_charts = []
        if group_cumulative_returns_data:
            for factor_name, cumulative_returns in group_cumulative_returns_data.items():
                group_chart = Line()
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
                    title = f'{factor_name} - 各分组累计对数收益率(log10)'
                    ylabel = '累计对数收益率(log10)'
                else:
                    title = f'{factor_name} - 各分组累计收益率'
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
                group_charts.append(group_chart)
        
        # 7. 性能对比表格
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
        page = Page(layout=Page.SimplePageLayout)
        
        # 添加所有图表
        # 添加统计指标表格
        page.add(stats_table)
        
        # 添加因子相关系数矩阵热力图
        page.add(factor_corr_heatmap)
        
        # 添加其他图表
        page.add(ic_chart, excess_chart, ls_chart, performance_table)
        
        # 添加所有因子的分组图表
        for group_chart in group_charts:
            page.add(group_chart)
        
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