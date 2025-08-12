import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from typing import Dict, List, Tuple, Optional
import os
from tqdm import tqdm

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class MultiFactorAnalyzer:
    """
    多因子分析工具类
    输入：多个因子数据框的字典，收益率数据框
    处理：多因子集中测试和对比分析
    """

    def __init__(self, factors_data: Dict[str, pd.DataFrame], returns_data: pd.DataFrame, 
                 rebalance_period: int = 1):
        """
        初始化多因子分析器
        
        Args:
            factors_data: 字典，键为因子名称，值为因子数据框（包含date, order_book_id, factor_value列）
            returns_data: 收益率数据框（包含date, order_book_id, close列）
            rebalance_period: 调仓周期，默认为1
        """
        self.factors_data = factors_data
        self.returns_data = returns_data
        self.rebalance_period = rebalance_period
        self.factor_names = list(factors_data.keys())
        self._align_data()
        
    def _align_data(self):
        """对齐所有因子数据和收益率数据"""
        print("正在对齐多因子数据...")
        
        # 确保日期格式正确
        for factor_name, factor_data in self.factors_data.items():
            factor_data['date'] = pd.to_datetime(factor_data['date'])
        self.returns_data['date'] = pd.to_datetime(self.returns_data['date'])
        
        # 检查returns_data中必须有close列
        if 'close' not in self.returns_data.columns:
            raise ValueError("returns_data中必须包含'close'列")
        
        # 计算收益率
        returns_data = self.returns_data.copy()
        returns_data.sort_values(['order_book_id', 'date'], inplace=True)
        returns_data['return'] = returns_data.groupby('order_book_id')['close'].pct_change().shift(-1)
        
        # 对齐所有因子数据
        self.aligned_factors = {}
        common_dates = None
        common_stocks = None
        
        for factor_name, factor_data in self.factors_data.items():
            # 合并因子数据和收益率数据，包含所有需要的列
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
            
        print(f"数据对齐完成：{len(common_dates)}个日期，{len(common_stocks)}只股票")
        print(f"有效因子数量：{len(self.aligned_factors)}")

    def _filter_tradable_stocks(self, data, for_buy=True):
        """
        过滤可交易的股票
        for_buy: True表示买入过滤，False表示卖出过滤
        """
        filtered_data = data.copy()
        
        # 检查是否存在相关列
        available_cols = [col for col in ['limit_up_flag', 'limit_down_flag', 'ST', 'suspended'] 
                         if col in filtered_data.columns]
        
        if not available_cols:
            return filtered_data
        
        # 买入过滤：不能买入涨停、ST、停牌、跌停的股票
        if for_buy:
            for col in available_cols:
                if col in filtered_data.columns:
                    # True表示在该状态，需要过滤掉
                    mask = ~filtered_data[col].astype(bool)
                    filtered_data = filtered_data[mask]
        
        # 卖出过滤：不能卖出跌停、停牌的股票
        else:
            for col in ['limit_down_flag', 'suspended']:
                if col in filtered_data.columns:
                    # True表示在该状态，需要过滤掉
                    mask = ~filtered_data[col].astype(bool)
                    filtered_data = filtered_data[mask]
        
        return filtered_data
        
    def calculate_factor_ic(self, factor_name: str, method: str = 'spearman') -> pd.Series:
        """计算单个因子的IC序列"""
        factor_data = self.aligned_factors[factor_name]
        
        # 创建未来第rebalance_period期的收益率列
        factor_data = factor_data.copy()
        factor_data['future_return'] = factor_data.groupby('order_book_id')['return'].shift(-self.rebalance_period)
        
        # 按日期分组计算IC
        def calculate_ic_for_date(group):
            if len(group) < 10:
                return np.nan
            
            # 应用买入过滤
            filtered_group = self._filter_tradable_stocks(group, for_buy=True)
            
            if len(filtered_group) < 10:
                return np.nan
                
            x = filtered_group['factor_value'].dropna()
            y = filtered_group['future_return'].dropna()
            
            # 确保x和y长度一致
            common_idx = x.index.intersection(y.index)
            if len(common_idx) < 10:
                return np.nan
                
            x = x.loc[common_idx]
            y = y.loc[common_idx]
            
            if method == 'spearman':
                return stats.spearmanr(x, y)[0]
            else:
                return stats.pearsonr(x, y)[0]
        
        ic_series = factor_data.groupby('date').apply(calculate_ic_for_date)
        return ic_series
    
    def calculate_all_factors_ic(self, method: str = 'spearman') -> pd.DataFrame:
        """计算所有因子的IC序列"""
        print(f"正在计算所有因子的IC（{method}相关系数）...")
        
        ic_data = {}
        for factor_name in tqdm(self.aligned_factors.keys(), desc="IC计算"):
            ic_series = self.calculate_factor_ic(factor_name, method)
            ic_data[factor_name] = ic_series
            
        ic_df = pd.DataFrame(ic_data)
        return ic_df
    
    def calculate_factor_stats(self, factor_name: str, method: str = 'spearman') -> Dict:
        """计算单个因子的统计指标"""
        ic_series = self.calculate_factor_ic(factor_name, method).dropna()
        
        if len(ic_series) == 0:
            return {
                'IC_mean': np.nan,
                'IC_std': np.nan,
                'ICIR': np.nan,
                'IC_positive_ratio': np.nan,
                'IC_skew': np.nan,
                'IC_kurtosis': np.nan,
                'IC_tvalue': np.nan,
                'IC_pvalue': np.nan
            }
            
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        icir = ic_mean / ic_std if ic_std != 0 else np.nan
        ic_positive_ratio = (ic_series > 0).mean()
        ic_skew = stats.skew(ic_series)
        ic_kurtosis = stats.kurtosis(ic_series, fisher=True)
        t_stat, p_value = stats.ttest_1samp(ic_series, 0, nan_policy='omit')
        
        return {
            'IC_mean': ic_mean,
            'IC_std': ic_std,
            'ICIR': icir,
            'IC_positive_ratio': ic_positive_ratio,
            'IC_skew': ic_skew,
            'IC_kurtosis': ic_kurtosis,
            'IC_tvalue': t_stat,
            'IC_pvalue': p_value
        }
    
    def calculate_all_factors_stats(self, method: str = 'spearman') -> pd.DataFrame:
        """计算所有因子的统计指标"""
        print("正在计算所有因子的统计指标...")
        
        stats_list = []
        for factor_name in tqdm(self.aligned_factors.keys(), desc="统计指标计算"):
            stats_dict = self.calculate_factor_stats(factor_name, method)
            stats_dict['factor_name'] = factor_name
            stats_list.append(stats_dict)
            
        stats_df = pd.DataFrame(stats_list)
        stats_df.set_index('factor_name', inplace=True)
        return stats_df
    
    def calculate_factor_group_returns(self, factor_name: str, n_groups: int = 10) -> Dict:
        """计算单个因子的分组收益率"""
        factor_data = self.aligned_factors[factor_name].copy()
        factor_data['future_return'] = factor_data.groupby('order_book_id')['return'].shift(-self.rebalance_period)
        
        # 创建分组
        def create_groups_for_date(group):
            # 应用买入过滤
            filtered_group = self._filter_tradable_stocks(group, for_buy=True)
            
            if len(filtered_group) < n_groups:
                return pd.Series(index=group.index, dtype=float)
            
            factor = filtered_group['factor_value'].dropna()
            
            if len(factor) < n_groups:
                return pd.Series(index=group.index, dtype=float)
                
            try:
                groups = pd.qcut(factor, n_groups, labels=False, duplicates='drop')
            except ValueError:
                ranks = factor.rank(method='first')
                group_size = len(ranks) // n_groups
                groups = (ranks - 1) // group_size
                groups = groups.clip(upper=n_groups-1)
                
            result = pd.Series(index=group.index, dtype=float)
            result.loc[factor.index] = groups
            return result
        
        group_series = factor_data.groupby('date').apply(create_groups_for_date)
        
        # 合并分组信息
        factor_data['group'] = group_series.values
        factor_data = factor_data.dropna()
        
        # 添加卖出过滤相关的列
        factor_data_with_sell_filter = factor_data.copy()
        if any(col in factor_data.columns for col in ['limit_down_flag', 'suspended']):
            factor_data_with_sell_filter = factor_data[['date', 'order_book_id', 'future_return', 'group'] + 
                                                      [col for col in ['limit_down_flag', 'suspended'] 
                                                       if col in factor_data.columns]]
        
        # 应用卖出过滤（在计算收益率时）
        def calculate_group_return_with_filter(group):
            # 过滤掉不能卖出的股票（跌停、停牌）
            filtered_group = self._filter_tradable_stocks(group, for_buy=False)
            if len(filtered_group) == 0:
                return np.nan
            return filtered_group['future_return'].mean()
        
        # 按日期和分组计算平均收益率（应用卖出过滤）
        group_returns = factor_data_with_sell_filter.groupby(['date', 'group']).apply(calculate_group_return_with_filter).unstack(fill_value=np.nan)
        
        # 确保所有分组都存在
        for g in range(n_groups):
            if g not in group_returns.columns:
                group_returns[g] = np.nan
                
        group_returns = group_returns.reindex(columns=range(n_groups))
        
        return {
            'group_returns': group_returns,
            'factor_data': factor_data
        }
    
    def calculate_all_factors_group_returns(self, n_groups: int = 10) -> Dict[str, Dict]:
        """计算所有因子的分组收益率"""
        print(f"正在计算所有因子的分组收益率（{n_groups}分组）...")
        
        group_returns_data = {}
        for factor_name in tqdm(self.aligned_factors.keys(), desc="分组收益率计算"):
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
        """绘制综合多因子分析图表"""
        print("正在生成多因子分析图表...")
        
        fig = plt.figure(figsize=(20, 20))  # 将高度从24改为20，使图表变矮
        gs = fig.add_gridspec(4, 2, height_ratios=[1.5, 2, 2, 2])
        
        # Sheet1: 因子测试统计结果热力图
        ax1 = fig.add_subplot(gs[0, :])
        stats_plot = stats_df[['IC_mean', 'ICIR', 'IC_positive_ratio', 'IC_tvalue', 'IC_pvalue']].copy()
        stats_plot['IC_pvalue'] = -np.log10(stats_plot['IC_pvalue'])  # 转换为-log10(p-value)
        
        sns.heatmap(stats_plot.T, annot=True, fmt='.4f', cmap='RdYlBu_r', 
                   center=0, ax=ax1, cbar_kws={'label': '数值'})
        ax1.set_title('多因子测试统计结果', fontsize=16, pad=20)
        ax1.set_xlabel('因子名称')
        ax1.set_ylabel('统计指标')
        
        # Sheet2: 因子累计IC折线图
        ax2 = fig.add_subplot(gs[1, 0])
        for factor_name in cum_ic_df.columns:
            ax2.plot(cum_ic_df.index, cum_ic_df[factor_name], 
                    label=factor_name, linewidth=2, alpha=0.8)
        ax2.set_title('因子累计IC对比', fontsize=14)
        ax2.set_xlabel('日期')
        ax2.set_ylabel('累计IC')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Sheet3: 第1层多头组合累计超额收益率
        ax3 = fig.add_subplot(gs[1, 1])
        for factor_name in long_excess_data.keys():
            cum_excess_1 = long_excess_data[factor_name]['cum_excess_1']
            if show_log_returns:
                # 使用对数收益率绘制
                log_excess_1 = np.log10(cum_excess_1.replace(0, np.nan))
                ax3.plot(cum_excess_1.index, log_excess_1, 
                        label=factor_name, linewidth=2, alpha=0.8)
            else:
                # 使用普通收益率绘制
                ax3.plot(cum_excess_1.index, cum_excess_1, 
                        label=factor_name, linewidth=2, alpha=0.8)
        
        if show_log_returns:
            ax3.set_title('第1层多头组合累计超额对数收益率(log10)', fontsize=14)
            ax3.set_ylabel('累计超额对数收益率(log10)')
        else:
            ax3.set_title('第1层多头组合累计超额收益率', fontsize=14)
            ax3.set_ylabel('累计超额收益率')
        ax3.set_xlabel('日期')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Sheet4: 第1层、倒1层多空组合累计收益率
        ax4 = fig.add_subplot(gs[2, 0])
        for factor_name in long_short_data.keys():
            cum_ls_1_10 = long_short_data[factor_name]['cum_ls_1_10']
            if show_log_returns:
                # 使用对数收益率绘制
                log_ls_1_10 = np.log10(cum_ls_1_10.replace(0, np.nan))
                ax4.plot(cum_ls_1_10.index, log_ls_1_10, 
                        label=factor_name, linewidth=2, alpha=0.8)
            else:
                # 使用普通收益率绘制
                ax4.plot(cum_ls_1_10.index, cum_ls_1_10, 
                        label=factor_name, linewidth=2, alpha=0.8)
        
        if show_log_returns:
            ax4.set_title('第1层、倒1层多空组合累计对数收益率(log10)', fontsize=14)
            ax4.set_ylabel('累计对数收益率(log10)')
        else:
            ax4.set_title('第1层、倒1层多空组合累计收益率', fontsize=14)
            ax4.set_ylabel('累计收益率')
        ax4.set_xlabel('日期')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        # 分组累计收益率分箱图
        ax5 = fig.add_subplot(gs[2, 1])
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
                        ax5.plot(cumulative_returns.index, log_returns,
                                 label=f'分组{group+1}', alpha=0.8)
                    else:
                        # 使用普通收益率绘制
                        ax5.plot(cumulative_returns.index, cumulative_returns[group],
                                 label=f'分组{group+1}', alpha=0.8)
            
            if show_log_returns:
                ax5.set_title(f'{first_factor} - 各分组累计对数收益率(log10)', fontsize=14)
                ax5.set_ylabel('累计对数收益率(log10)')
            else:
                ax5.set_title(f'{first_factor} - 各分组累计收益率', fontsize=14)
                ax5.set_ylabel('累计收益率')
            ax5.set_xlabel('日期')
            ax5.legend(loc='upper left', fontsize=8)
            ax5.grid(True, alpha=0.3)
        
        # 性能对比表格
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis('off')
        
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
        
        table = ax6.table(cellText=table_data, colLabels=col_labels, 
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.6)  # 将高度从2.0改为1.6，使表格变矮
        ax6.set_title('多因子性能对比表', fontsize=16, pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"多因子分析图表已保存到: {save_path}")
        
        plt.show()
    
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