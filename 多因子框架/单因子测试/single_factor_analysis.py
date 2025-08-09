import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class SingleFactorAnalyzer:
    """
    单因子分析工具类
    输入：长格式数据框，包含date、order_book_id、factor_value、close列
    """

    def __init__(self, factor_data, returns_data, factor_name='factor', rebalance_period=1):
        self.factor_data = factor_data
        self.returns_data = returns_data
        self.factor_name = factor_name
        self.rebalance_period = rebalance_period  # 调仓周期，默认为1（每期调仓）
        self._align_data()

    def _align_data(self):
        """对齐因子数据和收盘价数据，并计算收益率"""
        # 确保日期格式正确
        self.factor_data['date'] = pd.to_datetime(self.factor_data['date'])
        self.returns_data['date'] = pd.to_datetime(self.returns_data['date'])

        # 检查returns_data中必须有close列
        if 'close' not in self.returns_data.columns:
            raise ValueError("returns_data中必须包含'close'列")

        # 计算收益率
        returns_data = self.returns_data.copy()
        returns_data.sort_values(['order_book_id', 'date'], inplace=True)
        returns_data['return'] = returns_data.groupby('order_book_id')['close'].pct_change().shift(-1)
        
        # 合并数据，包含所有需要的列
        merged_data = pd.merge(
            self.factor_data[['date', 'order_book_id', 'factor_value']],
            returns_data[['date', 'order_book_id', 'return', 'close'] + 
                        [col for col in ['limit_up_flag', 'limit_down_flag', 'ST', 'suspended'] 
                         if col in returns_data.columns]],
            on=['date', 'order_book_id'],
            how='inner'
        )

        # 去除缺失值
        merged_data = merged_data.dropna()

        if len(merged_data) == 0:
            raise ValueError("因子数据和收益率数据没有共同的有效数据")

        self.merged_data = merged_data
        print(f"数据对齐完成：{len(merged_data)}条记录")
        print(f"时间范围：{merged_data['date'].min()} 到 {merged_data['date'].max()}")
        print(f"股票数量：{merged_data['order_book_id'].nunique()}")

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

    def calculate_ic(self, method='spearman'):
        """优化的IC计算，使用向量化操作，支持调仓周期"""
        print(f"正在计算IC（调仓周期：{self.rebalance_period}）...")

        # 创建日期映射
        dates = sorted(self.merged_data['date'].unique())
        date_to_idx = {date: idx for idx, date in enumerate(dates)}

        # 添加日期索引列
        merged_data = self.merged_data.copy()
        merged_data['date_idx'] = merged_data['date'].map(date_to_idx)

        # 创建未来第rebalance_period期的收益率列
        merged_data['future_return'] = merged_data.groupby('order_book_id')['return'].shift(-self.rebalance_period)

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

        # 使用groupby计算IC
        ic_series = merged_data.groupby('date').apply(calculate_ic_for_date)

        return ic_series

    def calculate_icir(self, method='spearman'):
        ic_series = self.calculate_ic(method).dropna()
        if len(ic_series) == 0:
            return {
                'IC_mean': np.nan,
                'IC_std': np.nan,
                'ICIR': np.nan,
                'IC_positive_ratio': np.nan,
                'IC_skew': np.nan,
                'IC_kurtosis': np.nan,
                'IC_tvalue': np.nan,
                'IC_pvalue': np.nan,
                'IC_series': ic_series
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
            'IC_pvalue': p_value,
            'IC_series': ic_series
        }

    def create_decile_groups(self, n_groups=10):
        """优化的分组创建，使用groupby"""
        print("正在创建分组...")

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

            # 创建Series，索引为原始group的索引
            result = pd.Series(index=group.index, dtype=float)
            result.loc[factor.index] = groups
            return result

        # 使用groupby创建分组
        group_series = self.merged_data.groupby('date').apply(create_groups_for_date)

        # 创建分组DataFrame，保持原始索引
        group_df = pd.DataFrame({
            'date': self.merged_data['date'],
            'order_book_id': self.merged_data['order_book_id'],
            'group': group_series.values
        })

        # 去除缺失值
        group_df = group_df.dropna()

        return group_df

    def calculate_group_returns(self, n_groups=10):
        """优化的分组收益率计算，支持调仓周期"""
        print(f"正在计算分组收益率（调仓周期：{self.rebalance_period}）...")

        # 创建分组
        group_labels = self.create_decile_groups(n_groups)

        # 添加未来第rebalance_period期的收益率
        merged_data = self.merged_data.copy()
        merged_data['future_return'] = merged_data.groupby('order_book_id')['return'].shift(-self.rebalance_period)

        # 合并分组信息
        merged_with_groups = pd.merge(
            merged_data[['date', 'order_book_id', 'future_return'] + 
                       [col for col in ['limit_down_flag', 'suspended'] 
                        if col in merged_data.columns]],
            group_labels,
            on=['date', 'order_book_id'],
            how='inner'
        )

        # 去除缺失值
        merged_with_groups = merged_with_groups.dropna()

        # 应用卖出过滤（在计算收益率时）
        def calculate_group_return_with_filter(group):
            # 过滤掉不能卖出的股票（跌停、停牌）
            filtered_group = self._filter_tradable_stocks(group, for_buy=False)
            if len(filtered_group) == 0:
                return np.nan
            return filtered_group['future_return'].mean()

        # 按日期和分组计算平均收益率（应用卖出过滤）
        group_returns = merged_with_groups.groupby(['date', 'group']).apply(calculate_group_return_with_filter).unstack(fill_value=np.nan)

        # 确保所有分组都存在
        for g in range(n_groups):
            if g not in group_returns.columns:
                group_returns[g] = np.nan

        # 按分组顺序排列
        group_returns = group_returns.reindex(columns=range(n_groups))

        return {
            'group_returns': group_returns,
            'group_labels': group_labels
        }

    def calculate_cumulative_returns(self, n_groups=10):
        group_data = self.calculate_group_returns(n_groups)
        group_returns = group_data['group_returns']
        cumulative_returns = (1 + group_returns).cumprod()
        return cumulative_returns

    def calculate_long_short_returns(self, n_groups=10):
        """计算多空组合收益率"""
        group_data = self.calculate_group_returns(n_groups)
        group_returns = group_data['group_returns']

        # 多头组（最高分组）和空头组（最低分组）
        long_group = n_groups - 1  # 最高分组
        short_group = 0  # 最低分组

        long_returns = group_returns[long_group].dropna()
        short_returns = group_returns[short_group].dropna()

        # 多空收益率 = 多头收益率 - 空头收益率
        long_short_returns = long_returns - short_returns

        # 计算累计收益率
        cumulative_ls_returns = (1 + long_short_returns).cumprod()

        # 计算统计指标
        ls_stats = {
            'mean_return': long_short_returns.mean(),
            'std_return': long_short_returns.std(),
            'sharpe_ratio': long_short_returns.mean() / long_short_returns.std() if long_short_returns.std() != 0 else np.nan,
            'max_drawdown': self._calculate_max_drawdown(cumulative_ls_returns),
            'win_rate': (long_short_returns > 0).mean(),
            'total_return': cumulative_ls_returns.iloc[-1] - 1 if len(cumulative_ls_returns) > 0 else np.nan
        }

        return {
            'long_short_returns': long_short_returns,
            'cumulative_ls_returns': cumulative_ls_returns,
            'long_returns': long_returns,
            'short_returns': short_returns,
            'stats': ls_stats
        }

    def _calculate_max_drawdown(self, cumulative_returns):
        if len(cumulative_returns) == 0:
            return np.nan
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()

    def _annual_stats_df(self, ic_series, long_returns, short_returns, long_short_returns):
        ic_series = ic_series.copy()
        long_returns = long_returns.copy()
        short_returns = short_returns.copy()
        long_short_returns = long_short_returns.copy()
        for s in [ic_series, long_returns, short_returns, long_short_returns]:
            if not isinstance(s.index, pd.DatetimeIndex):
                s.index = pd.to_datetime(s.index)
        common_index = ic_series.index.intersection(long_returns.index).intersection(short_returns.index).intersection(long_short_returns.index)
        ic_series = ic_series.reindex(common_index)
        long_returns = long_returns.reindex(common_index)
        short_returns = short_returns.reindex(common_index)
        long_short_returns = long_short_returns.reindex(common_index)
        years = ic_series.index.to_series().dt.year.unique()
        stats = []
        for year in sorted(years):
            mask = ic_series.index.year == year
            ic_mean = ic_series[mask].mean()
            long_ret = long_returns[mask]
            short_ret = short_returns[mask]
            ls_ret = long_short_returns[mask]
            long_annual = (1 + long_ret).prod() - 1 if long_ret.notna().sum() > 0 else np.nan
            short_annual = (1 + short_ret).prod() - 1 if short_ret.notna().sum() > 0 else np.nan
            sharpe = np.nan
            if ls_ret.std() and ls_ret.notna().sum() > 1:
                sharpe = (ls_ret.mean() / ls_ret.std()) * np.sqrt(252)
            cum = (1 + ls_ret.fillna(0)).cumprod()
            if len(cum) > 0:
                running_max = cum.expanding().max()
                drawdown = (cum - running_max) / running_max
                max_dd = drawdown.min()
            else:
                max_dd = np.nan
            stats.append({
                '年度': str(year),
                'IC均值': ic_mean,
                '多头收益': long_annual,
                '空头收益': short_annual,
                '多空夏普': sharpe,
                '多空最大回撤': max_dd
            })
        df = pd.DataFrame(stats)
        for col in ['多头收益', '空头收益', '多空最大回撤']:
            df[col] = df[col] * 100
        return df

    def plot_full_analysis(self, method='spearman', n_groups=10, figsize=(14, 22), show_plot=True, 
                          precomputed_data=None, save_path=None):
        """
        新排版：两列四行！
        """
        if precomputed_data is None:
            print("正在计算分析数据...")
            ic_stats = self.calculate_icir(method)
            ic_series = ic_stats['IC_series']
            ic_df = ic_series.to_frame('ic')
            ic_df.index = pd.to_datetime(ic_df.index)
            ic_cum = ic_df['ic'].cumsum()
            ic_monthly_mean = ic_df.resample('M').mean()
            ic_monthly_mean.index = ic_monthly_mean.index.to_period('M').to_timestamp()
            print("正在计算分组收益数据...")

            # 缓存分组数据，避免重复计算
            group_data = self.calculate_group_returns(n_groups)
            group_returns = group_data['group_returns']

            # 计算累计收益率
            cumulative_returns = (1 + group_returns).cumprod()

            # 计算多空组合数据
            long_group = n_groups - 1  # 最高分组
            short_group = 0  # 最低分组

            long_returns = group_returns[long_group].dropna()
            short_returns = group_returns[short_group].dropna()

            # 多空收益率 = 多头收益率 - 空头收益率
            long_short_returns = long_returns - short_returns

            # 计算累计收益率
            cumulative_ls_returns = (1 + long_short_returns).cumprod()
            running_max = cumulative_ls_returns.expanding().max()
            drawdown = (cumulative_ls_returns - running_max) / running_max

            # 计算统计指标
            ls_stats = {
                'mean_return': long_short_returns.mean(),
                'std_return': long_short_returns.std(),
                'sharpe_ratio': long_short_returns.mean() / long_short_returns.std() if long_short_returns.std() != 0 else np.nan,
                'max_drawdown': self._calculate_max_drawdown(cumulative_ls_returns),
                'win_rate': (long_short_returns > 0).mean(),
                'total_return': cumulative_ls_returns.iloc[-1] - 1 if len(cumulative_ls_returns) > 0 else np.nan
            }

            avg_returns = group_returns.mean()
        else:
            # 使用预计算的数据
            ic_stats = precomputed_data['ic_stats']
            ic_series = precomputed_data['ic_series']
            ic_df = ic_series.to_frame('ic')
            ic_df.index = pd.to_datetime(ic_df.index)
            ic_cum = ic_df['ic'].cumsum()
            ic_monthly_mean = ic_df.resample('M').mean()
            ic_monthly_mean.index = ic_monthly_mean.index.to_period('M').to_timestamp()

            group_returns = precomputed_data['group_returns']['group_returns']
            cumulative_returns = precomputed_data['cumulative_returns']
            long_returns = precomputed_data['long_short_returns']['long_returns']
            short_returns = precomputed_data['long_short_returns']['short_returns']
            long_short_returns = precomputed_data['long_short_returns']['long_short_returns']
            cumulative_ls_returns = precomputed_data['long_short_returns']['cumulative_ls_returns']
            running_max = cumulative_ls_returns.expanding().max()
            drawdown = (cumulative_ls_returns - running_max) / running_max
            ls_stats = precomputed_data['long_short_stats']
            avg_returns = group_returns.mean()
        stats = ls_stats
        ic_stats_text = (
            f"\nIC均值: {ic_stats['IC_mean']:.4f}"
            f"\nIC标准差: {ic_stats['IC_std']:.4f}"
            f"\nICIR: {ic_stats['ICIR']:.4f}"
            f"\nIC正比例: {ic_stats['IC_positive_ratio']:.2%}"
            f"\nIC偏度: {ic_stats['IC_skew']:.4f}"
            f"\nIC峰度: {ic_stats['IC_kurtosis']:.4f}"
            f"\nt值: {ic_stats['IC_tvalue']:.4f}"
            f"\np值: {ic_stats['IC_pvalue']:.4g}\n"
        )
        stats_text = (
            f"\n平均收益: {stats['mean_return']:.4f}"
            f"\n收益率标准差: {stats['std_return']:.4f}"
            f"\n夏普比率: {stats['sharpe_ratio']:.4f}"
            f"\n胜率: {stats['win_rate']:.2%}"
            f"\n最大回撤: {stats['max_drawdown']:.2%}"
            f"\n总收益: {stats['total_return']:.2%}\n"
        )
        annual_df = self._annual_stats_df(
            ic_series,
            long_returns,
            short_returns,
            long_short_returns
        )

        # 新排版：两列四行
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, 2, height_ratios=[2, 2, 1.2, 2])

        # 1. IC月度均值柱状图和IC累计值复合图（第1行第1列）
        ax1 = fig.add_subplot(gs[0, 0])
        color_bar = 'tab:blue'
        color_cum = 'tab:red'
        bar_x = ic_monthly_mean.index
        bar_y = ic_monthly_mean['ic'].values
        total_months = len(bar_x)
        if total_months <= 12:
            tick_indices = range(total_months)
            tick_labels = [d.strftime('%Y-%m') for d in bar_x]
        elif total_months <= 36:
            tick_indices = range(0, total_months, 3)
            tick_labels = [bar_x[i].strftime('%Y-%m') for i in tick_indices]
        elif total_months <= 60:
            tick_indices = range(0, total_months, 6)
            tick_labels = [bar_x[i].strftime('%Y-%m') for i in tick_indices]
        else:
            tick_indices = range(0, total_months, 12)
            tick_labels = [bar_x[i].strftime('%Y') for i in tick_indices]
        ax1.bar(bar_x, bar_y, width=20, color=color_bar, alpha=0.7, label='IC月度均值', align='center')
        ax1.set_ylabel('IC月度均值', color=color_bar)
        ax1.set_xlabel('月份')
        ax1.set_title(f'{self.factor_name} - IC月度均值 & IC累计值 (调仓周期: {self.rebalance_period})')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='y', labelcolor=color_bar)
        ax1.set_xticks([bar_x[i] for i in tick_indices])
        ax1.set_xticklabels(tick_labels, rotation=45)
        ax1_right = ax1.twinx()
        ax1_right.plot(ic_cum.index, ic_cum.values, color=color_cum, linewidth=2, label='IC累计值')
        ax1_right.set_ylabel('IC累计值', color=color_cum)
        ax1_right.tick_params(axis='y', labelcolor=color_cum)
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax1_right.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')

        # 2. 各分组累计收益率（第1行第2列）
        ax2 = fig.add_subplot(gs[0, 1])
        for group in range(n_groups):
            if group in cumulative_returns.columns:
                # 仅用于分箱图，使用底数为10的对数收益率
                log_returns = np.log10(cumulative_returns[group].replace(0, np.nan))
                ax2.plot(cumulative_returns.index, log_returns,
                         label=f'分组{group+1}', alpha=0.8)
        ax2.set_title(f'{self.factor_name} - 各分组累计对数收益率(log10)')
        ax2.set_ylabel('累计对数收益率(log10)')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. 多空累计收益和动态回撤（第2行第1列）
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(cumulative_ls_returns.index, cumulative_ls_returns.values, color='red', linewidth=2, label='多空累计收益')
        ax3.set_title(f'{self.factor_name} - 多空累计收益及动态回撤 (调仓周期: {self.rebalance_period})')
        ax3.set_ylabel('累计收益')
        ax3.set_xlabel('日期')
        ax3.grid(True, alpha=0.3)
        if len(drawdown) > 0:
            ax3_right = ax3.twinx()
            ax3_right.fill_between(drawdown.index, 0, drawdown.values, color='gray', alpha=0.3, label='动态回撤')
            ax3_right.set_ylabel('动态回撤', color='gray')
            ax3_right.tick_params(axis='y', labelcolor='gray')
            ax3_right.invert_yaxis()
            ax3_right.yaxis.set_ticks_position('none')
            ax3_right.grid(False)
        handles, labels = ax3.get_legend_handles_labels()
        ax3.legend(handles, labels, loc='upper left', fontsize=8)

        # 4. 分组平均收益率（第2行第2列）
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.bar(range(n_groups), avg_returns.values, alpha=0.7)
        ax4.set_title(f'{self.factor_name} - 各分组平均收益率')
        ax4.set_xlabel('分组')
        ax4.set_ylabel('平均收益率')
        ax4.grid(True, alpha=0.3)

        # 5. IC统计信息文本（第3行第1列）
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.text(0.05, 0.5, ic_stats_text, transform=ax5.transAxes,
                 fontsize=12, verticalalignment='center', fontname='Microsoft YaHei')
        ax5.set_title(f'{self.factor_name} - IC统计信息', fontsize=13)
        ax5.axis('off')

        # 6. 多空组合统计信息（第3行第2列）
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.text(0.05, 0.5, stats_text, transform=ax6.transAxes,
                 fontsize=12, verticalalignment='center', fontname='Microsoft YaHei')
        ax6.set_title(f'{self.factor_name} - 多空组合统计信息', fontsize=13)
        ax6.axis('off')

        # 7. 年度统计表（第4行第1列，跨两列）
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        table_data = []
        for i, row in annual_df.iterrows():
            table_data.append([
                row['年度'],
                f"{row['IC均值']:.4f}",
                f"{row['多头收益']:.2f}%",
                f"{row['空头收益']:.2f}%",
                f"{row['多空夏普']:.2f}" if not pd.isna(row['多空夏普']) else "",
                f"{row['多空最大回撤']:.2f}%" if not pd.isna(row['多空最大回撤']) else ""
            ])
        col_labels = ['年度', 'IC均值', '多头收益', '空头收益', '多空夏普', '多空最大回撤']
        table = ax7.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.1, 1.6)  # 这里将高度缩小一点（原来2.0，现为1.6）
        ax7.set_title('', fontsize=13, pad=10)

        plt.tight_layout(rect=[0, 0, 1, 1])
        
        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存到: {save_path}")
        
        if show_plot:
            plt.show()
        return fig

    def generate_report(self, n_groups=10, method='spearman', save_path=None):
        print("=" * 50)
        print(f"因子分析报告: {self.factor_name}")
        print(f"调仓周期: {self.rebalance_period}")
        print("=" * 50)
        ic_stats = self.calculate_icir(method)
        print(f"\nIC分析结果 ({method}相关系数):")
        print(f"IC均值: {ic_stats['IC_mean']:.4f}")
        print(f"IC标准差: {ic_stats['IC_std']:.4f}")
        print(f"ICIR: {ic_stats['ICIR']:.4f}")
        print(f"IC正比例: {ic_stats['IC_positive_ratio']:.2%}")
        print(f"IC偏度: {ic_stats['IC_skew']:.4f}")
        print(f"IC峰度: {ic_stats['IC_kurtosis']:.4f}")
        print(f"t值: {ic_stats['IC_tvalue']:.4f}")
        print(f"p值: {ic_stats['IC_pvalue']:.4g}")

        # 缓存分组数据，避免重复计算
        group_data = self.calculate_group_returns(n_groups)
        group_returns = group_data['group_returns']

        # 计算多空组合数据
        long_group = n_groups - 1  # 最高分组
        short_group = 0  # 最低分组

        long_returns = group_returns[long_group].dropna()
        short_returns = group_returns[short_group].dropna()

        # 多空收益率 = 多头收益率 - 空头收益率
        long_short_returns = long_returns - short_returns

        # 计算累计收益率
        cumulative_ls_returns = (1 + long_short_returns).cumprod()

        # 计算统计指标
        stats = {
            'mean_return': long_short_returns.mean(),
            'std_return': long_short_returns.std(),
            'sharpe_ratio': long_short_returns.mean() / long_short_returns.std() if long_short_returns.std() != 0 else np.nan,
            'max_drawdown': self._calculate_max_drawdown(cumulative_ls_returns),
            'win_rate': (long_short_returns > 0).mean(),
            'total_return': cumulative_ls_returns.iloc[-1] - 1 if len(cumulative_ls_returns) > 0 else np.nan
        }

        print(f"\n多空组合分析结果 ({n_groups}分组):")
        print(f"平均收益: {stats['mean_return']:.4f}")
        print(f"收益率标准差: {stats['std_return']:.4f}")
        print(f"夏普比率: {stats['sharpe_ratio']:.4f}")
        print(f"胜率: {stats['win_rate']:.2%}")
        print(f"最大回撤: {stats['max_drawdown']:.2%}")
        print(f"总收益: {stats['total_return']:.2%}")

        # 计算累计收益率
        cumulative_returns = (1 + group_returns).cumprod()

        # 准备预计算数据
        precomputed_data = {
            'ic_stats': ic_stats,
            'ic_series': ic_stats['IC_series'],
            'group_returns': group_data,
            'cumulative_returns': cumulative_returns,
            'long_short_returns': {
                'long_short_returns': long_short_returns,
                'cumulative_ls_returns': cumulative_ls_returns,
                'long_returns': long_returns,
                'short_returns': short_returns,
                'stats': stats
            },
            'long_short_stats': stats
        }

        self.plot_full_analysis(method=method, n_groups=n_groups, precomputed_data=precomputed_data, save_path=save_path)

        return {
            'ic_stats': ic_stats,
            'long_short_stats': stats,
            'group_returns': group_data,
            'cumulative_returns': cumulative_returns,
            'long_short_returns': {
                'long_short_returns': long_short_returns,
                'cumulative_ls_returns': cumulative_ls_returns,
                'long_returns': long_returns,
                'short_returns': short_returns,
                'stats': stats
            }
        }


def analyze_single_factor(factor_data, returns_data, factor_name='factor', n_groups=10, method='spearman', rebalance_period=1, save_path=None):
    analyzer = SingleFactorAnalyzer(factor_data, returns_data, factor_name, rebalance_period)
    return analyzer.generate_report(n_groups, method, save_path)