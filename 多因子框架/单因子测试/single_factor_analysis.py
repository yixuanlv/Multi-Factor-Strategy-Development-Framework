import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from tqdm import tqdm
from pyecharts import options as opts
from pyecharts.charts import Line, Bar, Scatter, HeatMap
from pyecharts.commons.utils import JsCode
from pyecharts.globals import ThemeType

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class SingleFactorAnalyzer:
    """
    单因子分析工具类
    输入：长格式数据框，包含date、order_book_id、factor_value、close列
    """

    def __init__(self, factor_data, returns_data, factor_name='factor', rebalance_period=1, enable_stock_filter=True, barra_data=None):
        self.factor_data = factor_data
        self.returns_data = returns_data
        self.factor_name = factor_name
        self.rebalance_period = rebalance_period
        self.enable_stock_filter = enable_stock_filter
        self.barra_data = barra_data
        
        self._align_data()
        self._create_rebalance_dates()

    def _align_data(self):
        """对齐因子数据和收盘价数据，并计算收益率"""
        # 转换日期格式
        self.factor_data['date'] = pd.to_datetime(self.factor_data['date'])
        self.returns_data['date'] = pd.to_datetime(self.returns_data['date'])

        # 计算收益率
        returns_data = self.returns_data.copy()
        returns_data['return'] = returns_data.groupby('order_book_id')['close'].pct_change()
        
        # 合并数据
        merged_data = pd.merge(
            self.factor_data[['date', 'order_book_id', 'factor_value']],
            returns_data[['date', 'order_book_id', 'return', 'close'] + 
                        [col for col in ['limit_up_flag', 'limit_down_flag', 'ST', 'suspended'] 
                         if col in returns_data.columns]],
            on=['date', 'order_book_id'],
            how='inner'
        ).dropna()

        self.merged_data = merged_data
        
    def _create_rebalance_dates(self):
        """创建调仓日序列"""
        all_dates = sorted(self.merged_data['date'].unique())
        self.rebalance_dates = all_dates[::self.rebalance_period] if self.rebalance_period > 1 else all_dates
        
    def _is_rebalance_date(self, date):
        """判断是否为调仓日"""
        return date in self.rebalance_dates

    def _filter_stocks(self, data, for_buy=True):
        """股票过滤函数"""
        if not self.enable_stock_filter:
            return data
            
        if for_buy:
            # 买入过滤：涨停、ST、停牌
            filter_cols = ['limit_up_flag', 'ST', 'suspended']
        else:
            # 卖出过滤：跌停、停牌
            filter_cols = ['limit_down_flag', 'suspended']
        
        # 创建过滤掩码
        mask = pd.Series(True, index=data.index)
        for col in filter_cols:
            if col in data.columns:
                mask &= ~data[col].astype(bool)
        
        return data[mask]

    def calculate_ic(self, method='spearman', lag=1):
        """计算IC值"""
        if lag == 1:
            print(f"  计算IC1-IC5值 ({method}相关系数)...")
        
        # 创建未来收益率列
        merged_data = self.merged_data.copy()
        merged_data['future_return'] = merged_data.groupby('order_book_id')['return'].shift(-lag)
        
        # 过滤可交易股票
        if self.enable_stock_filter:
            merged_data = self._filter_stocks(merged_data, for_buy=True)
        
        # 按日期分组计算IC
        ic_series = merged_data.groupby('date').apply(
            lambda x: self._calculate_correlation(x['factor_value'], x['future_return'], method)
        )
        
        if lag == 5:
            print(f"    IC1-IC5计算完成")
        return ic_series

    def _calculate_correlation(self, x, y, method='spearman'):
        """计算相关系数"""
        if len(x.dropna()) < 10:
            return np.nan
            
        try:
            if method.lower() == 'spearman':
                correlation, _ = stats.spearmanr(x, y)
            else:
                correlation = np.corrcoef(x, y)[0, 1]
            return correlation
        except:
            return np.nan

    def calculate_icir(self, method='spearman'):
        """计算ICIR统计指标"""
        # 计算IC1到IC5的统计指标
        ic_stats_all = {}
        
        for lag in range(1, 6):
            ic_series = self.calculate_ic(method, lag).dropna()
            if len(ic_series) == 0:
                ic_stats_all[f'IC{lag}'] = {
                    'IC_mean': np.nan, 'IC_std': np.nan, 'ICIR': np.nan,
                    'IC_positive_ratio': np.nan, 'IC_skew': np.nan, 'IC_kurtosis': np.nan,
                    'IC_tvalue': np.nan, 'IC_pvalue': np.nan, 'IC_series': ic_series
                }
            else:
                ic_mean = ic_series.mean()
                ic_std = ic_series.std()
                icir = ic_mean / ic_std if ic_std != 0 else np.nan
                ic_positive_ratio = (ic_series > 0).mean()
                ic_skew = stats.skew(ic_series)
                ic_kurtosis = stats.kurtosis(ic_series, fisher=True)
                t_stat, p_value = stats.ttest_1samp(ic_series, 0, nan_policy='omit')
                
                ic_stats_all[f'IC{lag}'] = {
                    'IC_mean': ic_mean, 'IC_std': ic_std, 'ICIR': icir,
                    'IC_positive_ratio': ic_positive_ratio, 'IC_skew': ic_skew, 'IC_kurtosis': ic_kurtosis,
                    'IC_tvalue': t_stat, 'IC_pvalue': p_value, 'IC_series': ic_series
                }
        
        # 返回IC1的统计指标作为主要结果，同时包含所有IC的统计
        ic_stats_all['IC_series'] = ic_stats_all['IC1']['IC_series']
        return ic_stats_all

    def create_decile_groups(self, n_groups=10):
        """创建分组并计算收益率"""
        print(f"  创建 {n_groups} 分组...")
        
        # 只在调仓日创建分组
        rebalance_dates_set = set(self.rebalance_dates)
        merged_data_filtered = self.merged_data[self.merged_data['date'].isin(rebalance_dates_set)].copy()
        
        if len(merged_data_filtered) == 0:
            empty_df = pd.DataFrame(columns=['date', 'order_book_id', 'group'])
            empty_returns = pd.DataFrame(index=pd.DatetimeIndex([]), columns=range(n_groups))
            return {'group_returns': empty_returns, 'group_labels': empty_df}
        
        # 过滤可交易股票
        if self.enable_stock_filter:
            merged_data_filtered = self._filter_stocks(merged_data_filtered, for_buy=True)
        
        # 创建分组
        group_data_list = []
        for date, group in merged_data_filtered.groupby('date'):
            if len(group) >= n_groups:
                factor = group['factor_value'].dropna()
                if len(factor) >= n_groups:
                    # 创建分组
                    ranks = factor.rank(method='first', ascending=True)
                    group_size = len(ranks) // n_groups
                    groups = (ranks - 1) // group_size
                    groups = groups.clip(upper=n_groups-1)
                    
                    date_groups = pd.DataFrame({
                        'date': date,
                        'order_book_id': group.loc[factor.index, 'order_book_id'].values,
                        'group': groups.values
                    })
                    group_data_list.append(date_groups)
        
        if not group_data_list:
            empty_df = pd.DataFrame(columns=['date', 'order_book_id', 'group'])
            empty_returns = pd.DataFrame(index=pd.DatetimeIndex([]), columns=range(n_groups))
            return {'group_returns': empty_returns, 'group_labels': empty_df}
        
        # 合并所有分组数据
        group_df = pd.concat(group_data_list, ignore_index=True)
        
        # 计算未来收益率
        print(f"    计算未来收益率...")
        stock_ids = group_df['order_book_id'].unique()
        merged_data_subset = self.merged_data[self.merged_data['order_book_id'].isin(stock_ids)].copy()
        merged_data_subset['future_return'] = merged_data_subset.groupby('order_book_id')['return'].shift(-self.rebalance_period)
        
        # 合并分组和收益率
        merged_with_groups = pd.merge(
            group_df,
            merged_data_subset[['date', 'order_book_id', 'future_return']],
            on=['date', 'order_book_id'],
            how='left'
        )
        
        # 计算分组收益率
        group_returns = merged_with_groups.groupby(['date', 'group'])['future_return'].mean().unstack(fill_value=np.nan)
        
        # 处理最后几天的NaN值
        valid_dates = group_returns.dropna(how='all').index
        if len(valid_dates) > 0:
            group_returns = group_returns.loc[valid_dates]
        
        # 确保所有分组都存在
        for g in range(n_groups):
            if g not in group_returns.columns:
                group_returns[g] = np.nan
        group_returns = group_returns.reindex(columns=range(n_groups))

        print(f"    分组创建完成，共 {len(group_returns)} 个交易日")
        return {'group_returns': group_returns, 'group_labels': group_df}

    def calculate_returns(self, n_groups=10):
        """计算收益率数据"""
        group_data = self.create_decile_groups(n_groups)
        group_returns = group_data['group_returns']
        
        # 计算累计收益率
        group_returns_filled = group_returns.fillna(0)
        cumulative_returns = (1 + group_returns_filled).cumprod()
        
        # 计算多空组合收益率
        long_short_data = self._calculate_long_short_returns(n_groups, group_data)
        
        return {
            'group_returns': group_data,
            'cumulative_returns': cumulative_returns,
            'long_short_returns': long_short_data
        }
    
    def _calculate_long_short_returns(self, n_groups=10, group_data=None):
        """计算多空组合收益率"""
        if group_data is None:
            group_data = self.create_decile_groups(n_groups)
        
        group_returns = group_data['group_returns']
        
        if group_returns.empty:
            return {
                'long_short_returns': pd.Series(dtype=float),
                'cumulative_ls_returns': pd.Series(dtype=float),
                'long_returns': pd.Series(dtype=float),
                'short_returns': pd.Series(dtype=float),
                'stats': {
                    'mean_return': np.nan, 'std_return': np.nan, 'sharpe_ratio': np.nan,
                    'max_drawdown': np.nan, 'win_rate': np.nan, 'total_return': np.nan
                }
            }
        
        # 根据IC自动判断多空方向
        # 先计算IC均值来判断因子方向
        ic_series = self.calculate_ic()
        ic_mean = ic_series.mean()
        
        if ic_mean < 0:
            # IC为负，因子与收益负相关，多1空10
            long_group = 0  # 最低分组
            short_group = n_groups - 1  # 最高分组
        else:
            # IC为正，因子与收益正相关，多10空1
            long_group = n_groups - 1  # 最高分组
            short_group = 0  # 最低分组
        
        long_returns = group_returns[long_group] if long_group in group_returns.columns else pd.Series(dtype=float)
        short_returns = group_returns[short_group] if short_group in group_returns.columns else pd.Series(dtype=float)
        long_short_returns = long_returns - short_returns
        
        # 计算累计收益率
        long_short_returns_filled = long_short_returns.fillna(0)
        cumulative_ls_returns = (1 + long_short_returns_filled).cumprod()
        
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
        """计算最大回撤"""
        if len(cumulative_returns) == 0:
            return np.nan
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()

    def calculate_barra_style_correlation(self):
        """计算Barra风格因子相关性 (彻底向量化版本)"""
        if self.barra_data is None:
            return None

        print("    计算Barra风格因子相关性...")

        # 获取风格因子列（前11列）
        barra_columns = self.barra_data.columns.tolist()
        style_factors = [col for col in barra_columns if col not in ['date', 'order_book_id']][:11]

        # 构造因子宽表 (T × N)
        factor_wide = self.factor_data.pivot(index='date', columns='order_book_id', values='factor_value')

        # 构造所有风格因子宽表，堆叠成 (T × N × K)
        barra_wide_all = []
        for f in style_factors:
            wide = self.barra_data.pivot(index='date', columns='order_book_id', values=f)
            barra_wide_all.append(wide)
        barra_wide_all = np.stack([df.reindex_like(factor_wide).to_numpy() for df in barra_wide_all], axis=-1)

        # 对齐
        common_dates = factor_wide.index
        X = factor_wide.to_numpy()                  # (T × N)
        Y = barra_wide_all                          # (T × N × K)
        T, N = X.shape
        K = Y.shape[2]

        print(f"      处理 {K} 个风格因子，{T} 个日期...")

        # mask (有效值位置)
        mask_x = ~np.isnan(X)                       # (T × N)
        mask_y = ~np.isnan(Y)                       # (T × N × K)
        mask = mask_x[:, :, None] & mask_y          # (T × N × K)

        # 中心化
        Xc = np.where(mask, X[:, :, None], np.nan)  # (T × N × K)
        Yc = np.where(mask, Y, np.nan)

        mean_x = np.nanmean(Xc, axis=1, keepdims=True)   # (T × 1 × K)
        mean_y = np.nanmean(Yc, axis=1, keepdims=True)   # (T × 1 × K)

        Xc = Xc - mean_x
        Yc = Yc - mean_y

        # 协方差与方差
        cov = np.nansum(Xc * Yc, axis=1)                # (T × K)
        var_x = np.nansum(Xc**2, axis=1)                # (T × K)
        var_y = np.nansum(Yc**2, axis=1)                # (T × K)

        corr = cov / np.sqrt(var_x * var_y)             # (T × K)

        # 最小样本数过滤 (例如>=10只股票)
        valid_counts = np.sum(mask, axis=1)             # (T × K)
        corr[valid_counts < 10] = np.nan

        # 转换为 DataFrame
        corr_df = pd.DataFrame(corr, index=common_dates, columns=style_factors)

        # rolling 计算
        rolling_corr = corr_df.rolling(window=20, min_periods=5).mean()

        print(f"    完成，共计算 {len(corr_df)} 个交易日")

        return {
            'daily_correlation': corr_df,
            'rolling_correlation': rolling_corr,
            'style_factors': style_factors
        }

        
    def calculate_industry_exposure(self, n_groups=10):
        """计算行业因子暴露差异 """
        if self.barra_data is None:
            return None

        print("    计算Barra行业因子暴露...")

        # 获取行业因子列
        barra_columns = self.barra_data.columns.tolist()
        industry_factors = [col for col in barra_columns if col not in ['date', 'order_book_id']][-31:]

        # 获取分组数据
        group_data = self.create_decile_groups(n_groups)
        group_labels = group_data['group_labels']
        if group_labels.empty:
            return None

        long_group = n_groups - 1
        short_group = 0

        # 合并 group_labels 和 barra_data，一次性处理
        merged = pd.merge(
            group_labels,
            self.barra_data[['date', 'order_book_id'] + industry_factors],
            on=['date', 'order_book_id'],
            how='inner'
        )

        # 按日期+组别，计算行业暴露均值
        exposure = merged.groupby(['date', 'group'])[industry_factors].mean()

        # 多头和空头分别取出来
        long_exposure = exposure.xs(long_group, level='group')
        short_exposure = exposure.xs(short_group, level='group')

        # 计算差异
        exposure_diff = long_exposure - short_exposure

        # 半年聚合 - 优化精度
        long_exposure.index = pd.to_datetime(long_exposure.index)
        short_exposure.index = pd.to_datetime(short_exposure.index)
        exposure_diff.index = pd.to_datetime(exposure_diff.index)

        # 按季度聚合
        quarter_diff = exposure_diff.resample('Q').mean()
        
        # 设置最小阈值，减少噪声
        threshold = 0.00001
        quarter_diff = quarter_diff.where(quarter_diff.abs() >= threshold, 0)

        print(f"    完成，共计算 {len(quarter_diff)} 个季度")

        return {
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'exposure_diff': exposure_diff,
            'quarter_diff': quarter_diff,
            'industry_factors': industry_factors
        }




    def _annual_stats_df(self, ic_series, long_returns, short_returns, long_short_returns):
        """计算年度统计"""
        # 对齐数据
        for s in [ic_series, long_returns, short_returns, long_short_returns]:
            if not isinstance(s.index, pd.DatetimeIndex):
                s.index = pd.to_datetime(s.index)
        
        common_index = ic_series.index.intersection(long_returns.index).intersection(short_returns.index).intersection(long_short_returns.index)
        ic_series = ic_series.reindex(common_index)
        long_returns = long_returns.reindex(common_index)
        short_returns = short_returns.reindex(common_index)
        long_short_returns = long_short_returns.reindex(common_index)
        
        # 按年度计算统计
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
            max_dd = np.nan
            if len(cum) > 0:
                running_max = cum.expanding().max()
                drawdown = (cum - running_max) / running_max
                max_dd = drawdown.min()
            
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

    def plot_full_analysis(self, method='spearman', n_groups=10, figsize=(16, 28), show_plot=True, 
                          precomputed_data=None, save_path=None, show_log_returns=True):
        """绘制完整分析图表 - 使用pyecharts生成HTML"""
        if precomputed_data is None:
            # 计算数据
            ic_stats = self.calculate_icir(method)
            ic_series = ic_stats['IC_series']
            returns_data = self.calculate_returns(n_groups)
            group_data = returns_data['group_returns']
            group_returns = group_data['group_returns']
            cumulative_returns = returns_data['cumulative_returns']
            long_short_data = returns_data['long_short_returns']
            
            long_short_returns = long_short_data['long_short_returns']
            cumulative_ls_returns = long_short_data['cumulative_ls_returns']
            long_returns = long_short_data['long_returns']
            short_returns = long_short_data['short_returns']
            ls_stats = long_short_data['stats']
            
            # 计算动态回撤
            running_max = cumulative_ls_returns.expanding().max()
            drawdown = (cumulative_ls_returns - running_max) / running_max
            
            # 计算分组累计收益
            group_cumulative_returns = cumulative_returns.iloc[-1] if len(cumulative_returns) > 0 else pd.Series([np.nan] * n_groups)
            
            # 计算Barra相关数据
            barra_style_corr = self.calculate_barra_style_correlation()
            barra_industry_exposure = self.calculate_industry_exposure(n_groups)
        else:
            # 使用预计算的数据
            ic_stats = precomputed_data['ic_stats']
            ic_series = precomputed_data['ic_series']
            group_returns = precomputed_data['group_returns']['group_returns']
            cumulative_returns = precomputed_data['cumulative_returns']
            long_returns = precomputed_data['long_short_returns']['long_returns']
            short_returns = precomputed_data['long_short_returns']['short_returns']
            long_short_returns = precomputed_data['long_short_returns']['long_short_returns']
            cumulative_ls_returns = precomputed_data['long_short_returns']['cumulative_ls_returns']
            running_max = cumulative_ls_returns.expanding().max()
            drawdown = (cumulative_ls_returns - running_max) / running_max
            ls_stats = precomputed_data['long_short_stats']
            group_cumulative_returns = cumulative_returns.iloc[-1] if len(cumulative_returns) > 0 else pd.Series([np.nan] * n_groups)
            
            barra_style_corr = precomputed_data.get('barra_style_corr')
            barra_industry_exposure = precomputed_data.get('barra_industry_exposure')
            
            if barra_style_corr is None:
                barra_style_corr = self.calculate_barra_style_correlation()
            if barra_industry_exposure is None:
                barra_industry_exposure = self.calculate_industry_exposure(n_groups)

        # 准备文本信息 - 使用IC1的统计信息
        ic1_stats = ic_stats.get('IC1', {})
        ic_stats_text = (
            f"IC1均值: {ic1_stats.get('IC_mean', np.nan):.4f}<br/>"
            f"IC1标准差: {ic1_stats.get('IC_std', np.nan):.4f}<br/>"
            f"IC1IR: {ic1_stats.get('ICIR', np.nan):.4f}<br/>"
            f"IC1正比例: {ic1_stats.get('IC_positive_ratio', np.nan):.2%}<br/>"
            f"IC1偏度: {ic1_stats.get('IC_skew', np.nan):.4f}<br/>"
            f"IC1峰度: {ic1_stats.get('IC_kurtosis', np.nan):.4f}<br/>"
            f"IC1t值: {ic1_stats.get('IC_tvalue', np.nan):.4f}<br/>"
            f"IC1p值: {ic1_stats.get('IC_pvalue', np.nan):.4g}"
        )
        
        stats_text = (
            f"平均收益: {ls_stats['mean_return']:.4f}<br/>"
            f"收益率标准差: {ls_stats['std_return']:.4f}<br/>"
            f"夏普比率: {ls_stats['sharpe_ratio']:.4f}<br/>"
            f"胜率: {ls_stats['win_rate']:.2%}<br/>"
            f"最大回撤: {ls_stats['max_drawdown']:.2%}<br/>"
            f"总收益: {ls_stats['total_return']:.2%}"
        )
        
        annual_df = self._annual_stats_df(ic_series, long_returns, short_returns, long_short_returns)

        # 创建pyecharts图表
        # 1. IC月度均值和累计值
        ic_df = ic_series.to_frame('ic')
        ic_df.index = pd.to_datetime(ic_df.index)
        ic_cum = ic_df['ic'].cumsum()
        ic_monthly_mean = ic_df.resample('M').mean()
        ic_monthly_mean.index = ic_monthly_mean.index.to_period('M').to_timestamp()
        
        ic_chart = (
            Bar()
            .add_xaxis([d.strftime('%Y-%m') for d in ic_monthly_mean.index])
            .add_yaxis("IC月度均值", ic_monthly_mean['ic'].round(4).tolist(), yaxis_index=0,
                      label_opts=opts.LabelOpts(is_show=False))
            .extend_axis(
                yaxis=opts.AxisOpts(
                    name="IC累计值",
                    type_="value",
                    position="right",
                )
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"{self.factor_name} - IC月度均值 & IC累计值 (调仓周期: {self.rebalance_period})"),
                xaxis_opts=opts.AxisOpts(
                    name="月份",
                    type_="category",
                    axislabel_opts=opts.LabelOpts(rotate=45)
                ),
                yaxis_opts=opts.AxisOpts(
                    name="IC月度均值",
                    is_scale=True,
                    axislabel_opts=opts.LabelOpts(is_show=False),
                    axisline_opts=opts.AxisLineOpts(is_show=False),
                    axistick_opts=opts.AxisTickOpts(is_show=False)
                ),
                datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
                legend_opts=opts.LegendOpts(pos_top="5%"),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
            )
        )
        
        ic_line = (
            Line()
            .add_xaxis([d.strftime('%Y-%m') for d in ic_cum.index])
            .add_yaxis("IC累计值", ic_cum.round(4).tolist(), yaxis_index=1,
                      label_opts=opts.LabelOpts(is_show=False),
                      symbol_size=0)
        )
        
        ic_chart.overlap(ic_line)

        # 2. 各分组累计收益率
        group_chart = Line()
        if show_log_returns:
            for group in range(n_groups):
                if group in cumulative_returns.columns:
                    log_returns = np.log10(cumulative_returns[group].replace(0, np.nan))
                    group_chart.add_xaxis([d.strftime('%Y-%m-%d') for d in cumulative_returns.index])
                    group_chart.add_yaxis(f"分组{group+1}", log_returns.round(4).tolist(),
                                       label_opts=opts.LabelOpts(is_show=False),
                                       symbol_size=0)
            title = f"{self.factor_name} - 各分组累计对数收益率(log10)"
        else:
            for group in range(n_groups):
                if group in cumulative_returns.columns:
                    group_chart.add_xaxis([d.strftime('%Y-%m-%d') for d in cumulative_returns.index])
                    group_chart.add_yaxis(f"分组{group+1}", cumulative_returns[group].round(4).tolist(),
                                       label_opts=opts.LabelOpts(is_show=False),
                                       symbol_size=0)
            title = f"{self.factor_name} - 各分组累计收益率"
        
        group_chart.set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            xaxis_opts=opts.AxisOpts(
                name="日期",
                type_="category",
                axislabel_opts=opts.LabelOpts(rotate=45)
            ),
                            yaxis_opts=opts.AxisOpts(
                    name="累计收益率",
                    is_scale=True,
                    axislabel_opts=opts.LabelOpts(is_show=False),
                    axisline_opts=opts.AxisLineOpts(is_show=False),
                    axistick_opts=opts.AxisTickOpts(is_show=False)
                ),
            datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
            legend_opts=opts.LegendOpts(pos_top="5%"),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
        )

        # 3. 多空累计收益和动态回撤
        ls_chart = (
            Line()
            .add_xaxis([d.strftime('%Y-%m-%d') for d in cumulative_ls_returns.index])
            .add_yaxis("多空累计收益", cumulative_ls_returns.round(4).tolist(), yaxis_index=0,
                      label_opts=opts.LabelOpts(is_show=False),
                      symbol_size=0)
            .extend_axis(
                yaxis=opts.AxisOpts(
                    name="动态回撤",
                    type_="value",
                    position="right",
                    is_inverse=True
                )
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"{self.factor_name} - 多空累计收益及动态回撤 (调仓周期: {self.rebalance_period})"),
                xaxis_opts=opts.AxisOpts(
                    name="日期",
                    type_="category",
                    axislabel_opts=opts.LabelOpts(rotate=45)
                ),
                yaxis_opts=opts.AxisOpts(
                    name="累计收益",
                    is_scale=True,
                    axislabel_opts=opts.LabelOpts(is_show=False),
                    axisline_opts=opts.AxisLineOpts(is_show=False),
                    axistick_opts=opts.AxisTickOpts(is_show=False)
                ),
                datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
                legend_opts=opts.LegendOpts(pos_top="5%"),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
            )
        )
        
        if len(drawdown) > 0:
            # 使用面积图显示动态回撤为灰色阴影
            drawdown_area = (
                Line()
                .add_xaxis([d.strftime('%Y-%m-%d') for d in drawdown.index])
                .add_yaxis(
                    "动态回撤", 
                    drawdown.round(4).tolist(), 
                    yaxis_index=1,
                    is_symbol_show=False,
                    linestyle_opts=opts.LineStyleOpts(width=0),
                    itemstyle_opts=opts.ItemStyleOpts(color="rgba(128,128,128,0.3)"),
                    label_opts=opts.LabelOpts(is_show=False)
                )
                .set_series_opts(
                    areastyle_opts=opts.AreaStyleOpts(
                        color="rgba(128,128,128,0.3)",
                        opacity=0.3
                    )
                )
            )
            ls_chart.overlap(drawdown_area)

        # 4. 分组累计收益率柱状图
        group_bar = (
            Bar()
            .add_xaxis([f"分组{i+1}" for i in range(n_groups)])
            .add_yaxis("累计收益率", group_cumulative_returns.round(4).tolist(),
                      label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"{self.factor_name} - 各分组累计收益率"),
                xaxis_opts=opts.AxisOpts(
                    name="分组",
                    type_="category"
                ),
                yaxis_opts=opts.AxisOpts(
                    name="累计收益率",
                    is_scale=True,
                    axislabel_opts=opts.LabelOpts(is_show=False),
                    axisline_opts=opts.AxisLineOpts(is_show=False),
                    axistick_opts=opts.AxisTickOpts(is_show=False)
                ),
                datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
                legend_opts=opts.LegendOpts(pos_top="5%"),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
            )
        )

        # 5. IC统计信息表格
        from pyecharts.components import Table
        
        # 准备IC1到IC5的统计信息 - 5列表格
        headers = ["指标", "IC1", "IC2", "IC3", "IC4", "IC5"]
        ic_table_data = []
        
        # 定义指标行
        metrics = [
            ("IC均值", "IC_mean", ":.4f"),
            ("IC标准差", "IC_std", ":.4f"),
            ("ICIR", "ICIR", ":.4f"),
            ("IC正比例", "IC_positive_ratio", ":.2%"),
            ("IC偏度", "IC_skew", ":.4f"),
            ("IC峰度", "IC_kurtosis", ":.4f"),
            ("t值", "IC_tvalue", ":.4f"),
            ("p值", "IC_pvalue", ":.4g")
        ]
        
        for metric_name, metric_key, format_str in metrics:
            row = [metric_name]
            for lag in range(1, 6):
                ic_key = f'IC{lag}'
                if ic_key in ic_stats:
                    value = ic_stats[ic_key].get(metric_key, np.nan)
                    if not pd.isna(value):
                        if format_str == ":.2%":
                            row.append(f"{value:.2%}")
                        elif format_str == ":.4g":
                            row.append(f"{value:.4g}")
                        else:
                            row.append(f"{value:.4f}")
                    else:
                        row.append("NaN")
                else:
                    row.append("NaN")
            ic_table_data.append(row)
        
        ic_table = (
            Table()
            .add(headers, ic_table_data)
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"{self.factor_name} - IC1到IC5统计信息", pos_left="center")
            )
        )

        # 6. 多空组合统计信息表格
        stats_table = (
            Table()
            .add(["指标", "数值"], [
                ["平均收益", f"{ls_stats['mean_return']:.4f}"],
                ["收益率标准差", f"{ls_stats['std_return']:.4f}"],
                ["夏普比率", f"{ls_stats['sharpe_ratio']:.4f}"],
                ["胜率", f"{ls_stats['win_rate']:.2%}"],
                ["最大回撤", f"{ls_stats['max_drawdown']:.2%}"],
                ["总收益", f"{ls_stats['total_return']:.2%}"]
            ])
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"{self.factor_name} - 多空组合统计信息", pos_left="center")
            )
        )

        # 7. Barra风格因子相关系数
        if barra_style_corr is not None:
            daily_corr = barra_style_corr['daily_correlation']
            style_factors = barra_style_corr['style_factors']
            
            barra_style_chart = Line()
            for factor in style_factors:
                if factor in daily_corr.columns:
                    barra_style_chart.add_xaxis([d.strftime('%Y-%m-%d') for d in daily_corr.index])
                    barra_style_chart.add_yaxis(factor, daily_corr[factor].round(4).tolist(),
                                             label_opts=opts.LabelOpts(is_show=False),
                                             symbol_size=0)
            
            barra_style_chart.set_global_opts(
                title_opts=opts.TitleOpts(title=f"{self.factor_name} - 与Barra风格因子相关系数（日度）"),
                xaxis_opts=opts.AxisOpts(
                    name="日期",
                    type_="category",
                    axislabel_opts=opts.LabelOpts(rotate=45)
                ),
                yaxis_opts=opts.AxisOpts(
                    name="相关系数",
                    is_scale=True,
                    min_=-1,
                    max_=1,
                    axislabel_opts=opts.LabelOpts(is_show=False),
                    axisline_opts=opts.AxisLineOpts(is_show=False),
                    axistick_opts=opts.AxisTickOpts(is_show=False)
                ),
                datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
                legend_opts=opts.LegendOpts(pos_top="5%"),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
            )
        else:
            from pyecharts.components import Table
            barra_style_chart = (
                Table()
                .add(["Barra风格因子相关性"], [["无Barra数据"]])
                .set_global_opts(
                    title_opts=opts.TitleOpts(title=f"{self.factor_name} - Barra风格因子相关性")
                )
            )

        # 8. 行业因子暴露比例热力图
        if barra_industry_exposure is not None:
            quarter_diff = barra_industry_exposure['quarter_diff']
            industry_factors = barra_industry_exposure['industry_factors']
            
            if len(quarter_diff) > 0:
                # 准备热力图数据
                heatmap_data = []
                
                # 正确处理行业因子热力图数据
                try:
                    # 发现数据结构：时间(行) × 行业因子(列)
                    print(f"      热力图数据形状: {quarter_diff.shape}")
                    print(f"      时间（行索引）: {quarter_diff.index.tolist()[:3]}")
                    print(f"      行业因子（列索引）: {quarter_diff.columns.tolist()[:3]}")
                    
                    # 正确的迭代：时间作为X轴，行业因子作为Y轴
                    for i, time_period in enumerate(quarter_diff.index):  # 时间作为X轴
                        for j, industry_factor in enumerate(quarter_diff.columns):  # 行业因子作为Y轴
                            try:
                                value = quarter_diff.loc[time_period, industry_factor]
                                if not pd.isna(value):
                                    heatmap_data.append([i, j, round(float(value), 4)])
                            except Exception as e:
                                print(f"        跳过数据点 [{time_period}, {industry_factor}]: {e}")
                                continue
                    
                    if heatmap_data:
                        # 格式化时间标签 - 按季度显示
                        time_labels = []
                        for idx in quarter_diff.index:
                            if hasattr(idx, 'strftime'):
                                # 显示季度格式：2023Q1, 2023Q2等
                                year = idx.year
                                quarter = (idx.month - 1) // 3 + 1
                                time_labels.append(f"{year}Q{quarter}")
                            else:
                                time_labels.append(str(idx))
                        
                        # 优化热力图精度和颜色映射
                        # 计算更精确的数值范围，避免0值过多
                        abs_max = quarter_diff.abs().max().max()
                        if abs_max > 0:
                            # 使用95分位数来设置范围，减少极值影响
                            quantile_95 = quarter_diff.abs().quantile(0.95).max()
                            color_range = min(abs_max, quantile_95 * 1.2)
                        else:
                            color_range = 0.01  # 避免全为0的情况
                        
                        # 优化颜色映射，增加中间色调的区分度
                        industry_heatmap = (
                            HeatMap()
                            .add_xaxis(time_labels)
                            .add_yaxis("行业因子", list(quarter_diff.columns), heatmap_data,
                                     label_opts=opts.LabelOpts(is_show=False))
                            .set_global_opts(
                                title_opts=opts.TitleOpts(title=f"{self.factor_name} - 行业因子多空暴露差异（季度统计）"),
                                xaxis_opts=opts.AxisOpts(
                                    name="时间（季度）",
                                    type_="category",
                                    axislabel_opts=opts.LabelOpts(rotate=45)
                                ),
                                yaxis_opts=opts.AxisOpts(
                                    name="行业因子",
                                    type_="category"
                                ),
                                visualmap_opts=opts.VisualMapOpts(
                                    min_=-color_range,
                                    max_=color_range,
                                    pos_left="right",
                                    is_calculable=True,
                                    range_color=[
                                        "#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", 
                                        "#ffffcc", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"
                                    ],
                                    # 增加精度显示
                                    precision=4,
                                    # 优化分段
                                    split_number=10
                                )
                            )
                        )
                        print(f"      热力图生成成功，数据点数量: {len(heatmap_data)}")
                    else:
                        from pyecharts.components import Table
                        industry_heatmap = (
                            Table()
                            .add(["行业因子多空暴露差异"], [["热力图数据为空"]])
                            .set_global_opts(
                                title_opts=opts.TitleOpts(title=f"{self.factor_name} - 行业因子多空暴露差异")
                            )
                        )
                except Exception as e:
                    print(f"      警告：热力图生成失败，使用表格替代: {e}")
                    from pyecharts.components import Table
                    industry_heatmap = (
                        Table()
                        .add(["行业因子多空暴露差异"], [["热力图生成失败"]])
                        .set_global_opts(
                            title_opts=opts.TitleOpts(title=f"{self.factor_name} - 行业因子多空暴露差异")
                        )
                    )
            else:
                from pyecharts.components import Table
                industry_heatmap = (
                    Table()
                    .add(["行业因子多空暴露差异"], [["无季度统计数据"]])
                    .set_global_opts(
                        title_opts=opts.TitleOpts(title=f"{self.factor_name} - 行业因子多空暴露差异")
                    )
                )
        else:
            from pyecharts.components import Table
            industry_heatmap = (
                Table()
                .add(["行业因子多空暴露差异"], [["无Barra数据"]])
                .set_global_opts(
                    title_opts=opts.TitleOpts(title=f"{self.factor_name} - 行业因子多空暴露差异")
                )
            )

        # 9. 年度统计表
        from pyecharts.components import Table
        annual_table_data = []
        for i, row in annual_df.iterrows():
            annual_table_data.append([
                row['年度'],
                f"{row['IC均值']:.4f}",
                f"{row['多头收益']:.2f}%",
                f"{row['空头收益']:.2f}%",
                f"{row['多空夏普']:.2f}" if not pd.isna(row['多空夏普']) else "",
                f"{row['多空最大回撤']:.2f}%" if not pd.isna(row['多空最大回撤']) else ""
            ])
        
        annual_table = (
            Table()
            .add(["年度", "IC均值", "多头收益", "空头收益", "多空夏普", "多空最大回撤"], annual_table_data)
            .set_global_opts(
                title_opts=opts.TitleOpts(title="年度统计表", pos_left="center")
            )
        )

        # 生成HTML文件
        if save_path is None:
            save_path = f"{self.factor_name}_分析报告.html"
        
        # 使用Page组件组合所有图表 - 重新组织布局
        from pyecharts.charts import Page
        
        page = Page(layout=Page.SimplePageLayout)
        
        # 第一行：2*2图表布局
        page.add(ic_chart, group_chart, ls_chart, group_bar)
        
        # 第二行：行业暴露热力图和风格因子相关系数时序图
        page.add(barra_style_chart, industry_heatmap)
        
        # 第三行：所有数据表
        page.add(ic_table, stats_table, annual_table)
        
        # 保存为HTML文件
        page.render(save_path)
        
        print(f"HTML报告已保存到: {save_path}")
        
        # 默认显示HTML文件
        import webbrowser
        webbrowser.open(save_path)
        
        return page

    def generate_report(self, n_groups=10, method='spearman', save_path=None, show_log_returns=True):
        """生成分析报告"""
        print("=" * 50)
        print(f"因子分析报告: {self.factor_name}")
        print(f"调仓周期: {self.rebalance_period}")
        print("=" * 50)
        
        ic_stats = self.calculate_icir(method)
        print(f"\nIC分析结果 ({method}相关系数):")
        
        # 显示IC1到IC5的统计信息
        for lag in range(1, 6):
            ic_key = f'IC{lag}'
            if ic_key in ic_stats:
                ic_data = ic_stats[ic_key]
                print(f"\n{ic_key}:")
                print(f"  IC均值: {ic_data['IC_mean']:.4f}")
                print(f"  IC标准差: {ic_data['IC_std']:.4f}")
                print(f"  ICIR: {ic_data['ICIR']:.4f}")
                print(f"  IC正比例: {ic_data['IC_positive_ratio']:.2%}")
                print(f"  IC偏度: {ic_data['IC_skew']:.4f}")
                print(f"  IC峰度: {ic_data['IC_kurtosis']:.4f}")
                print(f"  t值: {ic_data['IC_tvalue']:.4f}")
                print(f"  p值: {ic_data['IC_pvalue']:.4g}")

        returns_data = self.calculate_returns(n_groups)
        long_short_data = returns_data['long_short_returns']
        stats = long_short_data['stats']
        
        print(f"\n多空组合分析结果 ({n_groups}分组):")
        print(f"平均收益: {stats['mean_return']:.4f}")
        print(f"收益率标准差: {stats['std_return']:.4f}")
        print(f"夏普比率: {stats['sharpe_ratio']:.4f}")
        print(f"胜率: {stats['win_rate']:.2%}")
        print(f"最大回撤: {stats['max_drawdown']:.2%}")
        print(f"总收益: {stats['total_return']:.2%}")
        
        # 准备预计算数据
        precomputed_data = {
            'ic_stats': ic_stats,
            'ic_series': ic_stats['IC_series'],
            'group_returns': returns_data['group_returns'],
            'cumulative_returns': returns_data['cumulative_returns'],
            'long_short_returns': long_short_data,
            'long_short_stats': stats,
            'barra_style_corr': None,
            'barra_industry_exposure': None
        }

        self.plot_full_analysis(method=method, n_groups=n_groups, precomputed_data=precomputed_data, save_path=save_path, show_log_returns=show_log_returns)

        return {
            'ic_stats': ic_stats,
            'long_short_stats': stats,
            'group_returns': returns_data['group_returns'],
            'cumulative_returns': returns_data['cumulative_returns'],
            'long_short_returns': long_short_data
        }


def analyze_single_factor(factor_data, returns_data, factor_name='factor', n_groups=10, method='spearman', rebalance_period=1, enable_stock_filter=True, save_path=None, show_log_returns=True, barra_data=None):
    """单因子分析主函数"""
    analyzer = SingleFactorAnalyzer(factor_data, returns_data, factor_name, rebalance_period, enable_stock_filter, barra_data)
    return analyzer.generate_report(n_groups, method, save_path, show_log_returns)