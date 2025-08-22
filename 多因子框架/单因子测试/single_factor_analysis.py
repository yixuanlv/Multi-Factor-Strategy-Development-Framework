import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from tqdm import tqdm
from pyecharts import options as opts
from pyecharts.charts import Line, Bar, HeatMap
from pyecharts.globals import ThemeType

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class SingleFactorAnalyzer:
    """
    单因子分析工具类（重构版）
    输入：长格式数据框，包含 date, order_book_id, factor_value, close 列
    - returns_data: 至少包含 date, order_book_id, close；如有 limit_up_flag/limit_down_flag/ST/suspended 会用于过滤
    - barra_data: （可选）包含 date, order_book_id + 11个风格因子 + 行业因子（列名放在一起也可）
    """

    # ============== 初始化与基础对齐 ==============
    def __init__(self, factor_data, returns_data, factor_name='factor',
                 rebalance_period=1, enable_stock_filter=True, barra_data=None):
        self.factor_data = factor_data
        self.returns_data = returns_data
        self.factor_name = factor_name
        self.rebalance_period = rebalance_period
        self.enable_stock_filter = bool(enable_stock_filter)
        # 避免复制barra_data，直接使用引用，在需要时再处理
        self.barra_data = barra_data  # 移除.copy()操作
        # 缓存IC统计结果，避免重复计算
        self._ic_stats_cache = {}
        print("  [初始化层] 读取因子数据与行情数据 ...")
        self._align_data()
        self._create_rebalance_dates()

    def _align_data(self):
        """对齐因子数据与行情数据并计算日收益"""
        self.factor_data['date'] = pd.to_datetime(self.factor_data['date'])
        self.returns_data['date'] = pd.to_datetime(self.returns_data['date'])

        ret = self.returns_data.sort_values(['order_book_id', 'date']).copy()
        ret['return'] = ret.groupby('order_book_id')['close'].pct_change()

        extra_cols = [c for c in ['limit_up_flag', 'limit_down_flag', 'ST', 'suspended'] if c in ret.columns]
        merged = pd.merge(
            self.factor_data[['date', 'order_book_id', 'factor_value']],
            ret[['date', 'order_book_id', 'return', 'close'] + extra_cols],
            on=['date', 'order_book_id'],
            how='inner'
        )
        # 不在这里dropna，保留收益NaN（例如停牌日），在统计时自然跳过
        self.merged_data = merged.sort_values(['date', 'order_book_id'])

        if self.barra_data is not None:
            self.barra_data['date'] = pd.to_datetime(self.barra_data['date'])
        print(f"  [初始化层] 合并因子数据与行情数据公共部分，共 {len(self.merged_data['date'].unique()):,} 日 ...")

    def _create_rebalance_dates(self):
        """创建调仓日序列"""
        all_dates = np.sort(self.merged_data['date'].unique())
        if len(all_dates) == 0:
            self.rebalance_dates = []
            return
        if self.rebalance_period > 1:
            self.rebalance_dates = list(all_dates[::self.rebalance_period])
        else:
            self.rebalance_dates = list(all_dates)

    # ============== 交易层：持仓与收益向量化计算 ==============
    def _filter_stocks_for_buy(self, df_on_day):
        """买入过滤（非涨停、非ST、非停牌），向量化布尔筛选"""
        if not self.enable_stock_filter:
            return df_on_day
        mask = pd.Series(True, index=df_on_day.index)
        if 'limit_up_flag' in df_on_day.columns:
            mask &= ~df_on_day['limit_up_flag'].astype(bool)
        if 'ST' in df_on_day.columns:
            mask &= ~df_on_day['ST'].astype(bool)
        if 'suspended' in df_on_day.columns:
            mask &= ~df_on_day['suspended'].astype(bool)
        return df_on_day[mask]

    def _make_groups_on_rebalance_day(self, df_day, n_groups):
        """
        单个调仓日：因子从小到大排序分组，返回列 ['rb_date','order_book_id','group']。
        使用rank -> 等容量划分，避免qcut重复边界问题。
        """
        df = df_day.dropna(subset=['factor_value']).copy()
        df = self._filter_stocks_for_buy(df)
        if len(df) < n_groups:
            return pd.DataFrame(columns=['rb_date', 'order_book_id', 'group'])

        ranks = df['factor_value'].rank(method='first', ascending=True)
        group_size = max(1, len(ranks) // n_groups)
        groups = ((ranks - 1) // group_size).clip(upper=n_groups - 1).astype(int)

        out = pd.DataFrame({
            'rb_date': df_day['date'].iloc[0],
            'order_book_id': df.loc[ranks.index, 'order_book_id'].values,
            'group': groups.values
        })
        return out

    def compute_positions_and_returns(self, n_groups=10):
        """
        交易层主函数（交易日循环 + 组间向量化 + 对齐安全）：
        - 初始虚拟总市值=1
        - 调仓日：上一日组总市值等权分配
        - 非调仓日：滚动市值 * (1+当日收益)
        - 输出：每日持仓、分组收益率、累计净值
        """
        import pandas as pd
        import numpy as np

        print(f"  [交易层] 构建调仓分组与每日持仓 (n_groups={n_groups}, 周期={self.rebalance_period}) ...")

        # -------- A. 数据准备 --------
        md = self.merged_data.sort_values(['order_book_id', 'date']).copy()
        md['future_return'] = md.groupby('order_book_id')['return'].shift(-1)

        rb_dates = pd.to_datetime(self.rebalance_dates)
        rb_set = set(rb_dates)

        # 调仓日分组
        df_rb = md[md['date'].isin(rb_set)].copy()
        group_labels_list = []
        for d, g in df_rb.groupby('date'):
            res = self._make_groups_on_rebalance_day(g, n_groups)
            if len(res) > 0:
                group_labels_list.append(res)

        if not group_labels_list:
            empty = pd.DataFrame(columns=['date', 'rb_date', 'order_book_id', 'group', 'weight', 'market_value'])
            idx = pd.DatetimeIndex([])
            grp = pd.DataFrame(index=idx, columns=range(n_groups))
            return {
                'positions_daily': empty,
                'group_daily_returns': grp,
                'group_cum_nav': grp,
                'rebalance_info': {'rebalance_dates': list(rb_dates), 'group_labels_rebalance': empty}
            }

        group_labels_rebalance = pd.concat(group_labels_list, ignore_index=True)
        group_labels_rebalance['rb_date'] = pd.to_datetime(group_labels_rebalance['rb_date'])

        # -------- B. 构建每日持仓框架 --------
        all_dates = np.sort(md['date'].unique())
        dates_df = pd.DataFrame({'date': pd.to_datetime(all_dates)}).sort_values('date')
        rb_df = pd.DataFrame({'rb_date': pd.to_datetime(rb_dates)}).sort_values('rb_date')
        map_df = pd.merge_asof(dates_df, rb_df, left_on='date', right_on='rb_date', direction='backward')
        map_df = map_df[map_df['rb_date'].notna()]
        positions_daily = map_df.merge(group_labels_rebalance, how='left', on='rb_date')[['date','rb_date','order_book_id','group']]
        positions_daily = positions_daily.merge(
            md[['date','order_book_id','future_return']],
            on=['date','order_book_id'], how='left'
        ).rename(columns={'future_return':'return'})

        positions_daily['market_value'] = 0.0
        positions_daily['weight'] = 0.0

        # -------- C. 初始化组总市值 --------
        group_cum_nav = pd.DataFrame(index=all_dates, columns=range(n_groups), dtype=float)
        group_cum_nav.iloc[0,:] = 1.0  # 初始组总市值=1

        # -------- D. 交易日循环 --------
        from tqdm import tqdm
        for i, date in enumerate(tqdm(all_dates, desc="交易日循环")):
            today_mask = positions_daily['date'] == date
            rb_today = date in rb_set

            today_df = positions_daily.loc[today_mask].copy()
            group_counts = today_df.groupby('group')['order_book_id'].transform('count')

            if rb_today:
                # 调仓日：上一日组总市值 / 成分股数量
                if i == 0:
                    prev_group_values = np.ones(n_groups)
                else:
                    prev_group_values = group_cum_nav.iloc[i-1].values
                prev_values_map = pd.Series(prev_group_values, index=range(n_groups))
                init_values = prev_values_map.loc[today_df['group']].values / group_counts.values
                positions_daily.loc[today_mask, 'market_value'] = init_values
            else:
                # 非调仓日：昨日市值 * (1+当日收益)
                prev_date = all_dates[i-1]

                # 上一日市值
                prev_df = positions_daily.loc[positions_daily['date']==prev_date, ['order_book_id','group','market_value']]
                prev_df = prev_df.set_index(['order_book_id','group'])

                # 当日收益
                returns_today = positions_daily.loc[today_mask, ['order_book_id','group','return']]
                returns_today = returns_today.set_index(['order_book_id','group'])

                # 对齐索引后相乘，保证长度一致
                mv_today = prev_df['market_value'].reindex(returns_today.index) * (1 + returns_today['return'])
                positions_daily.loc[today_mask, 'market_value'] = mv_today.values

            # 更新组总市值（向量化）
            group_cum_nav.loc[date] = positions_daily.loc[today_mask].groupby('group')['market_value'].sum()

            # 更新权重
            total_map = positions_daily.loc[today_mask].groupby('group')['market_value'].transform('sum')
            positions_daily.loc[today_mask, 'weight'] = positions_daily.loc[today_mask,'market_value'] / total_map

        # -------- E. 组收益率 --------
        group_daily_returns = group_cum_nav.pct_change().fillna(0)

        return {
            'positions_daily': positions_daily,
            'group_daily_returns': group_daily_returns,
            'group_cum_nav': group_cum_nav,
            'rebalance_info': {
                'rebalance_dates': list(rb_dates),
                'group_labels_rebalance': group_labels_rebalance
            }
        }



    # ============== 绩效层：IC/ICIR + 多空统计 ==============
    def compute_ic_and_stats(self, method='spearman', max_lag=5):
        """
        将 IC / ICIR 等指标合并在一个函数内：
        返回字典 { 'IC1': {...}, ..., 'IC{max_lag}': {...}, 'IC_series': Series(IC1) }
        """
        # 检查缓存中是否已有相同参数的结果
        cache_key = (method, max_lag)
        if cache_key in self._ic_stats_cache:
            print(f"  [绩效层] 使用缓存的 IC/ICIR 结果（method={method}, lag=1..{max_lag}） ...")
            return self._ic_stats_cache[cache_key]
            
        print(f"  [绩效层] 计算 IC/ICIR（method={method}, lag=1..{max_lag}） ...")
        md = self.merged_data.copy()

        def _corr(a, b):
            a = a.astype(float); b = b.astype(float)
            if a.notna().sum() < 10 or b.notna().sum() < 10:
                return np.nan
            if method.lower() == 'spearman':
                c, _ = stats.spearmanr(a, b, nan_policy='omit')
                return c
            else:
                return np.corrcoef(a.dropna(), b.loc[a.dropna().index].dropna())[0, 1]

        out = {}
        for lag in range(1, max_lag + 1):
            tmp = md.copy()
            # 计算从当日收盘价到未来第lag*rebalance_period个交易日的收益率
            actual_days = lag + self.rebalance_period - 1
            tmp['future_close'] = tmp.groupby('order_book_id')['close'].shift(-actual_days)
            tmp['future_return'] = (tmp['future_close'] - tmp['close']) / tmp['close']

            # IC计算也需要买入过滤：去除停牌、ST、涨停等不可交易的股票
            if self.enable_stock_filter:
                tmp = self._filter_stocks_for_buy(tmp)
            
            ic_series = (
                tmp.groupby('date')
                   .apply(lambda g: _corr(g['factor_value'], g['future_return']))
                   .astype(float)
            )

            ic_s = ic_series.dropna()
            if len(ic_s) == 0:
                stats_pack = dict(IC_mean=np.nan, IC_std=np.nan, ICIR=np.nan,
                                  IC_positive_ratio=np.nan, IC_skew=np.nan, IC_kurtosis=np.nan,
                                  IC_tvalue=np.nan, IC_pvalue=np.nan, IC_series=ic_series)
            else:
                ic_mean = ic_s.mean()
                ic_std = ic_s.std()
                icir = ic_mean / ic_std if ic_std != 0 else np.nan
                ic_pos = (ic_s > 0).mean()
                ic_skew = stats.skew(ic_s)
                ic_kurt = stats.kurtosis(ic_s, fisher=True)
                t_stat, p_val = stats.ttest_1samp(ic_s, 0, nan_policy='omit')
                stats_pack = dict(IC_mean=ic_mean, IC_std=ic_std, ICIR=icir,
                                  IC_positive_ratio=ic_pos, IC_skew=ic_skew, IC_kurtosis=ic_kurt,
                                  IC_tvalue=t_stat, IC_pvalue=p_val, IC_series=ic_series)
            
            # 使用实际预测天数作为IC的键名，使其更直观
            ic_key = f'IC{actual_days}D' if self.rebalance_period > 1 else f'IC{lag}'
            out[ic_key] = stats_pack

        # 设置默认的IC_series为第一个lag的结果
        first_key = list(out.keys())[0]
        out['IC_series'] = out[first_key]['IC_series']
        print("    IC/ICIR 计算完成。")
        
        # 将结果存入缓存
        self._ic_stats_cache[cache_key] = out
        return out

    def _annual_stats_df(self, ic_series, long_returns, short_returns, long_short_returns):
        """年度统计表"""
        # 对齐索引
        for s in [ic_series, long_returns, short_returns, long_short_returns]:
            if not isinstance(s.index, pd.DatetimeIndex):
                s.index = pd.to_datetime(s.index)

        common = ic_series.index.intersection(long_returns.index)\
                                 .intersection(short_returns.index)\
                                 .intersection(long_short_returns.index)
        ic_series = ic_series.reindex(common)
        long_returns = long_returns.reindex(common)
        short_returns = short_returns.reindex(common)
        ls_ret = long_short_returns.reindex(common)

        years = sorted(ic_series.index.year.unique())
        rows = []
        for y in years:
            mask = ic_series.index.year == y
            ic_mean = ic_series[mask].mean()
            long_y = long_returns[mask]
            short_y = short_returns[mask]
            ls_y = ls_ret[mask]

            long_ann = (1 + long_y).prod() - 1 if long_y.notna().sum() else np.nan
            short_ann = (1 + short_y).prod() - 1 if short_y.notna().sum() else np.nan
            sharpe = np.nan
            if ls_y.notna().sum() > 1 and ls_y.std() not in [0, np.nan]:
                sharpe = (ls_y.mean() / ls_y.std()) * np.sqrt(252)

            cum = (1 + ls_y.fillna(0)).cumprod()
            if len(cum) > 0:
                running_max = cum.expanding().max()
                dd = (cum - running_max) / running_max
                max_dd = dd.min()
            else:
                max_dd = np.nan

            rows.append({
                '年度': str(y),
                'IC均值': ic_mean,
                '多头收益': long_ann * 100 if pd.notna(long_ann) else np.nan,
                '空头收益': short_ann * 100 if pd.notna(short_ann) else np.nan,
                '多空夏普': sharpe if pd.notna(sharpe) else np.nan,
                '多空最大回撤': max_dd * 100 if pd.notna(max_dd) else np.nan
            })
        return pd.DataFrame(rows)

    def analyze_performance(self, positions_returns, method='spearman', n_groups=10):
        """
        基于交易结果汇总绩效：IC/ICIR + long-short + 年度统计
        返回：
        {
            'ic_stats': ...,
            'long_short': {
                'long_short_returns', 'cumulative_ls_returns',
                'long_returns', 'short_returns', 'stats'
            },
            'annual_df': DataFrame
        }
        """
        print("  [绩效层] 汇总绩效指标 ...")
        grp_ret = positions_returns['group_daily_returns']

        # 自动判断多空方向（基于IC均值）
        ic_stats = self.compute_ic_and_stats(method=method, max_lag=5)
        # 获取第一个IC统计结果（可能是IC1或IC{actual_days}D）
        first_ic_key = [k for k in ic_stats.keys() if k != 'IC_series'][0]
        ic_mean = ic_stats[first_ic_key]['IC_mean']
        if pd.isna(ic_mean) or ic_mean >= 0:
            long_group, short_group = n_groups - 1, 0  # 因子正相关：多高空低
        else:
            long_group, short_group = 0, n_groups - 1  # 因子负相关：多低空高

        long_returns = grp_ret.get(long_group, pd.Series(dtype=float))
        short_returns = grp_ret.get(short_group, pd.Series(dtype=float))

        long_short_returns = long_returns - short_returns
        # 多空累计净值
        cum_ls = (1 + long_short_returns.fillna(0)).cumprod()

        stats_pack = {
            'mean_return': long_short_returns.mean(),
            'std_return': long_short_returns.std(),
            'sharpe_ratio': (long_short_returns.mean() / long_short_returns.std()) if long_short_returns.std() not in [0, np.nan] else np.nan,
            'max_drawdown': self._calculate_max_drawdown(cum_ls),
            'win_rate': (long_short_returns > 0).mean(),
            'total_return': (cum_ls.iloc[-1] - 1) if len(cum_ls) else np.nan,
            'annualized_return': self._calculate_annualized_return(long_short_returns)
        }

        annual_df = self._annual_stats_df(ic_stats['IC_series'], long_returns, short_returns, long_short_returns)

        return {
            'ic_stats': ic_stats,
            'long_short': {
                'long_short_returns': long_short_returns,
                'cumulative_ls_returns': cum_ls,
                'long_returns': long_returns,
                'short_returns': short_returns,
                'stats': stats_pack
            },
            'annual_df': annual_df
        }

    def _calculate_max_drawdown(self, cumulative_returns):
        if len(cumulative_returns) == 0:
            return np.nan
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()
    
    def _calculate_annualized_return(self, daily_returns, trading_days_per_year=252):
        """计算年化收益率"""
        if len(daily_returns) == 0:
            return np.nan
        # 年化收益率 = (1 + 总收益率) ^ (年化天数/实际天数) - 1
        total_return = (1 + daily_returns.fillna(0)).prod() - 1
        years = len(daily_returns) / trading_days_per_year
        if years <= 0:
            return np.nan
        annualized_return = (1 + total_return) ** (1 / years) - 1
        return annualized_return

    # ============== Barra 归因层：风格相关 + 行业暴露 ==============
    def barra_analysis(self, positions_returns, n_groups=10):
        """
        将 Barra 相关的计算统一在一个函数中：
        1) 风格因子相关性：factor_value 与 11个风格因子的日度相关 & 20日滚动均值
        2) 行业暴露差异：使用"每日持仓"计算 long vs short 的行业暴露差异，并给出季度平均

        返回：
        {
            'style_correlation': {
                'daily_correlation': DataFrame[T×K],
                'rolling_correlation': DataFrame[T×K],
                'style_factors': list[str]
            },
            'industry_exposure': {
                'long_exposure': DataFrame[T×Industry],
                'short_exposure': DataFrame[T×Industry],
                'exposure_diff': DataFrame[T×Industry],
                'quarter_diff': DataFrame[Q×Industry],
                'industry_factors': list[str]
            }
        } 或 None (若无 barra_data)
        """
        if self.barra_data is None:
            return None

        print("  [Barra] 计算风格相关性与行业暴露差异 ...")

        # ---- 风格因子相关性（使用斯皮尔曼相关系数，彻底向量化） ----
        barra_columns = self.barra_data.columns.tolist()
        style_factors = [c for c in barra_columns if c not in ['date', 'order_book_id']][:11]

        # 构造因子宽表 T×N
        factor_wide = self.factor_data.pivot(index='date', columns='order_book_id', values='factor_value').sort_index()
        # 风格宽表堆叠 T×N×K
        barra_wide_all = []
        for f in style_factors:
            wide = self.barra_data.pivot(index='date', columns='order_book_id', values=f).reindex_like(factor_wide)
            barra_wide_all.append(wide)
        Y = np.stack([df.to_numpy() for df in barra_wide_all], axis=-1)   # (T×N×K)
        X = factor_wide.to_numpy()                                        # (T×N)

        T, N = X.shape
        K = Y.shape[2]
        
        # 使用斯皮尔曼相关系数计算
        corr = np.full((T, K), np.nan)
        for t in range(T):
            for k in range(K):
                x_t = X[t, :]
                y_t = Y[t, :, k]
                # 找到非NaN的索引
                valid_mask = ~(np.isnan(x_t) | np.isnan(y_t))
                if np.sum(valid_mask) >= 10:  # 至少需要10个有效值
                    x_valid = x_t[valid_mask]
                    y_valid = y_t[valid_mask]
                    try:
                        # 计算斯皮尔曼相关系数
                        corr_coef, _ = stats.spearmanr(x_valid, y_valid, nan_policy='omit')
                        corr[t, k] = corr_coef
                    except:
                        corr[t, k] = np.nan

        daily_corr = pd.DataFrame(corr, index=factor_wide.index, columns=style_factors)
        rolling_corr = daily_corr.rolling(window=20, min_periods=5).mean()

        style_pack = {
            'daily_correlation': daily_corr,
            'rolling_correlation': rolling_corr,
            'style_factors': style_factors
        }

        # ---- 行业暴露差异：使用每日持仓，优化内存使用 ----
        industry_factors = [c for c in barra_columns if c not in ['date', 'order_book_id']][-31:]
        pos_daily = positions_returns['positions_daily']  # date, rb_date, order_book_id, group

        if len(industry_factors) == 0 or len(pos_daily) == 0:
            industry_pack = {
                'long_exposure': pd.DataFrame(),
                'short_exposure': pd.DataFrame(),
                'exposure_diff': pd.DataFrame(),
                'quarter_diff': pd.DataFrame(),
                'industry_factors': industry_factors
            }
            return {'style_correlation': style_pack, 'industry_exposure': industry_pack}

        # 内存优化：分批处理行业因子，避免一次性加载所有数据
        print(f"    [Barra] 处理 {len(industry_factors)} 个行业因子，优化内存使用...")
        
        # 获取多空分组信息
        ic_stats = self.compute_ic_and_stats()
        first_ic_key = [k for k in ic_stats.keys() if k != 'IC_series'][0]
        ic_mean = ic_stats[first_ic_key]['IC_mean']
        
        # 获取实际的n_groups数量
        actual_n_groups = pos_daily['group'].dropna().astype(int).max() + 1 if pos_daily['group'].notna().any() else n_groups
        
        if pd.isna(ic_mean) or ic_mean >= 0:
            long_group, short_group = actual_n_groups - 1, 0
        else:
            long_group, short_group = 0, actual_n_groups - 1

        # 分批处理行业因子，避免内存溢出
        batch_size = 5  # 每次处理5个因子
        long_exposure_list = []
        short_exposure_list = []
        
        for i in range(0, len(industry_factors), batch_size):
            batch_factors = industry_factors[i:i+batch_size]
            print(f"      [Barra] 处理行业因子批次 {i//batch_size + 1}/{(len(industry_factors)-1)//batch_size + 1}: {batch_factors}")
            
            # 只选择当前批次的因子列
            barra_batch = self.barra_data[['date', 'order_book_id'] + batch_factors].copy()
            
            # 使用更高效的merge策略：先过滤pos_daily中实际存在的股票
            unique_stocks = pos_daily['order_book_id'].unique()
            barra_filtered = barra_batch[barra_batch['order_book_id'].isin(unique_stocks)]
            
            # 分批merge，避免内存溢出
            merged_batch = pos_daily.merge(barra_filtered, on=['date', 'order_book_id'], how='left')
            
            # 计算每日分组平均暴露
            exp_batch = merged_batch.groupby(['date', 'group'])[batch_factors].mean()
            
            # 提取多空分组暴露
            if long_group in exp_batch.index.get_level_values('group'):
                long_exp_batch = exp_batch.xs(long_group, level='group')[batch_factors]
                long_exposure_list.append(long_exp_batch)
            else:
                # 创建空的DataFrame
                empty_df = pd.DataFrame(index=merged_batch['date'].unique(), columns=batch_factors)
                long_exposure_list.append(empty_df)
                
            if short_group in exp_batch.index.get_level_values('group'):
                short_exp_batch = exp_batch.xs(short_group, level='group')[batch_factors]
                short_exposure_list.append(short_exp_batch)
            else:
                # 创建空的DataFrame
                empty_df = pd.DataFrame(index=merged_batch['date'].unique(), columns=batch_factors)
                short_exposure_list.append(empty_df)
            
            # 清理内存
            del barra_batch, barra_filtered, merged_batch, exp_batch
        
        # 合并所有批次的结果
        if long_exposure_list:
            long_exposure = pd.concat(long_exposure_list, axis=1)
            # 处理重复的日期索引
            long_exposure = long_exposure.groupby(level=0).first()
        else:
            long_exposure = pd.DataFrame(columns=industry_factors)
            
        if short_exposure_list:
            short_exposure = pd.concat(short_exposure_list, axis=1)
            # 处理重复的日期索引
            short_exposure = short_exposure.groupby(level=0).first()
        else:
            short_exposure = pd.DataFrame(columns=industry_factors)
        
        # 计算暴露差异
        exposure_diff = long_exposure - short_exposure

        # 季度统计
        for df in [long_exposure, short_exposure, exposure_diff]:
            if not isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
                df.index = pd.to_datetime(df.index)

        quarter_diff = exposure_diff.resample('Q').mean() if len(exposure_diff) else pd.DataFrame(columns=industry_factors)
        threshold = 1e-5
        quarter_diff = quarter_diff.where(quarter_diff.abs() >= threshold, 0)

        industry_pack = {
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'exposure_diff': exposure_diff,
            'quarter_diff': quarter_diff,
            'industry_factors': industry_factors
        }

        return {
            'style_correlation': style_pack,
            'industry_exposure': industry_pack
        }

    # ============== 可视化：沿用原结构，内部调用新流程输出 ==============
    def plot_full_analysis(self, method='spearman', n_groups=10, figsize=(16, 28),
                           show_plot=True, precomputed_data=None, save_path=None, show_log_returns=True):
        """
        使用 pyecharts 生成 HTML 报告（布局与原版一致，但对接新的三段式流程）
        """
        if precomputed_data is None:
            # 交易层
            pos_ret = self.compute_positions_and_returns(n_groups=n_groups)
            group_daily_returns = pos_ret['group_daily_returns']
            group_cum_nav = pos_ret['group_cum_nav']

            # 绩效层
            perf = self.analyze_performance(pos_ret, method=method, n_groups=n_groups)
            ic_stats = perf['ic_stats']
            ic_series = ic_stats['IC_series']
            long_short_data = perf['long_short']
            long_short_returns = long_short_data['long_short_returns']
            cumulative_ls_returns = long_short_data['cumulative_ls_returns']
            long_returns = long_short_data['long_returns']
            short_returns = long_short_data['short_returns']
            ls_stats = long_short_data['stats']
            annual_df = perf['annual_df']

            # 组累计收益（最终点）
            if len(group_cum_nav) > 0:
                group_cumulative_last = group_cum_nav.iloc[-1]
            else:
                group_cumulative_last = pd.Series([np.nan] * n_groups, index=range(n_groups))

            # Barra 归因
            barra_all = self.barra_analysis(pos_ret, n_groups=n_groups)
        else:
            # 兼容旧接口
            ic_stats = precomputed_data['ic_stats']
            ic_series = precomputed_data['ic_series']
            group_daily_returns = precomputed_data['group_returns']['group_returns']
            group_cum_nav = precomputed_data['cumulative_returns']
            long_returns = precomputed_data['long_short_returns']['long_returns']
            short_returns = precomputed_data['long_short_returns']['short_returns']
            long_short_returns = precomputed_data['long_short_returns']['long_short_returns']
            cumulative_ls_returns = precomputed_data['long_short_returns']['cumulative_ls_returns']
            ls_stats = precomputed_data['long_short_stats']
            
            # 构造 long_short_data 以保持一致性
            long_short_data = {
                'long_short_returns': long_short_returns,
                'cumulative_ls_returns': cumulative_ls_returns,
                'long_returns': long_returns,
                'short_returns': short_returns,
                'stats': ls_stats
            }
            
            if 'annual_df' in precomputed_data:
                annual_df = precomputed_data['annual_df']
            else:
                annual_df = self._annual_stats_df(ic_series, long_returns, short_returns, long_short_returns)

            if len(group_cum_nav) > 0:
                group_cumulative_last = group_cum_nav.iloc[-1]
            else:
                group_cumulative_last = pd.Series([np.nan] * group_daily_returns.shape[1])

            barra_all = precomputed_data.get('barra_all')
            if barra_all is None:
                # 尝试现算
                fake_pos_ret = {
                    'positions_daily': pd.DataFrame(),
                    'group_daily_returns': group_daily_returns,
                    'group_cum_nav': group_cum_nav,
                    'rebalance_info': {'rebalance_dates': self.rebalance_dates, 'group_labels_rebalance': pd.DataFrame()}
                }
                barra_all = self.barra_analysis(fake_pos_ret, n_groups=group_daily_returns.shape[1])

        # ---- 构建图表 ----
        from pyecharts.charts import Page
        page = Page(layout=Page.SimplePageLayout)

        # 1) IC月度均值 & 累计值
        ic_df = ic_series.to_frame('ic'); ic_df.index = pd.to_datetime(ic_df.index)
        ic_cum = ic_df['ic'].cumsum()
        ic_monthly_mean = ic_df.resample('M').mean()
        ic_monthly_mean.index = ic_monthly_mean.index.to_period('M').to_timestamp()

        ic_chart = (
            Bar()
            .add_xaxis([d.strftime('%Y-%m') for d in ic_monthly_mean.index])
            .add_yaxis("IC月度均值", ic_monthly_mean['ic'].round(4).tolist(), yaxis_index=0,
                       label_opts=opts.LabelOpts(is_show=False))
            .extend_axis(yaxis=opts.AxisOpts(name="IC累计值", type_="value", position="right"))
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"{self.factor_name} - IC月度均值 & IC累计值 (调仓周期: {self.rebalance_period})"),
                xaxis_opts=opts.AxisOpts(type_="category", axislabel_opts=opts.LabelOpts(rotate=45)),
                yaxis_opts=opts.AxisOpts(name="IC月度均值", is_scale=True,
                                         axislabel_opts=opts.LabelOpts(is_show=False),
                                         axisline_opts=opts.AxisLineOpts(is_show=False),
                                         axistick_opts=opts.AxisTickOpts(is_show=False)),
                datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
                legend_opts=opts.LegendOpts(pos_top="5%"),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
            )
        )
        ic_line = (
            Line()
            .add_xaxis([d.strftime('%Y-%m') for d in ic_cum.index])
            .add_yaxis("IC累计值", ic_cum.round(4).tolist(), yaxis_index=1,
                       label_opts=opts.LabelOpts(is_show=False), symbol_size=0)
        )
        ic_chart.overlap(ic_line)
        page.add(ic_chart)

        # 2) 各分组累计收益率
        group_chart = Line()
        if len(group_cum_nav) > 0:
            for g in group_cum_nav.columns:
                vals = np.log10(group_cum_nav[g].replace(0, np.nan)) if show_log_returns else group_cum_nav[g]
                group_chart.add_xaxis([d.strftime('%Y-%m-%d') for d in group_cum_nav.index])
                group_chart.add_yaxis(f"分组{int(g)+1}", vals.round(4).tolist(),
                                      label_opts=opts.LabelOpts(is_show=False), symbol_size=0)
        group_chart.set_global_opts(
            title_opts=opts.TitleOpts(title=f"{self.factor_name} - 各分组累计{'对数' if show_log_returns else ''}收益率"),
            xaxis_opts=opts.AxisOpts(type_="category", axislabel_opts=opts.LabelOpts(rotate=45)),
            yaxis_opts=opts.AxisOpts(is_scale=True,
                                     axislabel_opts=opts.LabelOpts(is_show=False),
                                     axisline_opts=opts.AxisLineOpts(is_show=False),
                                     axistick_opts=opts.AxisTickOpts(is_show=False)),
            datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
            legend_opts=opts.LegendOpts(pos_top="5%"),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
        )
        page.add(group_chart)

        # 3) 多空累计收益 + 动态回撤阴影
        cumulative_ls_returns = long_short_data['cumulative_ls_returns']
        running_max = cumulative_ls_returns.expanding().max()
        drawdown = (cumulative_ls_returns - running_max) / running_max if len(cumulative_ls_returns) else pd.Series(dtype=float)

        ls_chart = (
            Line()
            .add_xaxis([d.strftime('%Y-%m-%d') for d in cumulative_ls_returns.index])
            .add_yaxis("多空累计收益", cumulative_ls_returns.round(4).tolist(), yaxis_index=0,
                       label_opts=opts.LabelOpts(is_show=False), symbol_size=0)
            .extend_axis(yaxis=opts.AxisOpts(name="动态回撤", type_="value", position="right", is_inverse=True))
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"{self.factor_name} - 多空累计收益及动态回撤 (调仓周期: {self.rebalance_period})"),
                xaxis_opts=opts.AxisOpts(type_="category", axislabel_opts=opts.LabelOpts(rotate=45)),
                yaxis_opts=opts.AxisOpts(name="累计收益", is_scale=True,
                                         axislabel_opts=opts.LabelOpts(is_show=False),
                                         axisline_opts=opts.AxisLineOpts(is_show=False),
                                         axistick_opts=opts.AxisTickOpts(is_show=False)),
                datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
                legend_opts=opts.LegendOpts(pos_top="5%"),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
            )
        )
        if len(drawdown) > 0:
            dd_area = (
                Line()
                .add_xaxis([d.strftime('%Y-%m-%d') for d in drawdown.index])
                .add_yaxis("动态回撤", drawdown.round(4).tolist(), yaxis_index=1,
                           is_symbol_show=False,
                           linestyle_opts=opts.LineStyleOpts(width=0),
                           label_opts=opts.LabelOpts(is_show=False))
                .set_series_opts(areastyle_opts=opts.AreaStyleOpts(opacity=0.3))
            )
            ls_chart.overlap(dd_area)
        page.add(ls_chart)

        # 4) 分组累计收益率柱状图（期末）
        group_bar = (
            Bar()
            .add_xaxis([f"分组{i+1}" for i in range(len(group_cumulative_last))])
            .add_yaxis("累计收益率", group_cumulative_last.round(4).tolist(),
                       label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"{self.factor_name} - 各分组累计收益率"),
                xaxis_opts=opts.AxisOpts(type_="category"),
                yaxis_opts=opts.AxisOpts(is_scale=True,
                                         axislabel_opts=opts.LabelOpts(is_show=False),
                                         axisline_opts=opts.AxisLineOpts(is_show=False),
                                         axistick_opts=opts.AxisTickOpts(is_show=False)),
                datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
                legend_opts=opts.LegendOpts(pos_top="5%"),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
            )
        )
        page.add(group_bar)



        # 7) Barra 风格相关时序 & 8) 行业暴露热力图（第三行）
        if barra_all is not None:
            # 风格
            style_corr = barra_all['style_correlation']
            daily_corr = style_corr['daily_correlation']
            style_factors = style_corr['style_factors']
            barra_style_chart = Line()
            for f in style_factors:
                if f in daily_corr.columns:
                    barra_style_chart.add_xaxis([d.strftime('%Y-%m-%d') for d in daily_corr.index])
                    barra_style_chart.add_yaxis(f, daily_corr[f].round(4).tolist(),
                                                label_opts=opts.LabelOpts(is_show=False), symbol_size=0)
            barra_style_chart.set_global_opts(
                title_opts=opts.TitleOpts(title=f"{self.factor_name} - 与Barra风格因子相关系数（日度）"),
                xaxis_opts=opts.AxisOpts(type_="category", axislabel_opts=opts.LabelOpts(rotate=45)),
                yaxis_opts=opts.AxisOpts(name="相关系数", is_scale=True, min_=-1, max_=1,
                                         axislabel_opts=opts.LabelOpts(is_show=False),
                                         axisline_opts=opts.AxisLineOpts(is_show=False),
                                         axistick_opts=opts.AxisTickOpts(is_show=False)),
                datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
                legend_opts=opts.LegendOpts(pos_top="5%"),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
            )
            page.add(barra_style_chart)

        # 行业热力图
        industry_heatmap = None
        if barra_all is not None:
            industry_pack = barra_all['industry_exposure']
            quarter_diff = industry_pack['quarter_diff']
            if len(quarter_diff) > 0:
                heatmap_data = []
                time_labels = []
                for idx in quarter_diff.index:
                    year = idx.year
                    quarter = (idx.month - 1) // 3 + 1
                    time_labels.append(f"{year}Q{quarter}")
                
                # 优化行业标签显示：截断过长的标签，添加省略号
                industry_labels = []
                for ind in quarter_diff.columns:
                    if len(ind) > 15:
                        industry_labels.append(ind[:12] + "...")
                    else:
                        industry_labels.append(ind)
                
                for i, _t in enumerate(quarter_diff.index):
                    for j, ind in enumerate(quarter_diff.columns):
                        v = quarter_diff.loc[_t, ind]
                        if pd.notna(v):
                            heatmap_data.append([i, j, round(float(v), 4)])

                # 颜色范围
                abs_max = quarter_diff.abs().max().max()
                if abs_max > 0:
                    q95 = quarter_diff.abs().quantile(0.95).max()
                    color_range = float(min(abs_max, q95 * 1.2))
                else:
                    color_range = 0.01

                # 根据行业数量调整图表高度
                n_industries = len(industry_labels)
                chart_height = max(400, min(800, n_industries * 20 + 200))  # 动态高度

                industry_heatmap = (
                    HeatMap()
                    .add_xaxis(time_labels)
                    .add_yaxis("行业因子", industry_labels, heatmap_data,
                            label_opts=opts.LabelOpts(is_show=True, font_size=10, position="right"))
                    .set_global_opts(
                        title_opts=opts.TitleOpts(title=f"{self.factor_name} - 行业因子多空暴露差异（季度统计）"),
                        xaxis_opts=opts.AxisOpts(type_="category", axislabel_opts=opts.LabelOpts(rotate=45)),
                        yaxis_opts=opts.AxisOpts(type_="category",
                                                axislabel_opts=opts.LabelOpts(font_size=10, rotate=0)),
                        datazoom_opts=[
                            # X轴滑块（横向）
                            opts.DataZoomOpts(type_="slider", orient="horizontal", xaxis_index=0,
                                            range_start=0, range_end=100, pos_bottom="5%"),
                            # Y轴滑块（纵向）
                            opts.DataZoomOpts(type_="slider", orient="vertical", yaxis_index=0,
                                            range_start=0, range_end=100, pos_left="left"),
                            # 内部缩放：支持鼠标操作
                            opts.DataZoomOpts(type_="inside", xaxis_index=0),
                            opts.DataZoomOpts(type_="inside", yaxis_index=0)
                        ],
                        visualmap_opts=opts.VisualMapOpts(
                            min_=-color_range, max_=color_range, pos_left="right",
                            is_calculable=True, split_number=10,
                            range_color=["#0000FF", "#FFFFFF", "#FF0000"]
                        )
                    )
                    .set_series_opts(
                        label_opts=opts.LabelOpts(is_show=False, font_size=8, position="inside")
                    )
                )
        if industry_heatmap is not None:
            page.add(industry_heatmap)


        # 9) 数据表部分（第四行）
        from pyecharts.components import Table
        
        # IC统计信息表格（动态获取IC键名）
        ic_keys = [k for k in ic_stats.keys() if k != 'IC_series']
        headers = ["指标"] + ic_keys
        ic_table_data = []
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
        for metric_name, metric_key, fmt in metrics:
            row = [metric_name]
            for ic_key in ic_keys:
                val = ic_stats.get(ic_key, {}).get(metric_key, np.nan)
                if pd.isna(val):
                    row.append("NaN")
                else:
                    if fmt == ":.2%":
                        row.append(f"{val:.2%}")
                    elif fmt == ":.4g":
                        row.append(f"{val:.4g}")
                    else:
                        row.append(f"{val:.4f}")
            ic_table_data.append(row)
        ic_table = Table().add(headers, ic_table_data).set_global_opts(
            title_opts=opts.TitleOpts(title=f"{self.factor_name} - IC统计信息", pos_left="center")
        )
        page.add(ic_table)

        # 多空组合统计信息表格
        stats_table = Table().add(
            ["指标", "数值"],
            [
                ["平均收益", f"{ls_stats['mean_return']:.4f}"],
                ["收益率标准差", f"{ls_stats['std_return']:.4f}"],
                ["夏普比率", f"{ls_stats['sharpe_ratio']:.4f}"],
                ["胜率", f"{ls_stats['win_rate']:.2%}"],
                ["最大回撤", f"{ls_stats['max_drawdown']:.2%}"],
                ["总收益", f"{ls_stats['total_return']:.2%}"]
            ]
        ).set_global_opts(title_opts=opts.TitleOpts(title=f"{self.factor_name} - 多空组合统计信息", pos_left="center"))
        page.add(stats_table)

        # 年度统计表
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
        page.add(annual_table)

        # 生成 HTML
        if save_path is None:
            save_path = f"{self.factor_name}_分析报告.html"
        page.render(save_path)
        print(f"HTML报告已保存到: {save_path}")

        import webbrowser
        webbrowser.open(save_path)
        return page

    def generate_report(self, n_groups=10, method='spearman', save_path=None, show_log_returns=True):
        """命令行版打印 + 生成HTML"""
        print("=" * 50)
        print(f"因子分析报告: {self.factor_name}")
        print(f"调仓周期: {self.rebalance_period}")
        print("=" * 50)

        # 交易层
        pos_ret = self.compute_positions_and_returns(n_groups=n_groups)
        # 绩效层
        perf = self.analyze_performance(pos_ret, method=method, n_groups=n_groups)
        ic_stats = perf['ic_stats']
        ls_stats = perf['long_short']['stats']

        print(f"\nIC分析结果 ({method}相关系数):")
        
        # 创建IC统计指标DataFrame
        ic_df_data = []
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
        
        # 构建DataFrame数据
        ic_keys = [k for k in ic_stats.keys() if k != 'IC_series']
        for metric_name, metric_key, fmt in metrics:
            row = {'指标': metric_name}
            for ic_key in ic_keys:
                dat = ic_stats.get(ic_key, {})
                val = dat.get(metric_key, np.nan)
                if pd.isna(val):
                    row[ic_key] = "NaN"
                else:
                    if fmt == ":.2%":
                        row[ic_key] = f"{val:.2%}"
                    elif fmt == ":.4g":
                        row[ic_key] = f"{val:.4g}"
                    else:
                        row[ic_key] = f"{val:.4f}"
            ic_df_data.append(row)
        
        # 创建并打印IC统计DataFrame
        ic_df = pd.DataFrame(ic_df_data)
        ic_df = ic_df.set_index('指标')
        print("\n" + "="*80)
        print("IC统计指标汇总表")
        print("="*80)
        print(ic_df.to_string())
        print("="*80)
        


        print(f"\n多空组合分析结果 ({n_groups}分组):")
        print(f"  平均收益: {ls_stats['mean_return']:.4f}")
        print(f"  收益率标准差: {ls_stats['std_return']:.4f}")
        print(f"  夏普比率: {ls_stats['sharpe_ratio']:.4f}")
        print(f"  胜率: {ls_stats['win_rate']:.2%}")
        print(f"  最大回撤: {ls_stats['max_drawdown']:.2%}")
        print(f"  总收益: {ls_stats['total_return']:.2%}")
        print(f"  年化收益: {ls_stats['annualized_return']:.2%}")

        # 生成可视化报告
        precomputed_data = {
            'ic_stats': ic_stats,
            'ic_series': ic_stats['IC_series'],
            'group_returns': {'group_returns': pos_ret['group_daily_returns']},
            'cumulative_returns': pos_ret['group_cum_nav'],
            'long_short_returns': perf['long_short'],
            'long_short_stats': ls_stats,
            'annual_df': perf['annual_df'],
            'barra_all': self.barra_analysis(pos_ret, n_groups=n_groups)
        }
        self.plot_full_analysis(method=method, n_groups=n_groups,
                                precomputed_data=precomputed_data,
                                save_path=save_path, show_log_returns=show_log_returns)

        return {
            'ic_stats': ic_stats,
            'long_short_stats': ls_stats,
            'group_returns': {'group_returns': pos_ret['group_daily_returns']},
            'cumulative_returns': pos_ret['group_cum_nav'],
            'long_short_returns': perf['long_short']
        }


def analyze_single_factor(factor_data, returns_data, factor_name='factor',
                          n_groups=10, method='spearman', rebalance_period=1,
                          enable_stock_filter=True, save_path=None,
                          show_log_returns=True, barra_data=None):

    analyzer = SingleFactorAnalyzer(
        factor_data=factor_data,
        returns_data=returns_data,
        factor_name=factor_name,
        rebalance_period=rebalance_period,
        enable_stock_filter=enable_stock_filter,
        barra_data=barra_data
    )
    return analyzer.generate_report(n_groups=n_groups, method=method,
                                    save_path=save_path, show_log_returns=show_log_returns)
