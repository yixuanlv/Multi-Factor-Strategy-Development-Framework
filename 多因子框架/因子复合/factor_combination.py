# -*- coding: utf-8 -*-
"""
因子复合分析框架（已修改 build 方法，按照用户新要求：
对于 N 阶滚动的因子复合，复合因子按以下规则构造：

C(t,i) = Z(t,i)*W(t-1) + Z(t-1,i)*W(t-1) + Z(t-2,i)*W(t-2) + Z(t-3,i)*W(t-3) + ...

也就是说：
- k=0 (当天因子 Z(t)) 和 k=1 (前一天因子 Z(t-1)) 都使用 W(t-1) 作为权重；
- 对于 k>=2，使用对应的 W(t-k)；

其余逻辑（权重计算、收益对齐、IC 计算等）保持原样或轻微调整以兼容该合成规则。

该文件基于你原始的 factor_combination.py 做了局部重构：主要修改 build 方法实现上述合成规则。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import time

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from pyecharts import options as opts
from pyecharts.charts import Line, Bar, HeatMap, Page
from pyecharts.components import Table

# 导入单因子测试类
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '单因子测试'))
from single_factor_analysis import SingleFactorAnalyzer


# ------------------------- 工具函数 -------------------------

def _zscore_rowwise(df: pd.DataFrame) -> pd.DataFrame:
    """横截面Z-Score标准化"""
    def _z(s: pd.Series) -> pd.Series:
        m, v = s.mean(), s.std(ddof=0)
        return pd.Series(0.0, index=s.index) if not np.isfinite(v) or v == 0 else (s - m) / v
    
    return df.apply(_z, axis=1)


def _l1_normalize(w: pd.Series, eps: float = 1e-12) -> pd.Series:
    """L1归一化"""
    s = np.sum(np.abs(w.values))
    return pd.Series(0.0, index=w.index) if not np.isfinite(s) or s < eps else w / s


# ------------------------- 数据模型 -------------------------

@dataclass
class ComboResult:
    combined_factor: pd.DataFrame           # 合成因子
    weight_history: pd.DataFrame            # 权重历史
    ic_series: pd.Series                    # IC时间序列
    long_only_returns: pd.Series            # 纯多收益
    group_returns: pd.DataFrame             # 分组收益
    summary: dict                           # 统计指标


# ------------------------- 主类 -------------------------

class FactorCombiner:
    def __init__(
        self,
        factors: Dict[str, pd.DataFrame],
        prices: pd.DataFrame,
        rebalance_period: int = 1,
        enable_stock_filter: bool = True,
    ) -> None:
        self.factors_raw = {k: v.copy() for k, v in factors.items()}
        self.prices = prices.copy()
        self.rebalance_period = int(rebalance_period)
        self.enable_stock_filter = enable_stock_filter
        
        # 缓存标准化后的因子数据
        self._factors_zscore_cache = {}
        
        self._process_data()
        self._align()
        self._create_rebalance_dates()
    
    def _get_zscore_factors(self):
        """获取标准化后的因子数据（带缓存）"""
        if not self._factors_zscore_cache:
            for k, v in self.factors_wide.items():
                self._factors_zscore_cache[k] = _zscore_rowwise(v)
        return self._factors_zscore_cache

    def _process_data(self):
        """统一处理因子数据格式"""
        processed_factors = {}
        
        for factor_name, factor_data in self.factors_raw.items():
            # 检查是否为宽表格式（索引为日期，列为股票代码）
            if isinstance(factor_data.index, pd.DatetimeIndex) and factor_data.index.name == 'date':
                # 已经是宽表格式
                processed_factors[factor_name] = factor_data
            elif len(factor_data.columns) > 10:
                # 列数很多，可能是宽表格式
                if not isinstance(factor_data.index, pd.DatetimeIndex):
                    try:
                        factor_data.index = pd.to_datetime(factor_data.index)
                    except Exception:
                        continue
                if factor_data.index.name != 'date':
                    factor_data.index.name = 'date'
                processed_factors[factor_name] = factor_data
            elif len(factor_data.columns) == 3:
                # 长表转宽表
                col_names = list(factor_data.columns)
                date_cols = [col for col in col_names if 'date' in col.lower() or col == 'date']
                id_cols = [col for col in col_names if 'id' in col.lower() or 'code' in col.lower() or col == 'order_book_id']
                
                if date_cols and id_cols:
                    date_col = date_cols[0]
                    id_col = id_cols[0]
                    value_col = [col for col in col_names if col not in [date_col, id_col]][0]
                    
                    factor_data = factor_data.set_index(date_col).pivot(columns=id_col, values=value_col)
                    factor_data.index.name = 'date'
                    processed_factors[factor_name] = factor_data
                else:
                    # 如果无法识别列，保持原样
                    processed_factors[factor_name] = factor_data
            else:
                processed_factors[factor_name] = factor_data
        
        self.factors_raw = processed_factors

    def _align(self) -> None:
        """数据对齐"""
        try:
            px = self.prices.sort_values(['order_book_id', 'date'])
            px['date'] = pd.to_datetime(px['date'])
            px['return'] = px.groupby('order_book_id')['close'].pct_change(fill_method=None).shift(-1)
            px['future_return'] = px.groupby('order_book_id')['return'].shift(-1)
            
            # 构建收益数据
            ret = px[['date', 'order_book_id', 'future_return']].dropna(subset=['future_return'])
            ret_wide = ret.pivot(index='date', columns='order_book_id', values='future_return')
            
            common_dates = ret_wide.index
            common_cols = ret_wide.columns
            
            processed_factors = {}
            for name, f in self.factors_raw.items():
                if len(f.columns) == 3:
                    col_names = list(f.columns)
                    date_cols = [col for col in col_names if 'date' in col.lower() or col == 'date']
                    id_cols = [col for col in col_names if 'id' in col.lower() or col == 'code' in col.lower() or col == 'order_book_id']
                    
                    if date_cols and id_cols:
                        date_col = date_cols[0]
                        id_col = id_cols[0]
                        value_col = [col for col in col_names if col not in [date_col, id_col]][0]
                        
                        f_wide = f.set_index(date_col).pivot(columns=id_col, values=value_col)
                        f_wide.index.name = 'date'
                        processed_factors[name] = f_wide
                    else:
                        continue
                else:
                    processed_factors[name] = f
                
                common_dates = common_dates.intersection(processed_factors[name].index)
                common_cols = common_cols.intersection(processed_factors[name].columns)
                
            if len(common_dates) == 0 or len(common_cols) == 0:
                raise ValueError(f"没有共同的日期或股票: 日期数={len(common_dates)}, 股票数={len(common_cols)}")
            
            if len(processed_factors) == 0:
                raise ValueError("没有可用的因子数据")
                
            self.returns_wide = ret_wide.loc[common_dates, common_cols].sort_index()
            self.factors_wide = {k: v.loc[common_dates, common_cols].sort_index() for k, v in processed_factors.items()}
            self.factor_names = list(self.factors_wide.keys())
            
        except Exception as e:
            print(f"数据对齐失败: {str(e)}")
            raise

    def _create_rebalance_dates(self) -> None:
        """创建调仓日期列表"""
        all_dates = sorted(self.returns_wide.index.tolist())
        self.rebalance_dates = all_dates[::self.rebalance_period] if self.rebalance_period > 1 else all_dates

    def _filter_stocks_for_buy(self, date: pd.Timestamp) -> pd.Series:
        """股票筛选：过滤ST、停牌、涨停股票"""
        # 获取当日所有股票
        all_stocks = self.returns_wide.columns
        
        # 如果没有价格数据，返回所有股票
        if not hasattr(self, 'prices') or self.prices is None:
            return pd.Series(True, index=all_stocks)
        
        # 获取当日的数据
        date_data = self.prices[self.prices['date'] == date]
        if len(date_data) == 0:
            return pd.Series(True, index=all_stocks)
        
        # 创建过滤掩码
        mask = pd.Series(True, index=all_stocks)
        
        # 过滤涨停股票
        if 'limit_up_flag' in date_data.columns:
            limit_up_stocks = date_data[date_data['limit_up_flag'].astype(bool)]['order_book_id'].tolist()
            mask.loc[mask.index.isin(limit_up_stocks)] = False
        
        # 过滤ST股票
        if 'ST' in date_data.columns:
            st_stocks = date_data[date_data['ST'].astype(bool)]['order_book_id'].tolist()
            mask.loc[mask.index.isin(st_stocks)] = False
        
        # 过滤停牌股票
        if 'suspended' in date_data.columns:
            suspended_stocks = date_data[date_data['suspended'].astype(bool)]['order_book_id'].tolist()
            mask.loc[mask.index.isin(suspended_stocks)] = False
        
        return mask

    def _make_groups_on_rebalance_day(self, date: pd.Timestamp, n_groups: int = 10) -> pd.DataFrame:
        """单个调仓日分组"""
        factor_values = {}
        for factor_name, factor_data in self.factors_wide.items():
            if date in factor_data.index:
                factor_values[factor_name] = factor_data.loc[date]
        
        if not factor_values:
            return pd.DataFrame(columns=['rb_date', 'order_book_id', 'group'])
        
        # 合成因子值
        if hasattr(self, 'current_weights'):
            combined_factor = pd.Series(0.0, index=self.returns_wide.columns)
            for factor_name, factor_data in factor_values.items():
                if factor_name in self.current_weights:
                    combined_factor += factor_data * self.current_weights[factor_name]
        else:
            combined_factor = list(factor_values.values())[0]
        
        # 股票筛选
        valid_mask = self._filter_stocks_for_buy(date)
        valid_stocks = valid_mask[valid_mask].index
        
        if len(valid_stocks) < n_groups:
            return pd.DataFrame(columns=['rb_date', 'order_book_id', 'group'])
        
        valid_factor = combined_factor[valid_stocks].dropna()
        
        if len(valid_factor) < n_groups:
            return pd.DataFrame(columns=['rb_date', 'order_book_id', 'group'])
        
        # 分组
        try:
            ranks = valid_factor.rank(method='first', ascending=True)
            group_size = max(1, len(ranks) // n_groups)
            groups = ((ranks - 1) // group_size).clip(upper=n_groups - 1).astype(int)
            
            # 确保groups是有效的整数数组
            if np.any(np.isnan(groups)) or np.any(np.isinf(groups)):
                return pd.DataFrame(columns=['rb_date', 'order_book_id', 'group'])
            
            return pd.DataFrame({
                'rb_date': date,
                'order_book_id': valid_factor.index,
                'group': groups.values
            })
        except Exception:
            return pd.DataFrame(columns=['rb_date', 'order_book_id', 'group'])

    def compute_positions_and_returns(self, n_groups: int = 10) -> dict:
        """计算持仓和收益"""
        start_time = time.time()
        print(f"  [交易层] 构建调仓分组与每日持仓 (n_groups={n_groups}, 调仓周期={self.rebalance_period}) ...")
        
        # 统计过滤信息
        filter_stats = {
            'limit_up_count': 0,
            'st_count': 0,
            'suspended_count': 0,
            'total_filtered': 0
        }
        
        # 1. 预计算所有调仓日的分组
        group_labels_list = []
        for date in self.rebalance_dates:
            if date in self.returns_wide.index:
                # 统计当日过滤的股票数量
                if self.enable_stock_filter:
                    valid_mask = self._filter_stocks_for_buy(date)
                    total_stocks = len(valid_mask)
                    valid_stocks = valid_mask.sum()
                    filtered_stocks = total_stocks - valid_stocks
                    
                    if filtered_stocks > 0:
                        # 获取具体的过滤原因
                        date_data = self.prices[self.prices['date'] == date]
                        if len(date_data) > 0:
                            if 'limit_up_flag' in date_data.columns:
                                limit_up_count = date_data['limit_up_flag'].astype(bool).sum()
                                filter_stats['limit_up_count'] += limit_up_count
                            
                            if 'ST' in date_data.columns:
                                st_count = date_data['ST'].astype(bool).sum()
                                filter_stats['st_count'] += st_count
                            
                            if 'suspended' in date_data.columns:
                                suspended_count = date_data['suspended'].astype(bool).sum()
                                filter_stats['suspended_count'] += suspended_count
                        
                        filter_stats['total_filtered'] += filtered_stocks
                
                res = self._make_groups_on_rebalance_day(date, n_groups)
                if len(res) > 0:
                    group_labels_list.append(res)
        
        if not group_labels_list:
            raise ValueError("没有生成任何分组数据")
        
        group_labels_rebalance = pd.concat(group_labels_list, ignore_index=True)
        
        # 2. 简化的日期映射
        all_dates = sorted(self.returns_wide.index)
        dates_df = pd.DataFrame({'date': all_dates})
        rb_df = pd.DataFrame({'rb_date': pd.to_datetime(self.rebalance_dates)})
        map_df = pd.merge_asof(dates_df, rb_df, left_on='date', right_on='rb_date', direction='backward')
        positions_daily = map_df.merge(group_labels_rebalance, how='left', on='rb_date')
        positions_daily = positions_daily[['date', 'rb_date', 'order_book_id', 'group']]
        
        # 3. 合并收益数据
        ret_long = self.returns_wide.stack().rename('ret').reset_index()
        ret_long.columns = ['date', 'order_book_id', 'ret']
        pr = positions_daily.merge(ret_long, on=['date', 'order_book_id'], how='left')
        
        # 4. 过滤和类型转换
        pr = pr.dropna(subset=['group']).copy()
        pr['group'] = pr['group'].astype(int)
        
        # 5. 计算权重
        counts = pr.groupby(['rb_date','group'])['order_book_id'].nunique().rename('n').reset_index()
        pr = pr.merge(counts, on=['rb_date','group'], how='left')
        pr['start_weight'] = 1.0 / pr['n']
        
        # 6. 计算累计收益
        pr['gross'] = (pr['ret'] + 1.0).fillna(1.0)
        pr['stock_cum'] = pr.groupby(['rb_date','group','order_book_id'])['gross'].transform('cumprod')
        pr['holding_weight'] = np.where(
            pr['rb_date'] == pr['date'],
            pr['start_weight'],
            pr['start_weight'] * pr['stock_cum']
        )
        
        # 7. 权重归一化
        pr['holding_weight'] = pr.groupby(['date','group'])['holding_weight'].transform(lambda x: x / x.sum() if x.sum() != 0 else 0)
        
        # 8. 计算分组收益
        group_returns = pr.groupby(['date', 'group']).agg({
            'ret': lambda x: np.average(x, weights=pr.loc[x.index, 'holding_weight'])
        }).reset_index()
        group_returns.columns = ['date', 'group', 'group_return']
        
        # 9. 创建宽表
        pr_nodup = pr.drop_duplicates(subset=['date', 'group'], keep='first')
        group_daily_returns = pr_nodup.pivot(index='date', columns='group', values='group_return').fillna(0.0)
        group_daily_returns = group_daily_returns.reindex(sorted(group_daily_returns.columns), axis=1)
        
        # 10. 累计净值
        group_cum_nav = (group_daily_returns + 1).cumprod()
        group_daily_returns = group_daily_returns.reindex(all_dates, method='ffill').fillna(0.0)
        group_cum_nav = group_cum_nav.reindex(all_dates, method='ffill')
        
        print(f"  收益计算完成，耗时: {time.time() - start_time:.2f}秒")
        # 显示过滤统计信息（合并为一行）
        if self.enable_stock_filter and filter_stats['total_filtered'] > 0:
            print(f"[交易层] 过滤统计: 涨停: {filter_stats['limit_up_count']} 次, ST: {filter_stats['st_count']} 次, 停牌: {filter_stats['suspended_count']} 次, 总计: {filter_stats['total_filtered']} 次")
        elif self.enable_stock_filter:
            print(f"[交易层] 未过滤任何股票")
        else:
            print(f"[交易层] 股票过滤已禁用")
        
        return {
            'group_daily_returns': group_daily_returns,
            'group_cum_nav': group_cum_nav,
            'positions_daily': positions_daily,
            'rebalance_info': {
                'rebalance_dates': self.rebalance_dates,
                'group_labels_rebalance': group_labels_rebalance
            }
        }

    def _compute_weight_history(self, method: str, N: int) -> pd.DataFrame:
        """计算权重历史"""
        try:
            dates = self.returns_wide.index
            y = self.returns_wide
            Xz = self._get_zscore_factors()

            if method == 'univariate':
                ic_matrix = np.full((len(dates), len(self.factor_names)), np.nan)
                for i, dt in enumerate(tqdm(dates, desc='univariate betas')):
                    try:
                        yy = y.loc[dt]
                        for j, fname in enumerate(self.factor_names):
                            if fname in Xz and dt in Xz[fname].index:
                                factor_vals = Xz[fname].loc[dt]
                                ic_val = self._calculate_ic(factor_vals, yy)
                                ic_matrix[i, j] = ic_val
                    except Exception:
                        pass
                raw_df = pd.DataFrame(ic_matrix, index=dates, columns=self.factor_names)
            elif method == 'multivariate':
                beta_matrix = np.full((len(dates), len(self.factor_names)), np.nan)
                for i, dt in enumerate(tqdm(dates, desc='multivariate betas')):
                    try:
                        yy = y.loc[dt]
                        Xmat = pd.DataFrame({f: Xz[f].loc[dt] for f in self.factor_names if f in Xz and dt in Xz[f].index})
                        if not Xmat.empty and len(Xmat.columns) > 0:
                            betas = self._calculate_multivariate_beta(Xmat, yy)
                            if len(betas) == len(self.factor_names):
                                beta_matrix[i, :] = betas
                    except Exception:
                        pass
                raw_df = pd.DataFrame(beta_matrix, index=dates, columns=self.factor_names)
            elif method == 'rank_ic':
                ic_matrix = np.full((len(dates), len(self.factor_names)), np.nan)
                for i, dt in enumerate(tqdm(dates, desc='rank IC')):
                    try:
                        yy = y.loc[dt]
                        for j, fname in enumerate(self.factor_names):
                            if fname in Xz and dt in Xz[fname].index:
                                factor_vals = Xz[fname].loc[dt]
                                ic_val = self._calculate_ic(factor_vals, yy)
                                ic_matrix[i, j] = ic_val
                    except Exception:
                        pass
                raw_df = pd.DataFrame(ic_matrix, index=dates, columns=self.factor_names)
            else:
                raise ValueError("method 必须是 {'univariate','multivariate','rank_ic'} 之一")

            if raw_df.isna().all().all():
                raise ValueError("没有生成任何有效权重数据")

            # 恢复N日滚动平均逻辑
            min_periods = min(5, N)  # 确保min_periods <= N
            weight_hist = raw_df.rolling(N, min_periods=min_periods).mean().apply(_l1_normalize, axis=1)
            return weight_hist.fillna(0.0)
            
        except Exception as e:
            raise

    def _calculate_ic(self, X: pd.Series, y: pd.Series) -> float:
        """计算IC"""
        mask = ~(np.isnan(X.values) | np.isnan(y.values))
        if np.sum(mask) < 10:
            return np.nan
            
        x_vals = X.values[mask]
        y_vals = y.values[mask]
        
        if np.std(x_vals, ddof=0) == 0 or np.std(y_vals, ddof=0) == 0:
            return np.nan
            
        try:
            return float(stats.spearmanr(x_vals, y_vals)[0])
        except Exception:
            return np.nan

    def _calculate_multivariate_beta(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """计算多元回归系数"""
        mask = ~(np.isnan(X.values).any(axis=1) | np.isnan(y.values))
        if np.sum(mask) < X.shape[1] + 5:
            return np.array([np.nan] * X.shape[1])
            
        Xv, yv = X.values[mask], y.values[mask]
        
        try:
            X_mean = np.nanmean(Xv, axis=0)
            X_std = np.nanstd(Xv, axis=0, ddof=0)
            X_std[X_std == 0] = 1.0
            Xv_scaled = (Xv - X_mean) / X_std
            
            XtX = Xv_scaled.T @ Xv_scaled
            Xty = Xv_scaled.T @ yv
            
            try:
                beta = np.linalg.solve(XtX, Xty)
                return beta.astype(float)
            except np.linalg.LinAlgError:
                beta = np.linalg.pinv(XtX) @ Xty
                return beta.astype(float)
                
        except Exception:
            return np.array([np.nan] * X.shape[1])

    def build(self, method: str, N: int, n_groups: int = 10, lag_days: int = 5, normalize_lag: bool = True) -> ComboResult:
        """
        按用户指定的新逻辑构建合成因子：
        - 计算权重历史（不改动原逻辑）
        - 合成规则：
            对于 k = 0 (Z(t)) 和 k = 1 (Z(t-1)) 使用 W(t-1);
            对于 k >= 2 使用 W(t-k).
        - lag_days 指明最大回溯天数 L（包含当天 t 的 k=0）。
        """
        start_time = time.time()
        try:
            weight_hist = self._compute_weight_history(method, N)
            # 兜底 current_weights
            self.current_weights = (weight_hist.iloc[-1] if len(weight_hist) > 0 else pd.Series(0.0, index=self.factor_names))

            # 准备数据
            Z = self._get_zscore_factors()
            idx = self.returns_wide.index
            cols = self.returns_wide.columns

            # 对齐权重矩阵
            W = weight_hist.reindex(idx).fillna(0.0)
            W_prev = W.shift(1).fillna(0.0)  # W(t-1)

            # 预分配合成矩阵
            combined = pd.DataFrame(0.0, index=idx, columns=cols)

            L = int(lag_days) if lag_days and lag_days > 0 else 1

            # 逐因子合成（按用户指定规则）
            for fname in self.factor_names:
                Zf = Z[fname].reindex(index=idx, columns=cols)

                # 对每个滞后 k (0..L-1)
                for k in range(0, L):
                    # 因子值 Z(t-k)
                    Z_shift_k = Zf.shift(k).fillna(0.0)

                    # 权重选择规则：
                    # k==0 或 k==1 -> 使用 W_prev (即 W(t-1))
                    # 否则 -> 使用 W.shift(k) (即 W(t-k))
                    if k in (0, 1):
                        weight_series = W_prev[fname]
                    else:
                        weight_series = W[fname].shift(k).fillna(0.0)

                    contrib_k = Z_shift_k.mul(weight_series, axis=0).fillna(0.0)
                    combined = combined.add(contrib_k, fill_value=0.0)

            # 归一化：按有效滞后数做均值化，保持尺度稳定
            if normalize_lag:
                eff = pd.Series([min(L, i+0) for i in range(len(idx))], index=idx).replace(0, np.nan)
                combined = combined.div(eff, axis=0).fillna(0.0)

            self.combined_factor = combined

            # 计算收益与IC
            # 将合成因子转换为单因子测试需要的格式
            combined_long = combined.stack().reset_index()
            combined_long.columns = ['date', 'order_book_id', 'factor_value']
            combined_long = combined_long.merge(
                self.prices[['date', 'order_book_id', 'close']], 
                on=['date', 'order_book_id'], 
                how='inner'
            )
            
            # 创建单因子分析器
            single_analyzer = SingleFactorAnalyzer(
                factor_data=combined_long,
                returns_data=self.prices,
                factor_name='combined_factor',
                rebalance_period=self.rebalance_period
            )
            
            # 使用单因子测试类计算持仓和收益
            positions_returns = single_analyzer.compute_positions_and_returns(n_groups=n_groups)
            
            # 计算IC统计（统一使用max_lag=5，避免重复调用）
            ic_stats = single_analyzer.compute_ic_and_stats(method='spearman', max_lag=1)
            ic_series = ic_stats['IC_series']
            
            # 直接计算多空收益和统计指标，避免重复调用analyze_performance
            long_short_data = self._calculate_performance_metrics(positions_returns, ic_stats, n_groups)
            
            # 计算纯多、纯空、多空的年化收益和夏普比例
            # 纯多收益（最高分组）
            high_group_returns = positions_returns['group_daily_returns'].iloc[:, -1].fillna(0)
            high_annual_return = (1 + high_group_returns).prod() ** (252 / len(high_group_returns)) - 1
            high_volatility = high_group_returns.std() * np.sqrt(252)
            high_sharpe = high_annual_return / high_volatility if high_volatility > 0 else np.nan
            
            # 纯空收益（最低分组）
            low_group_returns = positions_returns['group_daily_returns'].iloc[:, 0].fillna(0)
            low_annual_return = (1 + low_group_returns).prod() ** (252 / len(low_group_returns)) - 1
            low_volatility = low_group_returns.std() * np.sqrt(252)
            low_sharpe = low_annual_return / low_volatility if low_volatility > 0 else np.nan
            
            # 多空组合收益
            long_short_returns = high_group_returns - low_group_returns
            ls_annual_return = (1 + long_short_returns).prod() ** (252 / len(long_short_returns)) - 1
            ls_volatility = long_short_returns.std() * np.sqrt(252)
            ls_sharpe = ls_annual_return / ls_volatility if ls_volatility > 0 else np.nan
            
            # 提取需要的多空数据
            long_short_data = {
                'long_only_returns': long_short_data['long_short']['long_returns'],
                'long_returns': long_short_data['long_short']['long_returns'],
                'short_returns': long_short_data['long_short']['short_returns'],
                'long_short_returns': long_short_data['long_short']['long_short_returns'],
                'summary': {
                    'annual_return': long_short_data['long_short']['stats']['annualized_return'],
                    'annual_volatility': long_short_data['long_short']['stats']['std_return'] * np.sqrt(252),
                    'sharpe_ratio': long_short_data['long_short']['stats']['sharpe_ratio'],
                    'long_annual_return': long_short_data['long_short']['stats']['annualized_return'],
                    'long_sharpe_ratio': long_short_data['long_short']['stats']['sharpe_ratio'],
                    # 新增：纯多、纯空、多空的年化收益和夏普比例
                    'high_annual_return': high_annual_return,
                    'high_sharpe_ratio': high_sharpe,
                    'low_annual_return': low_annual_return,
                    'low_sharpe_ratio': low_sharpe,
                    'ls_annual_return': ls_annual_return,
                    'ls_sharpe_ratio': ls_sharpe,
                }
            }

            ic_mean = float(ic_series.mean()) if len(ic_series) else np.nan
            ic_std = float(ic_series.std()) if len(ic_series) else np.nan
            icir = ic_mean / ic_std if (ic_std and ic_std != 0) else np.nan
            ic_pos = float((ic_series > 0).mean()) if len(ic_series) else np.nan

            long_short_data['summary'].update({
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'icir': icir,
                'ic_positive_ratio': ic_pos,
            })

            return ComboResult(
                combined_factor=combined,
                weight_history=weight_hist,
                ic_series=ic_series,
                long_only_returns=long_short_data['long_only_returns'],
                group_returns=positions_returns['group_daily_returns'],
                summary=long_short_data['summary'],
            )

        except Exception as e:
            raise

    def _calculate_performance_metrics(self, positions_returns, ic_stats, n_groups):
        """计算绩效指标，避免重复调用SingleFactorAnalyzer的方法"""
        print("  [绩效层] 汇总绩效指标 ...")
        
        grp_ret = positions_returns['group_daily_returns']
        
        # 自动判断多空方向（基于IC均值）
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

        # 计算多空统计指标
        stats_pack = {
            'mean_return': long_short_returns.mean(),
            'std_return': long_short_returns.std(),
            'sharpe_ratio': (long_short_returns.mean() / long_short_returns.std()) if long_short_returns.std() not in [0, np.nan] else np.nan,
            'max_drawdown': self._calculate_max_drawdown(cum_ls),
            'win_rate': (long_short_returns > 0).mean(),
            'total_return': (cum_ls.iloc[-1] - 1) if len(cum_ls) else np.nan,
            'annualized_return': self._calculate_annualized_return(long_short_returns)
        }

        return {
            'long_short': {
                'long_short_returns': long_short_returns,
                'cumulative_ls_returns': cum_ls,
                'long_returns': long_returns,
                'short_returns': short_returns,
                'stats': stats_pack
            }
        }

    def _calculate_max_drawdown(self, cumulative_returns):
        """计算最大回撤"""
        if len(cumulative_returns) == 0:
            return np.nan
        running_max = cumulative_returns.expanding().max()
        dd = (cumulative_returns - running_max) / running_max
        return dd.min()

    def _calculate_annualized_return(self, returns):
        """计算年化收益率"""
        if len(returns) == 0:
            return np.nan
        total_return = (1 + returns.fillna(0)).prod() - 1
        years = len(returns) / 252.0
        return (1 + total_return) ** (1.0 / years) - 1 if years > 0 else np.nan



    def render_report(self, result_map: Dict[str, ComboResult], html_path: str) -> None:
        """生成HTML报告"""
        page = Page(layout=Page.SimplePageLayout)

        # 1) 权重热力图
        weights_concat = []
        for m, res in result_map.items():
            tmp = res.weight_history.copy()
            tmp.columns = pd.MultiIndex.from_product([[m], tmp.columns])
            weights_concat.append(tmp)
        W = pd.concat(weights_concat, axis=1).sort_index()

        y_labels = [d.strftime('%Y-%m-%d') for d in W.index]
        x_labels = [f"{m}|{f}" for m, f in W.columns]
        data = []
        for i, dt in enumerate(W.index):
            row = W.loc[dt].values
            for j, v in enumerate(row):
                if pd.notna(v):
                    data.append([j, i, float(np.round(v, 6))])
        
        heat = (
            HeatMap()
            .add_xaxis(x_labels)
            .add_yaxis("权重", y_labels, data, label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(
                title_opts=opts.TitleOpts(title="滚动权重热力图"),
                visualmap_opts=opts.VisualMapOpts(
                    min_=float(np.nanmin(W.values)), 
                    max_=float(np.nanmax(W.values)),
                    range_color=["#FF0000", "#FFFFFF", "#0000FF"]
                ),
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45)),
                datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
            )
        )
        page.add(heat)

        # 2) 累计IC
        line_ic = Line()
        idx_ref = None
        for m, res in result_map.items():
            s = res.ic_series.cumsum().dropna()
            if idx_ref is None:
                idx_ref = s.index
                line_ic.add_xaxis([d.strftime('%Y-%m-%d') for d in idx_ref])
            s = s.reindex(idx_ref).ffill().fillna(0)
            line_ic.add_yaxis(m, s.values.tolist(), is_smooth=True, symbol_size=0, label_opts=opts.LabelOpts(is_show=False))
        line_ic.set_global_opts(
            title_opts=opts.TitleOpts(title="累计IC对比"),
            datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45)),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
        )
        page.add(line_ic)

        # 3) 最高分组累计收益（log10）
        line_ls = Line()
        idx_ref = None
        for m, res in result_map.items():
            s = (1 + res.long_only_returns.fillna(0)).cumprod()
            if idx_ref is None:
                idx_ref = s.index
                line_ls.add_xaxis([d.strftime('%Y-%m-%d') for d in idx_ref])
            s = s.reindex(idx_ref).ffill().fillna(1)
            s_log = np.log10(s)
            line_ls.add_yaxis(m, s_log.values.tolist(), is_smooth=True, symbol_size=0, label_opts=opts.LabelOpts(is_show=False))
        line_ls.set_global_opts(
            title_opts=opts.TitleOpts(title="最高分组累计收益对比 (因子值最大的10%, log10)"),
            datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45)),
            yaxis_opts=opts.AxisOpts(name="累计收益 (log10)"),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
        )
        page.add(line_ls)

        # 4) 最低分组累计收益（log10）
        line_ls_low = Line()
        idx_ref = None
        for m, res in result_map.items():
            # 获取最低分组收益
            grp_ret = res.group_returns
            if grp_ret.shape[1] >= 1:
                low_group_returns = grp_ret.iloc[:, 0].fillna(0)  # 第一列是最低分组
                s = (1 + low_group_returns).cumprod()
                if idx_ref is None:
                    idx_ref = s.index
                    line_ls_low.add_xaxis([d.strftime('%Y-%m-%d') for d in idx_ref])
                s = s.reindex(idx_ref).ffill().fillna(1)
                s_log = np.log10(s)
                line_ls_low.add_yaxis(m, s_log.values.tolist(), is_smooth=True, symbol_size=0, label_opts=opts.LabelOpts(is_show=False))
        if idx_ref is not None:
            line_ls_low.set_global_opts(
                title_opts=opts.TitleOpts(title="最低分组累计收益对比 (因子值最小的10%, log10)"),
                datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45)),
                yaxis_opts=opts.AxisOpts(name="累计收益 (log10)"),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
            )
            page.add(line_ls_low)

        # 5) 多空组合累计收益（log10）
        line_ls_combined = Line()
        idx_ref = None
        for m, res in result_map.items():
            # 计算多空组合收益
            grp_ret = res.group_returns
            if grp_ret.shape[1] >= 2:
                high_group_returns = grp_ret.iloc[:, -1].fillna(0)  # 最高分组
                low_group_returns = grp_ret.iloc[:, 0].fillna(0)    # 最低分组
                long_short_returns = high_group_returns - low_group_returns  # 多空组合
                s = (1 + long_short_returns).cumprod()
                if idx_ref is None:
                    idx_ref = s.index
                    line_ls_combined.add_xaxis([d.strftime('%Y-%m-%d') for d in idx_ref])
                s = s.reindex(idx_ref).ffill().fillna(1)
                s_log = np.log10(s)
                line_ls_combined.add_yaxis(m, s_log.values.tolist(), is_smooth=True, symbol_size=0, label_opts=opts.LabelOpts(is_show=False))
        if idx_ref is not None:
            line_ls_combined.set_global_opts(
                title_opts=opts.TitleOpts(title="多空组合累计收益对比 (最高分组-最低分组, log10)"),
                datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45)),
                yaxis_opts=opts.AxisOpts(name="累计收益 (log10)"),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")
            )
            page.add(line_ls_combined)

        # 6) 分组累计收益柱状图
        best_name = None
        best_sharpe = -np.inf
        for m, res in result_map.items():
            sh = res.summary.get('sharpe_ratio', np.nan)
            if pd.notna(sh) and sh > best_sharpe:
                best_sharpe = sh
                best_name = m
        
        if best_name is not None:
            grp_returns = result_map[best_name].group_returns
            grp_cumulative = (1 + grp_returns.fillna(0)).cumprod()
            grp_final = grp_cumulative.iloc[-1] if len(grp_cumulative) > 0 else pd.Series()
            
            if len(grp_final) > 0:
                x = [f"G{int(i)+1}" for i in grp_final.index.tolist()]
                y = [float(v) if pd.notna(v) else 1.0 for v in grp_final.values]
                
                tooltip_data = []
                for i, val in enumerate(y):
                    pct = (val - 1.0) * 100 if val > 0 else 0.0
                    tooltip_data.append(f"分组{i+1}<br/>累计净值: {val:.4f}<br/>收益率: {pct:.2f}%")
                
                bar = (
                    Bar()
                    .add_xaxis(x)
                    .add_yaxis("累计收益", y, label_opts=opts.LabelOpts(is_show=False))
                    .set_global_opts(
                        title_opts=opts.TitleOpts(title=f"最佳方法: {best_name} 分组累计收益"),
                        xaxis_opts=opts.AxisOpts(name="分组"),
                        yaxis_opts=opts.AxisOpts(name="累计收益"),
                        tooltip_opts=opts.TooltipOpts(
                            trigger="axis",
                            axis_pointer_type="shadow",
                            formatter=lambda params: tooltip_data[params[0].dataIndex] if params and len(params) > 0 else ""
                        )
                    )
                )
                page.add(bar)

        # 7) 性能表格
        rows = []
        for m, res in result_map.items():
            sm = res.summary
            rows.append([
                m,
                f"{sm.get('ic_mean', np.nan):.4f}" if pd.notna(sm.get('ic_mean', np.nan)) else 'N/A',
                f"{sm.get('icir', np.nan):.4f}" if pd.notna(sm.get('icir', np.nan)) else 'N/A',
                f"{sm.get('ic_positive_ratio', np.nan):.4f}" if pd.notna(sm.get('ic_positive_ratio', np.nan)) else 'N/A',
                f"{sm.get('high_annual_return', np.nan):.2%}" if pd.notna(sm.get('high_annual_return', np.nan)) else 'N/A',
                f"{sm.get('high_sharpe_ratio', np.nan):.2f}" if pd.notna(sm.get('high_sharpe_ratio', np.nan)) else 'N/A',
                f"{sm.get('low_annual_return', np.nan):.2%}" if pd.notna(sm.get('low_annual_return', np.nan)) else 'N/A',
                f"{sm.get('low_sharpe_ratio', np.nan):.2f}" if pd.notna(sm.get('low_sharpe_ratio', np.nan)) else 'N/A',
                f"{sm.get('ls_annual_return', np.nan):.2%}" if pd.notna(sm.get('ls_annual_return', np.nan)) else 'N/A',
                f"{sm.get('ls_sharpe_ratio', np.nan):.2f}" if pd.notna(sm.get('ls_sharpe_ratio', np.nan)) else 'N/A',
            ])
        table = (
            Table()
            .add(headers=["方法", "IC均值", "ICIR", "IC正比例", "纯多年化收益", "纯多夏普", "纯空年化收益", "纯空夏普", "多空年化收益", "多空夏普"], rows=rows)
        )
        page.add(table)
        
        page.render(html_path)