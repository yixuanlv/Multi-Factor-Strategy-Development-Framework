# -*- coding: utf-8 -*-
"""
单一全局窗口N的因子合成（可直接替换原脚本）
------------------------------------------------
支持三种权重方法（均为滚动窗口N上的时变权重）：
1) univariate  : 一元回归系数 (未来收益 ~ 单因子) 的滚动均值
2) multivariate : 多元回归系数 (未来收益 ~ 所有因子) 的滚动均值
3) rank_ic      : Rank IC(因子 vs 未来收益) 的滚动均值

要点与修正：
- 以**所有因子与收益的共同交集**统一对齐，避免“用第一个因子来决定收益矩阵”的偏差。
- 权重为**时变序列**：每个交易日都有一组权重（由过去N期的日度系数/IC平均得到）。
- 合成因子按日 Z-Score 标准化后线性加权，默认对权重做 L1 归一化（防止权重绝对值过大）。
- pyecharts 输出一个 HTML（包含权重热力图、累计IC、LS曲线、分组收益柱状图与性能表）。

依赖：pandas, numpy, scipy, scikit-learn, pyecharts, tqdm
"""

from __future__ import annotations
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from pyecharts import options as opts
from pyecharts.charts import Line, Bar, HeatMap, Page
from pyecharts.components import Table


# ------------------------- 工具函数 -------------------------

def _zscore_rowwise(df: pd.DataFrame) -> pd.DataFrame:
    """对每个日期（行）做横截面Z-Score；全NaN或std=0时返回0。"""
    def _z(s: pd.Series) -> pd.Series:
        m = s.mean()
        v = s.std(ddof=0)
        if not np.isfinite(v) or v == 0:
            return pd.Series(0.0, index=s.index)
        return (s - m) / v
    return df.apply(_z, axis=1)


def _safe_rank_rowwise(df: pd.DataFrame) -> pd.DataFrame:
    """对每日做百分位秩变换（0~1），空值先行填充为行均值以稳定秩。"""
    fill = df.T.fillna(df.mean(axis=1)).T
    return fill.rank(axis=1, pct=True)


def _l1_normalize(w: pd.Series, eps: float = 1e-12) -> pd.Series:
    s = np.sum(np.abs(w.values))
    if not np.isfinite(s) or s < eps:
        return pd.Series(0.0, index=w.index)
    return w / s


# ------------------------- 数据模型 -------------------------

@dataclass
class ComboResult:
    combined_factor: pd.DataFrame           # (date x stock) 合成因子
    weight_history: pd.DataFrame            # (date x factor) 权重历史
    ic_series: pd.Series                    # 合成因子的日度IC（Spearman）
    long_only_returns: pd.Series            # 纯多组合日度收益
    group_returns: pd.DataFrame             # 各分组日度未来收益
    summary: dict                           # 统计指标


# ------------------------- 主类 -------------------------

class FactorCombiner:
    def __init__(
        self,
        factors: Dict[str, pd.DataFrame],
        prices: pd.DataFrame,
        rebalance_period: int = 1,
        enable_stock_filter: bool = True,
        future_return_col: str = "return",
    ) -> None:
        """
        Parameters
        ----------
        factors : dict[name -> DataFrame]
            每个因子可以是长表格式（三列: ['date', 'order_book_id', 'factor_value']）
            或宽表格式（索引为日期，列为股票代码，值为因子值）
        prices : DataFrame
            至少三列: ['date', 'order_book_id', 'close']
            可选列: ['limit_up_flag', 'limit_down_flag', 'ST', 'suspended']
        rebalance_period : int
            调仓周期（>=1），表示每隔多少个交易日调仓一次
        enable_stock_filter : bool
            是否启用股票筛选（剔除ST、涨停、停牌等）
        """
        self.factors_raw = {k: v.copy() for k, v in factors.items()}
        self.prices = prices.copy()
        self.rebalance_period = int(rebalance_period)
        self.enable_stock_filter = bool(enable_stock_filter)

        # 检测并处理数据格式
        self._detect_and_convert_data_format()
        
        # 确保价格数据的日期列是datetime格式
        self.prices['date'] = pd.to_datetime(self.prices['date'])

        self.factor_names = list(self.factors_raw.keys())
        self._align()
        self._create_rebalance_dates()

    def _detect_and_convert_data_format(self):
        """处理因子数据格式，统一为宽表格式"""
        processed_factors = {}
        
        for factor_name, factor_data in self.factors_raw.items():
            # 检查是否为宽表格式（索引为日期，列为股票代码）
            if len(factor_data.columns) > 10:
                # 处理索引：确保是DatetimeIndex且名称为'date'
                if not isinstance(factor_data.index, pd.DatetimeIndex):
                    # 尝试将索引转换为DatetimeIndex
                    try:
                        factor_data.index = pd.to_datetime(factor_data.index)
                        
                    except Exception as e:
                        continue
                
                # 确保索引名称为 'date'
                if factor_data.index.name != 'date':
                    factor_data.index.name = 'date'
                
                processed_factors[factor_name] = factor_data
            else:
                # 如果不是宽表格式，尝试转换为宽表
                if len(factor_data.columns) == 3:
                    # 假设是三列的长表格式
                    col_names = list(factor_data.columns)
                    date_col = [col for col in col_names if 'date' in col.lower() or col == 'date'][0]
                    id_col = [col for col in col_names if 'id' in col.lower() or 'code' in col.lower() or col == 'order_book_id'][0]
                    value_col = [col for col in col_names if col not in [date_col, id_col]][0]
                    
                    # 转换为宽表
                    factor_data = factor_data.set_index(date_col).pivot(columns=id_col, values=value_col)
                    factor_data.index.name = 'date'
                    processed_factors[factor_name] = factor_data
                else:
                    # 对于行情数据，保持原格式，在_align中处理
                    processed_factors[factor_name] = factor_data
        
        self.factors_raw = processed_factors

    # ---- 对齐与宽表 ----
    def _align(self) -> None:
        """数据对齐和预处理"""
        try:
            print("开始数据对齐...")
            print(f"原始价格数据形状: {self.prices.shape}")
           # print(f"原始因子数据: {list(self.factors_raw.keys())}")
            
            # 1. 计算未来收益
            px = self.prices.sort_values(['order_book_id', 'date'])
            px['return'] = px.groupby('order_book_id')['close'].pct_change(fill_method=None).shift(-1)
            
            # 保留筛选列
            filter_cols = ['limit_up_flag', 'limit_down_flag', 'ST', 'suspended']
            available_filter_cols = [col for col in filter_cols if col in px.columns]
            
            ret = px[['date', 'order_book_id', 'return'] + available_filter_cols].dropna(subset=['return'])
            print(f"未来收益数据形状: {ret.shape}")
            
            # 2. 转换为宽表格式
            ret_wide = ret.pivot(index='date', columns='order_book_id', values='return')
            
            # 创建筛选列的宽表
            self.filter_flags = {}
            for col in available_filter_cols:
                flag_wide = ret.pivot(index='date', columns='order_book_id', values=col)
                self.filter_flags[col] = flag_wide
            
            print(f"收益宽表形状: {ret_wide.shape}")
            print(f"筛选标志列: {list(self.filter_flags.keys())}")
            
            # 3. 数据对齐（求所有矩阵的共同交集）
            common_dates = ret_wide.index
            common_cols = ret_wide.columns
            
            # 处理因子数据，确保都是宽表格式
            processed_factors = {}
            for name, f in self.factors_raw.items():
                #print(f"处理因子 {name}: 原始形状 {f.shape}")
                
                # 如果因子数据是长表格式，转换为宽表
                if len(f.columns) == 3:
                    col_names = list(f.columns)
                    date_col = [col for col in col_names if 'date' in col.lower() or col == 'date'][0]
                    id_col = [col for col in col_names if 'id' in col.lower() or 'code' in col.lower() or col == 'order_book_id'][0]
                    value_col = [col for col in col_names if col not in [date_col, id_col]][0]
                    
                    #print(f"  长表格式: date_col={date_col}, id_col={id_col}, value_col={value_col}")
                    
                    # 转换为宽表
                    f_wide = f.set_index(date_col).pivot(columns=id_col, values=value_col)
                    f_wide.index.name = 'date'
                    processed_factors[name] = f_wide
                    #print(f"  转换为宽表后形状: {f_wide.shape}")
                else:
                    # 已经是宽表格式
                    processed_factors[name] = f
                    #print(f"  已经是宽表格式: {f.shape}")
                
                # 更新共同交集
                common_dates = common_dates.intersection(processed_factors[name].index)
                common_cols = common_cols.intersection(processed_factors[name].columns)
                #print(f"  当前共同交集: 日期={len(common_dates)}, 股票={len(common_cols)}")
                
            if len(common_dates) == 0 or len(common_cols) == 0:
                raise ValueError(f"没有共同的日期或股票: 日期数={len(common_dates)}, 股票数={len(common_cols)}")
                
            # 4. 裁剪数据
            self.returns_wide = ret_wide.loc[common_dates, common_cols].sort_index()
            self.factors_wide = {k: v.loc[common_dates, common_cols].sort_index() for k, v in processed_factors.items()}
            self.factor_names = list(self.factors_wide.keys())
            
            # 裁剪筛选标志列
            for col in self.filter_flags:
                self.filter_flags[col] = self.filter_flags[col].loc[common_dates, common_cols].sort_index()
            
            # 5. 输出最终数据信息
            print(f"数据对齐完成: {len(common_dates)} 个交易日, {len(common_cols)} 只股票, {len(self.factor_names)} 个因子")
            #print(f"最终收益数据形状: {self.returns_wide.shape}")
            #for name, f in self.factors_wide.items():
                
                #print(f"  因子 {name} 最终形状: {f.shape}")
            
        except Exception as e:
            print(f"数据对齐失败: {str(e)}")
            raise

    def _create_rebalance_dates(self) -> None:
        """创建调仓日期列表，按照调仓周期选择日期"""
        all_dates = sorted(self.returns_wide.index.tolist())
        
        if self.rebalance_period > 1:
            # 按照调仓周期选择日期
            self.rebalance_dates = all_dates[::self.rebalance_period]
        else:
            # 每日调仓
            self.rebalance_dates = all_dates
            
        print(f"创建调仓日期列表: {len(self.rebalance_dates)} 个调仓日 (调仓周期: {self.rebalance_period})")

    def _filter_stocks_for_buy(self, date: pd.Timestamp) -> pd.Series:
        """买入过滤（非涨停、非ST、非停牌），返回当日可交易的股票列表"""
        if not self.enable_stock_filter:
            return pd.Series(True, index=self.returns_wide.columns)
        
        mask = pd.Series(True, index=self.returns_wide.columns)
        
        # 检查是否有相应的列，如果有则进行过滤
        if 'limit_up_flag' in self.filter_flags:
            flag_data = self.filter_flags['limit_up_flag'].loc[date]
            mask &= ~flag_data.astype(bool)
        if 'ST' in self.filter_flags:
            flag_data = self.filter_flags['ST'].loc[date]
            mask &= ~flag_data.astype(bool)
        if 'suspended' in self.filter_flags:
            flag_data = self.filter_flags['suspended'].loc[date]
            mask &= ~flag_data.astype(bool)
            
        return mask

    def _make_groups_on_rebalance_day(self, date: pd.Timestamp, n_groups: int = 10) -> pd.DataFrame:
        """
        单个调仓日：因子从小到大排序分组，返回列 ['rb_date','order_book_id','group']。
        使用rank -> 等容量划分，避免qcut重复边界问题。
        """
        # 获取当日因子值
        factor_values = {}
        for factor_name, factor_data in self.factors_wide.items():
            if date in factor_data.index:
                factor_values[factor_name] = factor_data.loc[date]
        
        if not factor_values:
            return pd.DataFrame(columns=['rb_date', 'order_book_id', 'group'])
        
        # 合成因子值（使用当前权重）
        if hasattr(self, 'current_weights'):
            combined_factor = pd.Series(0.0, index=self.returns_wide.columns)
            for factor_name, factor_data in factor_values.items():
                if factor_name in self.current_weights:
                    combined_factor += factor_data * self.current_weights[factor_name]
        else:
            # 如果没有权重，使用第一个因子
            combined_factor = list(factor_values.values())[0]
        
        # 应用股票筛选
        valid_mask = self._filter_stocks_for_buy(date)
        valid_stocks = valid_mask[valid_mask].index
        
        if len(valid_stocks) < n_groups:
            return pd.DataFrame(columns=['rb_date', 'order_book_id', 'group'])
        
        # 获取有效股票的因子值
        valid_factor = combined_factor[valid_stocks].dropna()
        
        if len(valid_factor) < n_groups:
            return pd.DataFrame(columns=['rb_date', 'order_book_id', 'group'])
        
        # 按因子值排序分组
        ranks = valid_factor.rank(method='first', ascending=True)
        group_size = max(1, len(ranks) // n_groups)
        groups = ((ranks - 1) // group_size).clip(upper=n_groups - 1).astype(int)
        
        out = pd.DataFrame({
            'rb_date': date,
            'order_book_id': valid_factor.index,
            'group': groups.values
        })
        return out

    def compute_positions_and_returns(self, n_groups: int = 10) -> dict:
        """
        计算持仓和收益：
        - 每个调仓周期内部，调仓日等权买入
        - 调仓周期内每日个股市值 = 初始权重 × (1+return).cumprod()
        - 分组收益率 = 组内股票收益率的加权平均
        """
        print(f"  [交易层] 构建调仓分组与每日持仓 (n_groups={n_groups}, 调仓周期={self.rebalance_period}) ...")
        
        # 调仓日分组
        group_labels_list = []
        for date in self.rebalance_dates:
            if date in self.returns_wide.index:
                res = self._make_groups_on_rebalance_day(date, n_groups)
                if len(res) > 0:
                    group_labels_list.append(res)
        
        if not group_labels_list:
            raise ValueError("没有生成任何分组数据")
        
        group_labels_rebalance = pd.concat(group_labels_list, ignore_index=True)
        
        # 每个交易日映射到最近调仓日
        all_dates = sorted(self.returns_wide.index)
        dates_df = pd.DataFrame({'date': all_dates})
        rb_df = pd.DataFrame({'rb_date': pd.to_datetime(self.rebalance_dates)})
        map_df = pd.merge_asof(dates_df, rb_df, left_on='date', right_on='rb_date', direction='backward')
        positions_daily = map_df.merge(group_labels_rebalance, how='left', on='rb_date')
        positions_daily = positions_daily[['date', 'rb_date', 'order_book_id', 'group']]
        
        # 合并未来收益
        ret_long = self.returns_wide.stack().rename('ret').reset_index()
        pr = positions_daily.merge(ret_long, on=['date', 'order_book_id'], how='left')
        pr['group'] = pr['group'].astype(int)
        
        # 确保数据对齐：只保留有分组信息的记录
        pr = pr.dropna(subset=['group'])
        
        # 计算初始权重（基于调仓日）
        counts = pr.groupby(['rb_date','group'])['order_book_id'].nunique().rename('n').reset_index()
        pr = pr.merge(counts, on=['rb_date','group'], how='left')
        pr['start_weight'] = 1.0 / pr['n']
        
        # 计算累计收益
        pr['gross'] = pr['ret'] + 1.0
        
        # 处理nan值：将nan替换为1（停牌日收益为0）
        pr['gross'] = pr['gross'].fillna(1.0)
        
        # 计算累计收益
        pr['stock_cum'] = pr.groupby(['rb_date','group','order_book_id'])['gross'].cumprod()
        
        # 计算持仓权重：初始权重 × 累计收益
        pr['holding_weight'] = np.where(
            pr['rb_date'] == pr['date'],
            pr['start_weight'],
            pr['start_weight'] * pr['stock_cum']
        )
        
        # 对每个日期每个分组内的持仓权重进行归一化，使其和为1
        pr['holding_weight'] = pr.groupby(['date','group'])['holding_weight'].transform(lambda x: x / x.sum() if x.sum() != 0 else 0)
        
        # 向量化计算分组每日收益率
        group_returns = pr.groupby(['date', 'group']).apply(
            lambda x: np.average(x['ret'], weights=x['holding_weight'])).reset_index()
        group_returns.columns = ['date', 'group', 'group_return']
        
        # 将组收益率合并回原始数据框
        pr = pr.merge(group_returns, on=['date', 'group'], how='left')
        
        # 先去重，只保留每个（date, group）组合的第一条记录，避免pivot时报错
        pr_nodup = pr.drop_duplicates(subset=['date', 'group'], keep='first')
        group_daily_returns = pr_nodup.pivot(index='date', columns='group', values='group_return').fillna(0.0)
        
        # 按列排序确保分组顺序一致
        group_daily_returns = group_daily_returns.reindex(sorted(group_daily_returns.columns), axis=1)
        
        # 计算分组累计净值
        group_cum_nav = (group_daily_returns+1).cumprod()
        
        # 确保所有日期都有数据，缺失值用前值填充
        group_daily_returns = group_daily_returns.reindex(all_dates, method='ffill').fillna(0.0)
        group_cum_nav = group_cum_nav.reindex(all_dates, method='ffill')
        
        return {
            'group_daily_returns': group_daily_returns,
            'group_cum_nav': group_cum_nav,
            'positions_daily': positions_daily,
            'rebalance_info': {
                'rebalance_dates': self.rebalance_dates,
                'group_labels_rebalance': group_labels_rebalance
            }
        }

    # ---- 日度横截面系数 / IC ----
    def _cs_univariate_beta(self, X: pd.Series, y: pd.Series) -> float:
        """计算横截面一元回归系数"""
        # 确保输入数据对齐
        common_idx = X.index.intersection(y.index)
        if len(common_idx) < 10:
            return np.nan
            
        x = X.loc[common_idx]
        y_vals = y.loc[common_idx]
        
        # 去除空值
        valid_mask = x.notna() & y_vals.notna()
        if valid_mask.sum() < 10:
            return np.nan
            
        x_clean = x[valid_mask]
        y_clean = y_vals[valid_mask]
        
        # 检查数据是否有效
        if len(x_clean) < 10:
            return np.nan
            
        # 标准化x
        x_mean = x_clean.mean()
        x_std = x_clean.std(ddof=0)
        if x_std == 0:
            return np.nan
            
        x_scaled = (x_clean - x_mean) / x_std
        
        try:
            model = LinearRegression()
            model.fit(x_scaled.values.reshape(-1, 1), y_clean.values)
            return float(model.coef_[0])
        except Exception:
            return np.nan

    def _cs_multivariate_beta(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """计算横截面多元回归系数"""
        # 确保输入数据对齐
        common_idx = X.index.intersection(y.index)
        if len(common_idx) < X.shape[1] + 5:
            return np.array([np.nan] * X.shape[1])
            
        X_subset = X.loc[common_idx]
        y_vals = y.loc[common_idx]
        
        # 去除空值
        valid_mask = X_subset.notna().all(axis=1) & y_vals.notna()
        if valid_mask.sum() < X.shape[1] + 5:
            return np.array([np.nan] * X.shape[1])
            
        X_clean = X_subset[valid_mask]
        y_clean = y_vals[valid_mask]
        
        # 检查数据是否有效
        if len(X_clean) < X.shape[1] + 5:
            return np.array([np.nan] * X.shape[1])
            
        Xv = X_clean.values
        yv = y_clean.values
        
        try:
            # 标准化X
            Xv_scaled = StandardScaler().fit_transform(Xv)
            model = LinearRegression()
            model.fit(Xv_scaled, yv)
            return model.coef_.astype(float)
        except Exception:
            return np.array([np.nan] * X.shape[1])

    def _cs_rank_ic(self, X: pd.Series, y: pd.Series) -> float:
        """计算横截面Rank IC (Spearman相关系数)"""
        # 确保输入数据对齐
        common_idx = X.index.intersection(y.index)
        if len(common_idx) < 10:
            return np.nan
            
        x = X.loc[common_idx]
        y_vals = y.loc[common_idx]
        
        # 去除空值
        valid_mask = x.notna() & y_vals.notna()
        if valid_mask.sum() < 10:
            return np.nan
            
        x_clean = x[valid_mask]
        y_clean = y_vals[valid_mask]
        
        # 检查数据是否有效
        if len(x_clean) < 10:
            return np.nan
            
        # 检查是否为常量
        if x_clean.std(ddof=0) == 0 or y_clean.std(ddof=0) == 0:
            return np.nan
            
        try:
            # 使用rank方法计算Spearman相关系数
            x_rank = x_clean.rank(method='average')
            y_rank = y_clean.rank(method='average')
            
            # 计算rank相关系数
            n = len(x_rank)
            if n < 2:
                return np.nan
                
            # 使用scipy的spearmanr计算
            corr, _ = stats.spearmanr(x_rank, y_rank)
            
            # 调试信息（可选，生产环境可以注释掉）
            if np.isnan(corr):
                print(f"Rank IC计算警告: 相关系数为NaN, 数据长度={n}, x_std={x_clean.std():.6f}, y_std={y_clean.std():.6f}")
            
            return float(corr) if not np.isnan(corr) else np.nan
            
        except Exception as e:
            print(f"Rank IC计算异常: {str(e)}")
            return np.nan

    # ---- 权重时间序列（窗口N的滚动均值） ----
    def _compute_weight_history(self, method: str, N: int) -> pd.DataFrame:
        try:
            dates = self.returns_wide.index
            stocks = self.returns_wide.columns



            # 未来收益（与 returns_wide 同索引列）
            y = self.returns_wide

            # 准备 X: dict[name -> zscored factor]
            Xz = {k: _zscore_rowwise(v) for k, v in self.factors_wide.items()}

            # 日度原始"指标"
            raw = []  # list of Series(index=factor_names) per date

            if method == 'univariate':
                for dt in tqdm(dates, desc='univariate betas'):
                    try:
                        yy = y.loc[dt]
                        vals = {}
                        for fname in self.factor_names:
                            if fname in Xz and dt in Xz[fname].index:
                                vals[fname] = self._cs_univariate_beta(Xz[fname].loc[dt], yy)
                            else:
                                vals[fname] = np.nan
                        raw.append(pd.Series(vals))
                    except Exception as e:
                        raw.append(pd.Series([np.nan] * len(self.factor_names), index=self.factor_names))

            elif method == 'multivariate':
                for dt in tqdm(dates, desc='multivariate betas'):
                    try:
                        yy = y.loc[dt]
                        Xmat = pd.DataFrame({f: Xz[f].loc[dt] for f in self.factor_names if f in Xz and dt in Xz[f].index})
                        if not Xmat.empty and len(Xmat.columns) > 0:
                            betas = self._cs_multivariate_beta(Xmat, yy)
                            # 确保返回的betas长度与factor_names一致
                            if len(betas) == len(self.factor_names):
                                raw.append(pd.Series(betas, index=self.factor_names))
                            else:
                                raw.append(pd.Series([np.nan] * len(self.factor_names), index=self.factor_names))
                        else:
                            raw.append(pd.Series([np.nan] * len(self.factor_names), index=self.factor_names))
                    except Exception as e:
                        raw.append(pd.Series([np.nan] * len(self.factor_names), index=self.factor_names))

            elif method == 'rank_ic':
                for dt in tqdm(dates, desc='rank IC'):
                    try:
                        yy = y.loc[dt]
                        vals = {}
                        for fname in self.factor_names:
                            if fname in Xz and dt in Xz[fname].index:
                                vals[fname] = self._cs_rank_ic(Xz[fname].loc[dt], yy)
                            else:
                                vals[fname] = np.nan
                        raw.append(pd.Series(vals))
                    except Exception as e:
                        raw.append(pd.Series([np.nan] * len(self.factor_names), index=self.factor_names))
            else:
                raise ValueError("method 必须是 {'univariate','multivariate','rank_ic'} 之一")

            if not raw:
                raise ValueError("没有生成任何权重数据")

            raw_df = pd.DataFrame(raw, index=dates)  # (date x factor)
            
            # 滚动均值 => 权重历史；再做 L1 归一化
            weight_hist = raw_df.rolling(N, min_periods=max(5, N//3)).mean().apply(_l1_normalize, axis=1)
            return weight_hist.fillna(0.0)
            
        except Exception as e:
            raise

    # ---- 生成合成因子与评估 ----
    def build(self, method: str, N: int) -> ComboResult:
        try:
            weight_hist = self._compute_weight_history(method, N)  # (date x factor)
            
            # 合成因子：sum_t( w_t[f] * Z(X_f,t) )
            Z = {k: _zscore_rowwise(v) for k, v in self.factors_wide.items()}
            
            # 累加
            combined = pd.DataFrame(0.0, index=self.returns_wide.index, columns=self.returns_wide.columns)
            
            for fname in self.factor_names:
                if fname in Z and fname in weight_hist.columns:
                    w = weight_hist[fname].reindex(combined.index).fillna(0.0)
                    # 每日权重与当日横截面相乘
                    combined += Z[fname].multiply(w, axis=0)
            


            # 评估（未来收益采用 rebalance 期）
            result = self._analyze(combined)
            return ComboResult(
                combined_factor=combined,
                weight_history=weight_hist,
                ic_series=result['ic_series'],
                long_only_returns=result['long_only_returns'],
                group_returns=result['group_returns'],
                summary={k: result[k] for k in ['ic_mean','ic_std','icir','ic_positive_ratio','annual_return','annual_volatility','sharpe_ratio']}
            )
            
        except Exception as e:
            raise

    # ---- 评估 ----
    def _analyze(self, factor: pd.DataFrame) -> dict:
        # 长表 - 修复列名问题
        fac_long = factor.stack().rename('factor_value').reset_index()
        ret_long = self.returns_wide.stack().rename('ret').reset_index()
        
        # 确保列名正确
        fac_long.columns = ['date', 'order_book_id', 'factor_value']
        ret_long.columns = ['date', 'order_book_id', 'ret']
        
        df = pd.merge(fac_long, ret_long, on=['date','order_book_id'], how='inner')
        # future return 已在 returns_wide 中
        df = df.dropna()

        # IC（Spearman）
        ic_series = pd.Series(index=df['date'].unique(), dtype=float)
        for date in ic_series.index:
            date_data = df[df['date'] == date]
            if len(date_data) >= 10:
                x = date_data['factor_value']
                y = date_data['ret']
                
                # 检查输入数组是否为常量
                x_vals = x.values
                y_vals = y.values
                
                # 如果任一数组是常量（标准差为0），返回NaN
                if np.std(x_vals, ddof=0) == 0 or np.std(y_vals, ddof=0) == 0:
                    ic_series[date] = np.nan
                    continue
                    
                try:
                    ic_series[date] = float(stats.spearmanr(x, y)[0])
                except Exception:
                    ic_series[date] = np.nan
            else:
                ic_series[date] = np.nan

        ic_mean = float(ic_series.mean()) if len(ic_series) else np.nan
        ic_std = float(ic_series.std()) if len(ic_series) else np.nan
        icir = ic_mean / ic_std if (ic_std and ic_std != 0) else np.nan
        ic_pos = float((ic_series > 0).mean()) if len(ic_series) else np.nan

        # 分组未来收益（十分位，不足时降级）
        df['group'] = np.nan
        for date in df['date'].unique():
            date_data = df[df['date'] == date]
            s = date_data['factor_value']
            n = s.notna().sum()
            
            if n < 10:
                continue
                
            try:
                q = pd.qcut(s.rank(method='first'), q=min(10, max(2, n//20)), labels=False, duplicates='drop')
                df.loc[date_data.index, 'group'] = q
            except Exception:
                try:
                    q = pd.qcut(s.rank(method='first'), q=2, labels=False, duplicates='drop')
                    df.loc[date_data.index, 'group'] = q
                except Exception:
                    continue

        df = df.dropna(subset=['group'])
        group_ret = df.groupby(['date','group'])['ret'].mean().unstack()

        # 纯多 = 最高分组（因子值最大的那一组）
        if group_ret.shape[1] >= 1:
            long_only = group_ret.iloc[:, -1].fillna(0)  # 最高分组（因子值最大的那一组）
        else:
            long_only = pd.Series(dtype=float)

        # 年化指标（按252）
        if len(long_only) > 0:
            total = float((1.0 + long_only.fillna(0)).prod() - 1.0)
            years = len(long_only) / 252.0
            ann = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 else np.nan
            vol = float(long_only.std() * np.sqrt(252.0))
            sharpe = ann / vol if (vol and vol != 0) else np.nan
        else:
            ann = vol = sharpe = np.nan

        return dict(
            ic_series=ic_series,
            ic_mean=ic_mean,
            ic_std=ic_std,
            icir=icir,
            ic_positive_ratio=ic_pos,
            group_returns=group_ret,
            long_only_returns=long_only,  # 改为纯多收益
            annual_return=ann,
            annual_volatility=vol,
            sharpe_ratio=sharpe,
        )

    # ---- 可视化 ----
    def render_report(self, result_map: Dict[str, ComboResult], html_path: str) -> None:
        page = Page(layout=Page.SimplePageLayout)

        # 1) 权重热力图（拼接所有方法）
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
                    range_color=["#FF0000", "#FFFFFF", "#0000FF"]  # 红白蓝从大到小
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
            tooltip_opts=opts.TooltipOpts(
                trigger="axis",
                axis_pointer_type="cross",
                formatter=opts.TooltipOpts(
                    formatter="{b}<br/>{a}: {c:.4f}"
                )
            )
        )
        page.add(line_ic)

        # 3) 纯多累计收益（log10对数形式）
        line_ls = Line()
        idx_ref = None
        for m, res in result_map.items():
            s = (1 + res.long_only_returns.fillna(0)).cumprod()
            if idx_ref is None:
                idx_ref = s.index
                line_ls.add_xaxis([d.strftime('%Y-%m-%d') for d in idx_ref])
            s = s.reindex(idx_ref).ffill().fillna(1)
            # 转换为log10对数形式
            s_log = np.log10(s)
            line_ls.add_yaxis(m, s_log.values.tolist(), is_smooth=True, symbol_size=0, label_opts=opts.LabelOpts(is_show=False))
        line_ls.set_global_opts(
            title_opts=opts.TitleOpts(title="最高分组累计收益对比 (因子值最大的10%, log10)"),
            datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45)),
            yaxis_opts=opts.AxisOpts(name="累计收益 (log10)"),
            tooltip_opts=opts.TooltipOpts(
                trigger="axis",
                axis_pointer_type="cross",
                formatter=opts.TooltipOpts(
                    formatter="{b}<br/>{a}: {c:.4f}"
                )
            )
        )
        page.add(line_ls)

        # 4) 选择夏普最高的方法，画分组累计收益柱状
        best_name = None
        best_sharpe = -np.inf
        for m, res in result_map.items():
            sh = res.summary.get('sharpe_ratio', np.nan)
            if pd.notna(sh) and sh > best_sharpe:
                best_sharpe = sh
                best_name = m
        if best_name is not None:
            # 计算各分组的累计收益
            grp_returns = result_map[best_name].group_returns
            grp_cumulative = (1 + grp_returns.fillna(0)).cumprod()
            
            # 取最后一期的累计值
            grp_final = grp_cumulative.iloc[-1] if len(grp_cumulative) > 0 else pd.Series()
            
            if len(grp_final) > 0:
                x = [f"G{int(i)+1}" for i in grp_final.index.tolist()]
                y = [float(v) if pd.notna(v) else 1.0 for v in grp_final.values]
                
                # 计算收益率百分比
                y_percent = [(v - 1.0) * 100 if v > 0 else 0.0 for v in y]
                
                # 创建悬停提示数据
                tooltip_data = []
                for i, (val, pct) in enumerate(zip(y, y_percent)):
                    tooltip_data.append(f"分组{i+1}<br/>累计净值: {val:.4f}<br/>收益率: {pct:.2f}%")
                
                bar = (
                    Bar()
                    .add_xaxis(x)
                    .add_yaxis(
                        "累计收益", 
                        y, 
                        label_opts=opts.LabelOpts(is_show=False),
                        tooltip_opts=opts.TooltipOpts(
                            formatter=opts.TooltipOpts(
                                formatter="{b}<br/>{a}: {c:.4f}<br/>收益率: {c_percent:.2f}%"
                            )
                        )
                    )
                    .set_global_opts(
                        title_opts=opts.TitleOpts(title=f"最佳方法: {best_name} 分组累计收益"),
                        xaxis_opts=opts.AxisOpts(name="分组"),
                        yaxis_opts=opts.AxisOpts(name="累计收益"),
                        tooltip_opts=opts.TooltipOpts(
                            trigger="axis",
                            axis_pointer_type="shadow",
                            formatter=opts.TooltipOpts(
                                formatter=lambda params: tooltip_data[params[0].dataIndex] if params and len(params) > 0 else ""
                            )
                        )
                    )
                )
                page.add(bar)

        # 5) 性能表格
        rows = []
        for m, res in result_map.items():
            sm = res.summary
            rows.append([
                m,
                f"{sm.get('ic_mean', np.nan):.4f}" if pd.notna(sm.get('ic_mean', np.nan)) else 'N/A',
                f"{sm.get('icir', np.nan):.4f}" if pd.notna(sm.get('icir', np.nan)) else 'N/A',
                f"{sm.get('ic_positive_ratio', np.nan):.4f}" if pd.notna(sm.get('ic_positive_ratio', np.nan)) else 'N/A',
                f"{sm.get('annual_return', np.nan):.2%}" if pd.notna(sm.get('annual_return', np.nan)) else 'N/A',
                f"{sm.get('sharpe_ratio', np.nan):.2f}" if pd.notna(sm.get('sharpe_ratio', np.nan)) else 'N/A',
            ])
        table = (
            Table()
            .add(headers=["方法", "IC均值", "ICIR", "IC正比例", "年化收益", "夏普"], rows=rows)
        )
        page.add(table)
        
        # 保存HTML文件
        page.render(html_path)