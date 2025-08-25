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
    ls_returns: pd.Series                   # 多空组合日度收益
    group_returns: pd.DataFrame             # 各分组日度未来收益
    summary: dict                           # 统计指标


# ------------------------- 主类 -------------------------

class FactorCombiner:
    def __init__(
        self,
        factors: Dict[str, pd.DataFrame],
        prices: pd.DataFrame,
        rebalance_period: int = 1,
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
        rebalance_period : int
            未来收益偏移期（>=1）
        """
        self.factors_raw = {k: v.copy() for k, v in factors.items()}
        self.prices = prices.copy()
        self.rebalance = int(rebalance_period)

        # 检测并处理数据格式
        self._detect_and_convert_data_format()
        
        # 确保价格数据的日期列是datetime格式
        self.prices['date'] = pd.to_datetime(self.prices['date'])

        self.factor_names = list(self.factors_raw.keys())
        self._align()

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
            print(f"原始因子数据: {list(self.factors_raw.keys())}")
            
            # 1. 计算未来收益
            px = self.prices.sort_values(['order_book_id', 'date'])
            px['return'] = px.groupby('order_book_id')['close'].pct_change(fill_method=None).shift(-self.rebalance)
            ret = px[['date', 'order_book_id', 'return']].dropna()
            print(f"未来收益数据形状: {ret.shape}")
            
            # 2. 转换为宽表格式
            ret_wide = ret.pivot(index='date', columns='order_book_id', values='return')
            print(f"收益宽表形状: {ret_wide.shape}")
            
            # 3. 数据对齐（求所有矩阵的共同交集）
            common_dates = ret_wide.index
            common_cols = ret_wide.columns
            
            # 处理因子数据，确保都是宽表格式
            processed_factors = {}
            for name, f in self.factors_raw.items():
                print(f"处理因子 {name}: 原始形状 {f.shape}")
                
                # 如果因子数据是长表格式，转换为宽表
                if len(f.columns) == 3:
                    col_names = list(f.columns)
                    date_col = [col for col in col_names if 'date' in col.lower() or col == 'date'][0]
                    id_col = [col for col in col_names if 'id' in col.lower() or 'code' in col.lower() or col == 'order_book_id'][0]
                    value_col = [col for col in col_names if col not in [date_col, id_col]][0]
                    
                    print(f"  长表格式: date_col={date_col}, id_col={id_col}, value_col={value_col}")
                    
                    # 转换为宽表
                    f_wide = f.set_index(date_col).pivot(columns=id_col, values=value_col)
                    f_wide.index.name = 'date'
                    processed_factors[name] = f_wide
                    print(f"  转换为宽表后形状: {f_wide.shape}")
                else:
                    # 已经是宽表格式
                    processed_factors[name] = f
                    print(f"  已经是宽表格式: {f.shape}")
                
                # 更新共同交集
                common_dates = common_dates.intersection(processed_factors[name].index)
                common_cols = common_cols.intersection(processed_factors[name].columns)
                print(f"  当前共同交集: 日期={len(common_dates)}, 股票={len(common_cols)}")
                
            if len(common_dates) == 0 or len(common_cols) == 0:
                raise ValueError(f"没有共同的日期或股票: 日期数={len(common_dates)}, 股票数={len(common_cols)}")
                
            # 4. 裁剪数据
            self.returns_wide = ret_wide.loc[common_dates, common_cols].sort_index()
            self.factors_wide = {k: v.loc[common_dates, common_cols].sort_index() for k, v in processed_factors.items()}
            self.factor_names = list(self.factors_wide.keys())
            
            # 5. 输出最终数据信息
            print(f"数据对齐完成: {len(common_dates)} 个交易日, {len(common_cols)} 只股票, {len(self.factor_names)} 个因子")
            print(f"最终收益数据形状: {self.returns_wide.shape}")
            for name, f in self.factors_wide.items():
                print(f"  因子 {name} 最终形状: {f.shape}")
            
        except Exception as e:
            print(f"数据对齐失败: {str(e)}")
            raise

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
                ls_returns=result['long_short_returns'],
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

        # 多空 = 最高组 - 最低组
        if group_ret.shape[1] >= 2:
            ls = group_ret.iloc[:, -1].fillna(0) - group_ret.iloc[:, 0].fillna(0)
        else:
            ls = pd.Series(dtype=float)

        # 年化指标（按252）
        if len(ls) > 0:
            total = float((1.0 + ls.fillna(0)).prod() - 1.0)
            years = len(ls) / 252.0
            ann = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 else np.nan
            vol = float(ls.std() * np.sqrt(252.0))
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
            long_short_returns=ls,
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
                visualmap_opts=opts.VisualMapOpts(min_=float(np.nanmin(W.values)), max_=float(np.nanmax(W.values))),
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
        )
        page.add(line_ic)

        # 3) 多空累计收益
        line_ls = Line()
        idx_ref = None
        for m, res in result_map.items():
            s = (1 + res.ls_returns.fillna(0)).cumprod()
            if idx_ref is None:
                idx_ref = s.index
                line_ls.add_xaxis([d.strftime('%Y-%m-%d') for d in idx_ref])
            s = s.reindex(idx_ref).ffill().fillna(1)
            line_ls.add_yaxis(m, s.values.tolist(), is_smooth=True, symbol_size=0, label_opts=opts.LabelOpts(is_show=False))
        line_ls.set_global_opts(
            title_opts=opts.TitleOpts(title="多空累计收益对比"),
            datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45)),
        )
        page.add(line_ls)

        # 4) 选择夏普最高的方法，画分组平均收益柱状
        best_name = None
        best_sharpe = -np.inf
        for m, res in result_map.items():
            sh = res.summary.get('sharpe_ratio', np.nan)
            if pd.notna(sh) and sh > best_sharpe:
                best_sharpe = sh
                best_name = m
        if best_name is not None:
            grp_mean = result_map[best_name].group_returns.mean()
            x = [f"G{int(i)+1}" for i in grp_mean.index.tolist()]
            y = [float(v) if pd.notna(v) else 0.0 for v in grp_mean.values]
            bar = (
                Bar()
                .add_xaxis(x)
                .add_yaxis("平均未来收益", y)
                .set_global_opts(
                    title_opts=opts.TitleOpts(title=f"最佳方法: {best_name} 分组收益"),
                    xaxis_opts=opts.AxisOpts(name="分组"),
                    yaxis_opts=opts.AxisOpts(name="收益"),
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