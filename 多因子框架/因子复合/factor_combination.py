# -*- coding: utf-8 -*-
"""
因子复合分析框架
支持三种权重方法：univariate(一元回归)、multivariate(多元回归)、rank_ic(Rank IC)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

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
    """横截面Z-Score标准化"""
    def _z(s: pd.Series) -> pd.Series:
        m, v = s.mean(), s.std(ddof=0)
        return pd.Series(0.0, index=s.index) if not np.isfinite(v) or v == 0 else (s - m) / v
    
    # 检查数据质量
    print(f"    _zscore_rowwise: 输入DataFrame形状: {df.shape}")
    print(f"    _zscore_rowwise: 包含NaN的数量: {df.isna().sum().sum()}")
    print(f"    _zscore_rowwise: 包含无穷大的数量: {np.isinf(df.values).sum()}")
    
    result = df.apply(_z, axis=1)
    
    # 检查结果质量
    print(f"    _zscore_rowwise: 输出DataFrame形状: {result.shape}")
    print(f"    _zscore_rowwise: 输出包含NaN的数量: {result.isna().sum().sum()}")
    print(f"    _zscore_rowwise: 输出包含无穷大的数量: {np.isinf(result.values).sum()}")
    
    return result


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
        self.enable_stock_filter = bool(enable_stock_filter)
        
        self._process_data()
        self._align()
        self._create_rebalance_dates()

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
            print("开始数据对齐...")
            
            # 计算未来收益
            px = self.prices.sort_values(['order_book_id', 'date'])
            px['date'] = pd.to_datetime(px['date'])
            px['return'] = px.groupby('order_book_id')['close'].pct_change(fill_method=None).shift(-1)
            
            # 筛选列
            filter_cols = ['limit_up_flag', 'limit_down_flag', 'ST', 'suspended']
            available_filter_cols = [col for col in filter_cols if col in px.columns]
            
            ret = px[['date', 'order_book_id', 'return'] + available_filter_cols].dropna(subset=['return'])
            ret_wide = ret.pivot(index='date', columns='order_book_id', values='return')
            
            # 筛选标志
            self.filter_flags = {}
            for col in available_filter_cols:
                flag_wide = ret.pivot(index='date', columns='order_book_id', values=col)
                self.filter_flags[col] = flag_wide
            
            # 数据对齐
            common_dates = ret_wide.index
            common_cols = ret_wide.columns
            
            processed_factors = {}
            for name, f in self.factors_raw.items():
                if len(f.columns) == 3:
                    # 长表转宽表
                    col_names = list(f.columns)
                    date_cols = [col for col in col_names if 'date' in col.lower() or col == 'date']
                    id_cols = [col for col in col_names if 'id' in col.lower() or 'code' in col.lower() or col == 'order_book_id']
                    
                    if date_cols and id_cols:
                        date_col = date_cols[0]
                        id_col = id_cols[0]
                        value_col = [col for col in col_names if col not in [date_col, id_col]][0]
                        
                        f_wide = f.set_index(date_col).pivot(columns=id_col, values=value_col)
                        f_wide.index.name = 'date'
                        processed_factors[name] = f_wide
                    else:
                        # 如果无法识别列，跳过这个因子
                        continue
                else:
                    processed_factors[name] = f
                
                common_dates = common_dates.intersection(processed_factors[name].index)
                common_cols = common_cols.intersection(processed_factors[name].columns)
                
            if len(common_dates) == 0 or len(common_cols) == 0:
                raise ValueError(f"没有共同的日期或股票: 日期数={len(common_dates)}, 股票数={len(common_cols)}")
            
            if len(processed_factors) == 0:
                raise ValueError("没有可用的因子数据")
                
            # 裁剪数据
            self.returns_wide = ret_wide.loc[common_dates, common_cols].sort_index()
            self.factors_wide = {k: v.loc[common_dates, common_cols].sort_index() for k, v in processed_factors.items()}
            self.factor_names = list(self.factors_wide.keys())
            
            for col in self.filter_flags:
                self.filter_flags[col] = self.filter_flags[col].loc[common_dates, common_cols].sort_index()
            
            print(f"数据对齐完成: {len(common_dates)} 个交易日, {len(common_cols)} 只股票, {len(self.factor_names)} 个因子")
            
        except Exception as e:
            print(f"数据对齐失败: {str(e)}")
            raise

    def _create_rebalance_dates(self) -> None:
        """创建调仓日期列表"""
        all_dates = sorted(self.returns_wide.index.tolist())
        self.rebalance_dates = all_dates[::self.rebalance_period] if self.rebalance_period > 1 else all_dates
        print(f"创建调仓日期列表: {len(self.rebalance_dates)} 个调仓日 (调仓周期: {self.rebalance_period})")

    def _filter_stocks_for_buy(self, date: pd.Timestamp) -> pd.Series:
        """股票筛选"""
        if not self.enable_stock_filter:
            return pd.Series(True, index=self.returns_wide.columns)
        
        mask = pd.Series(True, index=self.returns_wide.columns)
        
        for col, flag_data in self.filter_flags.items():
            if date in flag_data.index:
                mask &= ~flag_data.loc[date].astype(bool)
            
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
        
        # 日期映射
        all_dates = sorted(self.returns_wide.index)
        dates_df = pd.DataFrame({'date': all_dates})
        rb_df = pd.DataFrame({'rb_date': pd.to_datetime(self.rebalance_dates)})
        map_df = pd.merge_asof(dates_df, rb_df, left_on='date', right_on='rb_date', direction='backward')
        positions_daily = map_df.merge(group_labels_rebalance, how='left', on='rb_date')
        positions_daily = positions_daily[['date', 'rb_date', 'order_book_id', 'group']]
        
        # 合并收益
        ret_long = self.returns_wide.stack().rename('ret').reset_index()
        # 确保列名正确
        ret_long.columns = ['date', 'order_book_id', 'ret']
        pr = positions_daily.merge(ret_long, on=['date', 'order_book_id'], how='left')
        
        # 确保group列是有效的整数，过滤掉无效数据
        pr = pr.dropna(subset=['group']).copy()  # 创建副本避免SettingWithCopyWarning
        pr['group'] = pr['group'].astype(int)
        
        # 计算权重
        counts = pr.groupby(['rb_date','group'])['order_book_id'].nunique().rename('n').reset_index()
        pr = pr.merge(counts, on=['rb_date','group'], how='left')
        pr['start_weight'] = 1.0 / pr['n']
        
        # 累计收益
        pr['gross'] = (pr['ret'] + 1.0).fillna(1.0)
        pr['stock_cum'] = pr.groupby(['rb_date','group','order_book_id'])['gross'].cumprod()
        pr['holding_weight'] = np.where(
            pr['rb_date'] == pr['date'],
            pr['start_weight'],
            pr['start_weight'] * pr['stock_cum']
        )
        
        # 权重归一化
        pr['holding_weight'] = pr.groupby(['date','group'])['holding_weight'].transform(lambda x: x / x.sum() if x.sum() != 0 else 0)
        
        # 分组收益
        group_returns = []
        for (date, group), group_data in pr.groupby(['date', 'group']):
            if len(group_data) > 0:
                avg_return = np.average(group_data['ret'], weights=group_data['holding_weight'])
                group_returns.append({'date': date, 'group': group, 'group_return': avg_return})
        group_returns = pd.DataFrame(group_returns)
        
        pr = pr.merge(group_returns, on=['date', 'group'], how='left')
        pr_nodup = pr.drop_duplicates(subset=['date', 'group'], keep='first')
        group_daily_returns = pr_nodup.pivot(index='date', columns='group', values='group_return').fillna(0.0)
        group_daily_returns = group_daily_returns.reindex(sorted(group_daily_returns.columns), axis=1)
        
        # 累计净值
        group_cum_nav = (group_daily_returns+1).cumprod()
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

    def _compute_weight_history(self, method: str, N: int) -> pd.DataFrame:
        """计算权重历史"""
        try:
            print(f"开始计算权重历史，方法: {method}, 滚动窗口: {N}")
            dates = self.returns_wide.index
            y = self.returns_wide
            Xz = {k: _zscore_rowwise(v) for k, v in self.factors_wide.items()}
            raw = []

            if method == 'univariate':
                print(f"使用univariate方法，处理 {len(dates)} 个交易日...")
                for dt in tqdm(dates, desc='univariate betas'):
                    try:
                        yy = y.loc[dt]
                        vals = {}
                        for fname in self.factor_names:
                            if fname in Xz and dt in Xz[fname].index:
                                factor_vals = Xz[fname].loc[dt]
                                ic_val = self._calculate_ic(factor_vals, yy)
                                vals[fname] = ic_val
                            else:
                                vals[fname] = np.nan
                        raw.append(pd.Series(vals))
                    except Exception as e:
                        print(f"处理日期 {dt} 时出错: {e}")
                        raw.append(pd.Series([np.nan] * len(self.factor_names), index=self.factor_names))
                print(f"univariate方法完成，生成了 {len(raw)} 个权重记录")

            elif method == 'multivariate':
                print(f"使用multivariate方法，处理 {len(dates)} 个交易日...")
                for dt in tqdm(dates, desc='multivariate betas'):
                    try:
                        yy = y.loc[dt]
                        Xmat = pd.DataFrame({f: Xz[f].loc[dt] for f in self.factor_names if f in Xz and dt in Xz[f].index})
                        if not Xmat.empty and len(Xmat.columns) > 0:
                            betas = self._calculate_multivariate_beta(Xmat, yy)
                            if len(betas) == len(self.factor_names):
                                raw.append(pd.Series(betas, index=self.factor_names))
                            else:
                                raw.append(pd.Series([np.nan] * len(self.factor_names), index=self.factor_names))
                        else:
                            raw.append(pd.Series([np.nan] * len(self.factor_names), index=self.factor_names))
                    except Exception as e:
                        print(f"处理日期 {dt} 时出错: {e}")
                        raw.append(pd.Series([np.nan] * len(self.factor_names), index=self.factor_names))
                print(f"multivariate方法完成，生成了 {len(raw)} 个权重记录")

            elif method == 'rank_ic':
                print(f"使用rank_ic方法，处理 {len(dates)} 个交易日...")
                for dt in tqdm(dates, desc='rank IC'):
                    try:
                        yy = y.loc[dt]
                        vals = {}
                        for fname in self.factor_names:
                            if fname in Xz and dt in Xz[fname].index:
                                factor_vals = Xz[fname].loc[dt]
                                ic_val = self._calculate_ic(factor_vals, yy)
                                vals[fname] = ic_val
                            else:
                                vals[fname] = np.nan
                        raw.append(pd.Series(vals))
                    except Exception as e:
                        print(f"处理日期 {dt} 时出错: {e}")
                        raw.append(pd.Series([np.nan] * len(self.factor_names), index=self.factor_names))
                print(f"rank_ic方法完成，生成了 {len(raw)} 个权重记录")
            else:
                raise ValueError("method 必须是 {'univariate','multivariate','rank_ic'} 之一")

            if not raw:
                raise ValueError("没有生成任何权重数据")

            print(f"开始创建权重DataFrame...")
            raw_df = pd.DataFrame(raw, index=dates)
            print(f"权重DataFrame创建完成，形状: {raw_df.shape}")
            
            print(f"开始计算滚动平均，窗口大小: {N}")
            min_periods = min(5, N)  # 确保min_periods <= N
            weight_hist = raw_df.rolling(N, min_periods=min_periods).mean().apply(_l1_normalize, axis=1)
            print(f"滚动平均计算完成，权重历史形状: {weight_hist.shape}")
            
            result = weight_hist.fillna(0.0)
            print(f"权重历史计算完成，最终形状: {result.shape}")
            return result
            
        except Exception as e:
            print(f"计算权重历史时发生错误: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _calculate_ic(self, X: pd.Series, y: pd.Series) -> float:
        """计算IC"""
        if hasattr(self, 'filter_flags') and self.enable_stock_filter:
            current_date = X.index[0] if len(X.index) > 0 else None
            if current_date and current_date in self.returns_wide.index:
                valid_mask = self._filter_stocks_for_buy(current_date)
                X = X[valid_mask]
                y = y[valid_mask]
        
        df = pd.concat([X, y], axis=1).dropna()
        if len(df) < 10:
            return np.nan
            
        x, y_vals = df.iloc[:, 0], df.iloc[:, 1]
        
        if x.std(ddof=0) == 0 or y_vals.std(ddof=0) == 0:
            return np.nan
            
        try:
            return float(stats.spearmanr(x, y_vals)[0])
        except Exception:
            return np.nan

    def _calculate_multivariate_beta(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """计算多元回归系数"""
        if hasattr(self, 'filter_flags') and self.enable_stock_filter:
            current_date = X.index[0] if len(X.index) > 0 else None
            if current_date and current_date in self.returns_wide.index:
                valid_mask = self._filter_stocks_for_buy(current_date)
                X = X[valid_mask]
                y = y[valid_mask]
        
        df = pd.concat([X, y], axis=1).dropna()
        if len(df) < X.shape[1] + 5:
            return np.array([np.nan] * X.shape[1])
            
        Xv, yv = df.iloc[:, :X.shape[1]].values, df.iloc[:, -1].values
        
        try:
            Xv_scaled = StandardScaler().fit_transform(Xv)
            model = LinearRegression()
            model.fit(Xv_scaled, yv)
            return model.coef_.astype(float)
        except Exception:
            return np.array([np.nan] * X.shape[1])

    def build(self, method: str, N: int, n_groups: int = 10) -> ComboResult:
        """构建合成因子并评估"""
        try:
            print(f"开始构建合成因子，方法: {method}, 滚动窗口: {N}")
            
            # 计算权重历史
            print("步骤1: 计算权重历史...")
            weight_hist = self._compute_weight_history(method, N)
            self.current_weights = weight_hist.iloc[-1] if len(weight_hist) > 0 else pd.Series(0.0, index=self.factor_names)
            print(f"权重历史计算完成，形状: {weight_hist.shape}")
            
            # 合成因子
            print("步骤2: 合成因子...")
            print("  2.1: 开始Z-Score标准化...")
            try:
                Z = {}
                for k, v in self.factors_wide.items():
                    print(f"    标准化因子 {k}，形状: {v.shape}")
                    Z[k] = _zscore_rowwise(v)
                    print(f"    因子 {k} 标准化完成")
                print("  所有因子标准化完成")
            except Exception as e:
                print(f"  Z-Score标准化失败: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            print("  2.2: 创建合成因子DataFrame...")
            combined = pd.DataFrame(0.0, index=self.returns_wide.index, columns=self.returns_wide.columns)
            print(f"  合成因子DataFrame创建完成，形状: {combined.shape}")
            
            print("  2.3: 计算合成因子...")
            for fname in self.factor_names:
                if fname in Z and fname in weight_hist.columns:
                    print(f"    处理因子 {fname}...")
                    print(f"      权重历史形状: {weight_hist[fname].shape}")
                    print(f"      权重历史索引: {weight_hist[fname].index[:5]}...")
                    print(f"      合成因子索引: {combined.index[:5]}...")
                    
                    try:
                        w = weight_hist[fname].reindex(combined.index).fillna(0.0)
                        print(f"      权重重新索引完成，形状: {w.shape}")
                        print(f"      权重范围: [{w.min():.6f}, {w.max():.6f}]")
                        
                        factor_data = Z[fname]
                        print(f"      因子数据形状: {factor_data.shape}")
                        print(f"      因子数据索引: {factor_data.index[:5]}...")
                        
                        # 检查索引是否匹配
                        if not w.index.equals(factor_data.index):
                            print(f"      警告: 权重和因子数据索引不匹配")
                            print(f"        权重索引长度: {len(w.index)}")
                            print(f"        因子索引长度: {len(factor_data.index)}")
                        
                        print(f"      开始执行乘法运算...")
                        print(f"        权重数据类型: {w.dtype}")
                        print(f"        因子数据类型: {factor_data.dtypes}")
                        print(f"        权重内存使用: {w.memory_usage(deep=True)} bytes")
                        print(f"        因子数据内存使用: {factor_data.memory_usage(deep=True)} bytes")
                        
                        # 直接使用分块处理以避免内存问题
                        print(f"      使用分块处理...")
                        chunk_size = 500  # 减少块大小
                        result = pd.DataFrame(0.0, index=factor_data.index, columns=factor_data.columns)
                        
                        for i in range(0, len(factor_data.columns), chunk_size):
                            end_i = min(i + chunk_size, len(factor_data.columns))
                            chunk_cols = factor_data.columns[i:end_i]
                            chunk_data = factor_data[chunk_cols]
                            chunk_result = chunk_data.multiply(w, axis=0)
                            result[chunk_cols] = chunk_result
                            print(f"        处理列 {i}-{end_i} 完成")
                        
                        print(f"      分块处理完成，结果形状: {result.shape}")
                        
                        print(f"      开始累加到合成因子...")
                        combined += result
                        print(f"      累加到合成因子完成")
                        
                    except Exception as e:
                        print(f"      处理因子 {fname} 时出错: {e}")
                        import traceback
                        traceback.print_exc()
                        raise
                    
                    print(f"    因子 {fname} 处理完成")
            
            self.combined_factor = combined
            print(f"合成因子完成，形状: {combined.shape}")
            
            # 计算收益
            print("步骤3: 计算收益...")
            print("  3.1: 计算持仓和收益...")
            positions_returns = self.compute_positions_and_returns(n_groups=n_groups)
            print("  3.2: 计算IC时间序列...")
            ic_series = self._calculate_ic_series(combined)
            print("  3.3: 计算多空收益...")
            long_short_data = self._calculate_long_short_returns(positions_returns, n_groups)
            
            print("步骤4: 创建结果对象...")
            result = ComboResult(
                combined_factor=combined,
                weight_history=weight_hist,
                ic_series=ic_series,
                long_only_returns=long_short_data['long_only_returns'],
                group_returns=positions_returns['group_daily_returns'],
                summary=long_short_data['summary']
            )
            
            print("合成因子构建完成！")
            return result
            
        except Exception as e:
            print(f"构建合成因子时发生错误: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _calculate_ic_series(self, factor: pd.DataFrame) -> pd.Series:
        """计算IC时间序列"""
        fac_long = factor.stack().rename('factor_value').reset_index()
        ret_long = self.returns_wide.stack().rename('ret').reset_index()
        
        fac_long.columns = ['date', 'order_book_id', 'factor_value']
        ret_long.columns = ['date', 'order_book_id', 'ret']
        
        df = pd.merge(fac_long, ret_long, on=['date','order_book_id'], how='inner').dropna()
        ic_series = pd.Series(index=df['date'].unique(), dtype=float)
        
        for date in ic_series.index:
            date_data = df[df['date'] == date]
            if len(date_data) >= 10:
                x, y = date_data['factor_value'], date_data['ret']
                
                if self.enable_stock_filter and hasattr(self, 'filter_flags'):
                    valid_mask = self._filter_stocks_for_buy(date)
                    if len(valid_mask) > 0:
                        valid_stocks = valid_mask[valid_mask].index
                        if len(valid_stocks) > 0:
                            date_data = date_data[date_data['order_book_id'].isin(valid_stocks)]
                            if len(date_data) >= 10:
                                x, y = date_data['factor_value'], date_data['ret']
                
                x_vals, y_vals = x.values, y.values
                
                if np.std(x_vals, ddof=0) == 0 or np.std(y_vals, ddof=0) == 0:
                    ic_series[date] = np.nan
                    continue
                    
                try:
                    ic_series[date] = float(stats.spearmanr(x, y)[0])
                except Exception:
                    ic_series[date] = np.nan
            else:
                ic_series[date] = np.nan

        return ic_series

    def _calculate_long_short_returns(self, positions_returns: dict, n_groups: int) -> dict:
        """计算多空收益和统计指标"""
        grp_ret = positions_returns['group_daily_returns']
        
        # 选择最高分组
        if grp_ret.shape[1] >= 1:
            long_only = grp_ret.iloc[:, -1].fillna(0)
        else:
            long_only = pd.Series(dtype=float)

        # 年化指标
        if len(long_only) > 0:
            total = float((1.0 + long_only.fillna(0)).prod() - 1.0)
            years = len(long_only) / 252.0
            ann = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 else np.nan
            vol = float(long_only.std() * np.sqrt(252.0))
            sharpe = ann / vol if (vol and vol != 0) else np.nan
        else:
            ann = vol = sharpe = np.nan

        # IC统计
        ic_series = self._calculate_ic_series(self.combined_factor) if hasattr(self, 'combined_factor') else pd.Series(dtype=float)
        ic_mean = float(ic_series.mean()) if len(ic_series) else np.nan
        ic_std = float(ic_series.std()) if len(ic_series) else np.nan
        icir = ic_mean / ic_std if (ic_std and ic_std != 0) else np.nan
        ic_pos = float((ic_series > 0).mean()) if len(ic_series) else np.nan

        return {
            'long_only_returns': long_only,
            'summary': {
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'icir': icir,
                'ic_positive_ratio': ic_pos,
                'annual_return': ann,
                'annual_volatility': vol,
                'sharpe_ratio': sharpe,
            }
        }

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

        # 3) 纯多累计收益（log10）
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

        # 4) 分组累计收益柱状图
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
        
        page.render(html_path)