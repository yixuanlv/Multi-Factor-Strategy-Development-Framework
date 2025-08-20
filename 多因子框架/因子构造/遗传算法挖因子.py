# -*- coding: utf-8 -*-
"""
简化版遗传算法挖因子系统
基于单一dataframe输入，自动进行基本面因子变换和市值中性化
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal, Optional, Dict, List, Tuple
from sklearn.linear_model import LinearRegression
import random
import warnings
warnings.filterwarnings('ignore')

# -------- 1) 因子格式定义 ----------
@dataclass
class FactorFormat:
    """因子格式 Ne[f(y, x, y_lg, x_lg, y_tr, x_tr, y_tr_pd, x_tr_pd, y_tr_fm, x_tr_fm, mode), S]"""
    y: str
    x: Optional[str]
    y_lg: bool
    x_lg: bool
    y_tr: bool
    x_tr: bool
    y_tr_pd: Literal["q", "y"]
    x_tr_pd: Literal["q", "y"]
    y_tr_fm: Literal["diff", "pct", "std"]
    x_tr_fm: Literal["lag", "diff", "pct", "std"]
    mode: Literal["A", "B", "C"]

# -------- 2) 指标配置 ----------
@dataclass
class IndicatorConfig:
    name: str
    category: Literal["market", "financial"]
    can_be_y: bool
    can_be_x: bool
    allow_log: bool
    allow_time_transform: bool
    allow_lag: bool
    allow_diff: bool
    allow_pct: bool
    allow_std: bool

# 基本面因子配置
FINANCIAL_FACTORS = [
    'revenue_mrq_0', 'net_profit_mrq_0', 'total_assets_mrq_0',
    'revenue_ttm1_0', 'net_profit_ttm1_0', 'total_assets_ttm_0',
    'fcff_lyr', 'fcff_mrq_0', 'fcff_ttm', 'fcfe_ttm', 'fcfe_lyr', 'fcfe_mrq_0',
    'equity_prefer_stock', 'equity_preferred_stock', 'equity_ratio_lf', 
    'equity_ratio_lyr', 'equity_ratio_ttm', 'asset_impairment', 
    'asset_impairment_lossTTM', 'assets_depreciation_reserves', 'bad_debt_reserve',
    'cash_ratio_lf', 'cash_ratio_ttm', 'cash_rate_of_salesTTM', 'ebit_ttm',
    'ebit_mrq_0', 'ebit', 'gross_profit', 'gross_profitTTM'
]

# 构建指标注册表
def build_indicator_registry():
    registry = {}
    
    # 市场指标
    market_indicators = ['close', 'volume', 'return']
    for ind in market_indicators:
        registry[ind] = IndicatorConfig(
            ind, "market", True, True, False, True, True, True, True, True
        )
    
    # 基本面因子
    for factor in FINANCIAL_FACTORS:
        registry[factor] = IndicatorConfig(
            factor, "financial", True, True, True, True, True, True, True, False
        )
    
    # 市值因子
    registry['market_cap'] = IndicatorConfig(
        'market_cap', "financial", True, True, True, True, True, True, True, False
    )
    
    return registry

INDICATOR_REGISTRY = build_indicator_registry()

# -------- 3) 数据预处理 ----------
def preprocess_data(df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], pd.Series, pd.Series]:
    """预处理输入的dataframe，提取各指标数据"""
    print("开始数据预处理...")
    
    # 确保索引结构正确
    if 'order_book_id' in df.columns and 'date' in df.columns:
        df = df.set_index(['order_book_id', 'date'])
    
    # 提取各指标数据
    data_dict = {}
    
    # 市场指标
    for col in ['close', 'volume', 'return']:
        if col in df.columns:
            data_dict[col] = df[col]
            print(f"提取 {col}: {len(data_dict[col])} 个数据点")
    
    # 基本面因子
    for factor in FINANCIAL_FACTORS:
        if factor in df.columns:
            data_dict[factor] = df[factor]
            print(f"提取 {factor}: {len(data_dict[factor])} 个数据点")
    
    # 市值数据
    if 'market_cap' in df.columns:
        size = df['market_cap']
        print(f"提取 market_cap: {len(size)} 个数据点")
    else:
        size = pd.Series(dtype=float)
        print("警告: 未找到市值数据")
    
    # 未来收益率
    future_ret = data_dict.get('return', pd.Series(dtype=float))
    
    print(f"数据预处理完成: {len(data_dict)} 个指标")
    return data_dict, size, future_ret

# -------- 4) 因子构建函数 ----------
def apply_transforms(raw: pd.Series, cfg: IndicatorConfig, 
                    use_log: bool, time_tr: bool,
                    tr_pd: Literal["q", "y"], tr_fm: str,
                    lag_n: int = 1) -> pd.Series:
    """根据配置对指标进行变换"""
    s = raw.copy()
    
    # 确保索引结构正确
    if s.index.nlevels != 2:
        return s
    
    if s.index.names[0] != 'date':
        s = s.swaplevel(0, 1).sort_index()
    
    # 对数变换
    if use_log and cfg.allow_log:
        # 对财务指标进行对数变换，处理负值和零值
        if cfg.category == "financial":
            # 对于财务指标，先取绝对值再取对数，保留符号
            s = np.sign(s) * np.log1p(np.abs(s))
        else:
            s = np.log1p(np.abs(s))
    
    # 时间变换
    if time_tr and cfg.allow_time_transform:
        try:
            if tr_fm == "lag" and cfg.allow_lag:
                s = s.groupby(level=0).transform(lambda x: x.shift(lag_n))
            elif tr_fm == "diff" and cfg.allow_diff:
                s = s.groupby(level=0).transform(lambda x: x.shift(lag_n))
                s = s.groupby(level=0).transform(lambda x: x.diff())
            elif tr_fm == "pct" and cfg.allow_pct:
                lagged = s.groupby(level=0).transform(lambda x: x.shift(lag_n))
                s = (s - lagged) / (np.abs(lagged) + 1e-12)
            elif tr_fm == "std" and cfg.allow_std:
                s = s.groupby(level=0).transform(lambda x: x.rolling(4, min_periods=2).std())
        except:
            pass
    
    return s

def build_factor(factor_format: FactorFormat, data: Dict[str, pd.Series], 
                size: pd.Series) -> pd.Series:
    """根据因子格式构建因子"""
    
    y_cfg = INDICATOR_REGISTRY.get(factor_format.y)
    x_cfg = INDICATOR_REGISTRY.get(factor_format.x) if factor_format.x else None
    
    if not y_cfg or factor_format.y not in data:
        raise ValueError(f"指标 {factor_format.y} 不可用")
    
    # 应用变换到y
    y = apply_transforms(data[factor_format.y], y_cfg, 
                        factor_format.y_lg, factor_format.y_tr,
                        factor_format.y_tr_pd, factor_format.y_tr_fm)
    
    if factor_format.mode == "A":
        f = y
    else:
        if not x_cfg or factor_format.x not in data:
            raise ValueError(f"指标 {factor_format.x} 不可用")
        
        x = apply_transforms(data[factor_format.x], x_cfg,
                           factor_format.x_lg, factor_format.x_tr,
                           factor_format.x_tr_pd, factor_format.x_tr_fm)
        
        if factor_format.mode == "B":
            # 模式B：y/x，适合ROE、E/P等比值因子
            denom = x.replace(0, np.nan)
            valid = denom > 0
            f = pd.Series(np.nan, index=y.index)
            f[valid] = y[valid] / (denom[valid] + 1e-12)
            
            if f.isna().mean() > 0.5:
                factor_format.mode = "C"
        
        if factor_format.mode == "C":
            # 模式C：残差模式，适合增速和相对估值因子
            f = pd.Series(index=y.index, dtype=float)
            df = pd.DataFrame({"y": y, "x": x}).dropna()
            
            for dt, sub in df.groupby(level=0):
                if len(sub) < 10:
                    continue
                try:
                    lr = LinearRegression().fit(sub[["x"]], sub["y"])
                    pred = lr.predict(sub[["x"]])
                    resid = sub["y"] - pred
                    f.loc[sub.index] = resid
                except:
                    continue
    
    # Winsorize + 标准化
    def _zscore(g):
        s = g.clip(lower=g.quantile(0.01), upper=g.quantile(0.99))
        return (s - s.mean()) / (s.std(ddof=0) + 1e-12)
    
    f = f.groupby(level=0).transform(_zscore)
    
    # 市值中性化
    return neutralize_to_size(f, size)

def neutralize_to_size(factor: pd.Series, size: pd.Series) -> pd.Series:
    """对因子进行市值中性化"""
    try:
        if len(size) == 0:
            return factor
        
        common_index = factor.index.intersection(size.index)
        if len(common_index) == 0:
            return factor
        
        factor_common = factor.loc[common_index]
        size_common = size.loc[common_index]
        
        # 市值对数变换
        size_log = np.log(size_common.replace(0, np.nan))
        
        df = pd.DataFrame({
            "f": factor_common, 
            "size": size_log
        }).dropna()
        
        if len(df) < 20:
            return factor
        
        out = pd.Series(index=factor.index, dtype=float)
        out.loc[common_index] = factor_common
        
        for dt, sub in df.groupby(level=0):
            if len(sub) < 10:
                continue
            try:
                X = sub[["size"]].values
                y = sub["f"].values
                beta = np.linalg.lstsq(np.c_[np.ones(len(X)), X], y, rcond=None)[0]
                resid = y - (beta[0] + X[:, 0] * beta[1])
                out.loc[sub.index] = resid
            except:
                continue
        
        return out
    except Exception as e:
        return factor

# -------- 5) 因子评价指标 ----------
def ic_by_date(factor: pd.Series, future_ret: pd.Series) -> pd.Series:
    """计算因子与未来收益的IC"""
    ic = []
    for dt in sorted(set(factor.index.get_level_values(0)) & set(future_ret.index.get_level_values(0))):
        fa = factor.xs(dt)
        re = future_ret.xs(dt)
        idx = fa.index.intersection(re.index)
        if len(idx) < 20:
            ic.append((dt, np.nan))
            continue
        ic.append((dt, np.corrcoef(fa.loc[idx], re.loc[idx])[0, 1]))
    return pd.Series(dict(ic)).sort_index()

def ic_winrate(ic_s: pd.Series) -> float:
    """计算IC胜率"""
    s = ic_s.dropna()
    return (s.abs() > 0).mean() * (s.mean() >= 0) + (s.abs() > 0).mean() * (s.mean() < 0)

def ndcg_at_k(factor: pd.Series, future_ret: pd.Series, k_ratio: float = 0.1) -> float:
    """计算NDCG@k"""
    ks = []
    for dt in sorted(set(factor.index.get_level_values(0)) & set(future_ret.index.get_level_values(0))):
        fa = factor.xs(dt)
        re = future_ret.xs(dt)
        idx = fa.index.intersection(re.index)
        if len(idx) < 50:
            continue
        
        k = max(1, int(len(idx) * k_ratio))
        order = fa.loc[idx].rank(ascending=False, method="first")
        topk = order.nsmallest(k).index
        rel = pd.Series(range(1, len(idx) + 1), index=idx).loc[topk]
        
        dcg = np.sum(rel / np.log2(np.arange(2, k + 2)))
        ideal_idx = re.loc[idx].nlargest(k).index
        idcg = np.sum(rel.loc[ideal_idx] / np.log2(np.arange(2, k + 2)))
        
        ks.append(dcg / (idcg + 1e-12))
    
    return float(np.nanmean(ks)) if ks else np.nan

# -------- 6) 遗传算法优化 ----------
def sample_factor_format(available_indicators: List[str] = None) -> FactorFormat:
    """随机采样一个因子格式"""
    if available_indicators is None:
        available_indicators = FINANCIAL_FACTORS
    
    # 优先选择财务指标作为y
    financial_indicators = [ind for ind in available_indicators if ind in FINANCIAL_FACTORS]
    if financial_indicators:
        y_name = random.choice(financial_indicators)
    else:
        y_name = random.choice(available_indicators)
    
    # 选择组合模式
    mode = random.choice(["A", "B", "C"])
    
    if mode == "A":
        x_name = None
    else:
        if mode == "B":
            # 模式B：y/x，优先选择总资产作为分母
            if 'total_assets_mrq_0' in available_indicators:
                x_name = 'total_assets_mrq_0'
            else:
                x_name = random.choice(available_indicators)
        else:  # mode == "C"
            # 模式C：残差模式，优先选择市值
            if 'market_cap' in available_indicators:
                x_name = 'market_cap'
            else:
                x_name = random.choice(available_indicators)
    
    # 对数变换：财务指标通常需要取对数
    y_lg = random.random() < 0.8
    x_lg = random.random() < 0.8 if x_name else False
    
    # 时间变换：基本面因子通常需要时间变换
    y_tr = random.random() < 0.9
    x_tr = random.random() < 0.9 if x_name else False
    
    # 变换周期和形式
    y_tr_pd = random.choice(["q", "y"]) if y_tr else "q"
    x_tr_pd = random.choice(["q", "y"]) if x_tr else "q"
    
    y_tr_fm = random.choice(["diff", "pct"]) if y_tr else "diff"
    x_tr_fm = random.choice(["diff", "pct"]) if x_name else "diff"
    
    return FactorFormat(
        y=y_name,
        x=x_name,
        y_lg=y_lg, x_lg=x_lg,
        y_tr=y_tr, x_tr=x_tr,
        y_tr_pd=y_tr_pd, x_tr_pd=x_tr_pd,
        y_tr_fm=y_tr_fm, x_tr_fm=x_tr_fm,
        mode=mode
    )

def evaluate_factor(factor_format: FactorFormat, data: Dict[str, pd.Series], 
                   size: pd.Series, future_ret: pd.Series, k_ratio: float = 0.1) -> Tuple[float, float, float]:
    """评价因子，返回(|IC|, IC胜率, NDCG@k)"""
    try:
        factor = build_factor(factor_format, data, size)
        ic_s = ic_by_date(factor, future_ret)
        
        score_ic = float(np.nanmean(np.abs(ic_s)))
        score_win = float(ic_winrate(ic_s))
        score_ndcg = float(ndcg_at_k(factor, future_ret, k_ratio))
        
        return score_ic, score_win, score_ndcg
    except Exception as e:
        return 0.0, 0.0, 0.0

# -------- 7) 主函数 ----------
def main():
    """主函数：演示因子挖掘流程"""
    print("=== 简化版遗传算法挖因子系统 ===")
    
    # 示例数据生成（实际使用时替换为你的真实数据）
    print("生成示例数据...")
    np.random.seed(42)
    
    # 生成示例数据
    dates = pd.date_range('2019-01-01', '2023-12-31', freq='Q')
    stocks = [f'stock_{i:03d}' for i in range(100)]
    
    # 创建多级索引
    index = pd.MultiIndex.from_product([stocks, dates], names=['order_book_id', 'date'])
    
    # 生成示例数据
    data = {}
    data['close'] = pd.Series(np.random.randn(len(index)) * 100 + 100, index=index)
    data['volume'] = pd.Series(np.random.randn(len(index)) * 1000000 + 1000000, index=index)
    data['return'] = pd.Series(np.random.randn(len(index)) * 0.1, index=index)
    data['market_cap'] = pd.Series(np.random.randn(len(index)) * 10000000000 + 10000000000, index=index)
    
    # 生成基本面因子数据
    for factor in FINANCIAL_FACTORS[:10]:  # 只生成前10个因子作为示例
        if factor in ['total_assets_mrq_0', 'total_assets_ttm_0']:
            data[factor] = pd.Series(np.random.randn(len(index)) * 1000000000 + 1000000000, index=index)
        else:
            data[factor] = pd.Series(np.random.randn(len(index)) * 100000000 + 100000000, index=index)
    
    # 构建dataframe
    df = pd.DataFrame(data)
    df = df.reset_index()
    
    print(f"示例数据生成完成: {len(df)} 行, {len(df.columns)} 列")
    print(f"列名: {list(df.columns)}")
    
    # 数据预处理
    data_dict, size, future_ret = preprocess_data(df)
    
    if not data_dict or len(future_ret) == 0:
        print("数据预处理失败，退出")
        return
    
    # 遗传算法因子挖掘
    print("\n开始遗传算法因子挖掘...")
    available_indicators = list(data_dict.keys())
    print(f"可用指标: {len(available_indicators)} 个")
    
    best_score = 0
    best_format = None
    best_scores = []
    
    # 生成并测试多个因子格式
    n_trials = 30
    print(f"生成 {n_trials} 个随机因子格式进行测试...")
    
    for i in range(n_trials):
        random_format = sample_factor_format(available_indicators)
        try:
            ic_score, win_score, ndcg_score = evaluate_factor(random_format, data_dict, size, future_ret)
            total_score = ic_score + win_score + ndcg_score
            
            if total_score > best_score:
                best_score = total_score
                best_format = random_format
            
            best_scores.append(total_score)
            
            if (i + 1) % 10 == 0:
                print(f"  已测试 {i+1}/{n_trials} 个因子，当前最佳得分: {best_score:.4f}")
                
        except Exception as e:
            continue
    
    # 输出结果
    if best_format:
        print(f"\n最佳因子格式:")
        print(f"  y: {best_format.y}")
        print(f"  x: {best_format.x}")
        print(f"  y_lg: {best_format.y_lg}")
        print(f"  x_lg: {best_format.x_lg}")
        print(f"  y_tr: {best_format.y_tr}")
        print(f"  x_tr: {best_format.x_tr}")
        print(f"  y_tr_pd: {best_format.y_tr_pd}")
        print(f"  x_tr_pd: {best_format.x_tr_pd}")
        print(f"  y_tr_fm: {best_format.y_tr_fm}")
        print(f"  x_tr_fm: {best_format.x_tr_fm}")
        print(f"  mode: {best_format.mode}")
        print(f"  总分: {best_score:.4f}")
        
        # 重新构建最佳因子进行详细分析
        print("\n构建最佳因子进行详细分析...")
        best_factor = build_factor(best_format, data_dict, size)
        ic_s = ic_by_date(best_factor, future_ret)
        
        print(f"因子数据点: {len(best_factor)}")
        print(f"平均IC: {np.nanmean(np.abs(ic_s)):.4f}")
        print(f"IC胜率: {ic_winrate(ic_s):.4f}")
        print(f"NDCG@10%: {ndcg_at_k(best_factor, future_ret, 0.1):.4f}")
    
    print(f"\n所有因子得分统计:")
    print(f"  平均得分: {np.mean(best_scores):.4f}")
    print(f"  最高得分: {np.max(best_scores):.4f}")
    print(f"  最低得分: {np.min(best_scores):.4f}")
    
    print("\n=== 因子挖掘完成 ===")

if __name__ == "__main__":
    main()
