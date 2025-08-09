import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from typing import Dict, List, Tuple, Optional
import os
import pickle
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class FactorCombinationAnalyzer:
    """
    因子复合分析工具类
    输入：多个因子数据框的字典，收益率数据框，滚动加权窗口数量N，多元回归滚动加权窗口数量M
    处理：遍历M和N，计算权重，按行乘以因子值数据框，得到复合因子;标准化，集中测试
    输出：复合因子值excel，每个sheet为1个复合因子数据框;复合因子集中回测报表excel
    """

    def __init__(self, factors_data: Dict[str, pd.DataFrame], returns_data: pd.DataFrame, 
                 rebalance_period: int = 1):
        """
        初始化因子复合分析器
        
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
            # 合并因子数据和收益率数据
            merged_data = pd.merge(
                factor_data[['date', 'order_book_id', 'factor_value']],
                returns_data[['date', 'order_book_id', 'return']],
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
        
        # 创建宽格式数据用于后续分析
        self._create_wide_format_data()
        
    def _create_wide_format_data(self):
        """创建宽格式数据用于因子复合"""
        print("正在创建宽格式数据...")
        
        # 创建因子宽格式数据
        factor_wide_data = {}
        for factor_name, factor_data in self.aligned_factors.items():
            wide_data = factor_data.pivot(index='date', columns='order_book_id', values='factor_value')
            factor_wide_data[factor_name] = wide_data
        
        # 创建收益率宽格式数据
        returns_wide_data = {}
        for factor_name, factor_data in self.aligned_factors.items():
            wide_data = factor_data.pivot(index='date', columns='order_book_id', values='return')
            returns_wide_data[factor_name] = wide_data
        
        # 使用第一个因子的收益率数据作为基准
        self.returns_wide = returns_wide_data[list(returns_wide_data.keys())[0]]
        self.factor_wide_data = factor_wide_data
        
        print(f"宽格式数据创建完成，时间范围: {self.returns_wide.index.min()} 到 {self.returns_wide.index.max()}")
    
    def calculate_single_factor_returns(self, factor_name: str) -> pd.Series:
        """计算单个因子的因子收益率序列"""
        factor_data = self.aligned_factors[factor_name].copy()
        factor_data['future_return'] = factor_data.groupby('order_book_id')['return'].shift(-self.rebalance_period)
        
        # 按日期分组计算因子收益率
        def calculate_factor_return_for_date(group):
            if len(group) < 10:
                return np.nan
                
            x = group['factor_value'].dropna()
            y = group['future_return'].dropna()
            
            # 确保x和y长度一致
            common_idx = x.index.intersection(y.index)
            if len(common_idx) < 10:
                return np.nan
                
            x = x.loc[common_idx]
            y = y.loc[common_idx]
            
            # 标准化因子值
            x_std = (x - x.mean()) / x.std()
            
            # 计算因子收益率（回归系数）
            try:
                model = LinearRegression()
                model.fit(x_std.values.reshape(-1, 1), y.values)
                return model.coef_[0]
            except:
                return np.nan
        
        factor_returns = factor_data.groupby('date').apply(calculate_factor_return_for_date)
        return factor_returns
    
    def calculate_ic_series(self, factor_name: str, method: str = 'spearman') -> pd.Series:
        """计算单个因子的IC序列"""
        factor_data = self.aligned_factors[factor_name].copy()
        factor_data['future_return'] = factor_data.groupby('order_book_id')['return'].shift(-self.rebalance_period)
        
        # 按日期分组计算IC
        def calculate_ic_for_date(group):
            if len(group) < 10:
                return np.nan
                
            x = group['factor_value'].dropna()
            y = group['future_return'].dropna()
            
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
    
    def calculate_multiple_factor_returns(self, method: str = 'returns') -> pd.DataFrame:
        """计算多因子回归的因子收益率"""
        print("正在计算多因子回归的因子收益率...")
        
        # 准备数据
        all_dates = sorted(self.returns_wide.index)
        factor_returns_data = {}
        
        for date in tqdm(all_dates, desc="多因子回归"):
            try:
                # 获取当期因子数据
                factor_matrix = []
                for factor_name in self.factor_names:
                    if factor_name in self.factor_wide_data and date in self.factor_wide_data[factor_name].index:
                        factor_values = self.factor_wide_data[factor_name].loc[date].dropna()
                        factor_matrix.append(factor_values)
                
                if len(factor_matrix) == 0:
                    continue
                
                # 对齐股票
                common_stocks = set.intersection(*[set(f.index) for f in factor_matrix])
                if len(common_stocks) < 10:
                    continue
                
                # 获取未来收益率
                future_date_idx = all_dates.index(date) + self.rebalance_period
                if future_date_idx >= len(all_dates):
                    continue
                    
                future_date = all_dates[future_date_idx]
                if future_date not in self.returns_wide.index:
                    continue
                
                future_returns = self.returns_wide.loc[future_date][list(common_stocks)].dropna()
                
                # 准备回归数据
                X = []
                for factor_values in factor_matrix:
                    X.append(factor_values[list(common_stocks)].values)
                X = np.column_stack(X)
                
                if method == 'returns':
                    y = future_returns.values
                else:  # rank_standardized
                    y = future_returns.rank(pct=True).values
                
                # 标准化因子值
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # 多元回归
                model = LinearRegression()
                model.fit(X_scaled, y)
                
                # 保存因子收益率
                for i, factor_name in enumerate(self.factor_names):
                    if factor_name not in factor_returns_data:
                        factor_returns_data[factor_name] = {}
                    factor_returns_data[factor_name][date] = model.coef_[i]
                    
            except Exception as e:
                continue
        
        # 转换为DataFrame
        factor_returns_df = pd.DataFrame(factor_returns_data)
        return factor_returns_df
    
    def calculate_weights_univariate_regression(self, window_type: str = 'fixed', 
                                              window_size: int = None) -> pd.DataFrame:
        """
        一元回归加权方法
        
        Args:
            window_type: 窗口类型 ('fixed', 'expanding', 'rolling')
            window_size: 窗口大小（仅对rolling有效）
        """
        print(f"正在计算一元回归权重 (窗口类型: {window_type})...")
        
        # 计算每个因子的因子收益率序列
        factor_returns_dict = {}
        for factor_name in tqdm(self.factor_names, desc="计算因子收益率"):
            factor_returns = self.calculate_single_factor_returns(factor_name)
            factor_returns_dict[factor_name] = factor_returns
        
        factor_returns_df = pd.DataFrame(factor_returns_dict)
        
        # 计算权重
        if window_type == 'fixed':
            # 直接取因子收益率序列均值
            weights = factor_returns_df.mean()
        elif window_type == 'expanding':
            # 取截止当前的因子收益率序列均值
            weights = factor_returns_df.expanding().mean().iloc[-1]
        elif window_type == 'rolling':
            # 取滚动回看N个窗口的因子收益率序列均值
            if window_size is None:
                raise ValueError("rolling窗口类型需要指定window_size参数")
            weights = factor_returns_df.rolling(window_size).mean().iloc[-1]
        else:
            raise ValueError(f"不支持的窗口类型: {window_type}")
        
        # 处理NaN权重，将其设为0
        weights = weights.fillna(0)
        
        return weights
    
    def calculate_weights_ranking(self, method: str = 'add') -> Dict[str, pd.DataFrame]:
        """
        排序加权方法
        
        Args:
            method: 方法类型 ('add', 'multiply')
        """
        print(f"正在计算排序权重 (方法: {method})...")
        
        # 创建排序后的因子数据
        ranked_factors = {}
        for factor_name in tqdm(self.factor_names, desc="计算排序权重"):
            factor_wide = self.factor_wide_data[factor_name]
            # 处理全为NaN的行
            factor_wide_clean = factor_wide.fillna(factor_wide.mean().mean() if not factor_wide.isna().all().all() else 0)
            ranked_factor = factor_wide_clean.rank(axis=1, pct=True)
            
            if method == 'multiply':
                # 排序相乘：用因子值在每期的大小顺序序号除以最大序号
                max_val = ranked_factor.max().max()
                if max_val > 0:
                    ranked_factor = ranked_factor / max_val
                else:
                    ranked_factor = ranked_factor.fillna(0)
            
            ranked_factors[factor_name] = ranked_factor
        
        return ranked_factors
    
    def calculate_weights_ic(self, window_type: str = 'fixed', window_size: int = None,
                           method: str = 'spearman') -> pd.DataFrame:
        """
        IC/Rank_IC加权方法
        
        Args:
            window_type: 窗口类型 ('fixed', 'expanding', 'rolling')
            window_size: 窗口大小（仅对rolling有效）
            method: IC计算方法 ('spearman' 或 'pearson')
        """
        print(f"正在计算IC权重 (窗口类型: {window_type}, 方法: {method})...")
        
        # 计算每个因子的IC序列
        ic_dict = {}
        for factor_name in tqdm(self.factor_names, desc="计算IC序列"):
            ic_series = self.calculate_ic_series(factor_name, method)
            ic_dict[factor_name] = ic_series
        
        ic_df = pd.DataFrame(ic_dict)
        
        # 计算权重
        if window_type == 'fixed':
            # 直接取IC序列均值
            weights = ic_df.mean()
        elif window_type == 'expanding':
            # 取截止当前的IC序列均值
            weights = ic_df.expanding().mean().iloc[-1]
        elif window_type == 'rolling':
            # 取滚动回看N个窗口的IC序列均值
            if window_size is None:
                raise ValueError("rolling窗口类型需要指定window_size参数")
            weights = ic_df.rolling(window_size).mean().iloc[-1]
        else:
            raise ValueError(f"不支持的窗口类型: {window_type}")
        
        # 处理NaN权重，将其设为0
        weights = weights.fillna(0)
        
        return weights
    
    def calculate_weights_multivariate_regression(self, window_type: str = 'fixed', 
                                                window_size: int = None,
                                                method: str = 'returns') -> pd.DataFrame:
        """
        多元回归加权方法
        
        Args:
            window_type: 窗口类型 ('fixed', 'expanding', 'rolling')
            window_size: 窗口大小（仅对rolling有效）
            method: 回归目标 ('returns' 或 'rank_standardized')
        """
        print(f"正在计算多元回归权重 (窗口类型: {window_type}, 方法: {method})...")
        
        # 计算多因子回归的因子收益率
        factor_returns_df = self.calculate_multiple_factor_returns(method)
        
        # 计算权重
        if window_type == 'fixed':
            # 直接取因子收益率序列均值
            weights = factor_returns_df.mean()
        elif window_type == 'expanding':
            # 取截止当前的因子收益率序列均值
            weights = factor_returns_df.expanding().mean().iloc[-1]
        elif window_type == 'rolling':
            # 取滚动回看M个窗口的因子收益率序列均值
            if window_size is None:
                raise ValueError("rolling窗口类型需要指定window_size参数")
            weights = factor_returns_df.rolling(window_size).mean().iloc[-1]
        else:
            raise ValueError(f"不支持的窗口类型: {window_type}")
        
        # 处理NaN权重，将其设为0
        weights = weights.fillna(0)
        
        return weights
    
    def create_combined_factor(self, method: str, **kwargs) -> pd.DataFrame:
        """
        创建复合因子
        
        Args:
            method: 复合方法 ('univariate', 'ranking', 'ic', 'multivariate')
            **kwargs: 其他参数
        """
        print(f"正在创建复合因子 (方法: {method})...")
        
        try:
            if method == 'univariate':
                weights = self.calculate_weights_univariate_regression(**kwargs)
                return self._combine_factors_with_weights(weights)
                
            elif method == 'ranking':
                ranking_method = kwargs.get('ranking_method', 'add')
                ranked_factors = self.calculate_weights_ranking(method=ranking_method)
                return self._combine_ranked_factors(ranked_factors, method=ranking_method)
                
            elif method == 'ic':
                weights = self.calculate_weights_ic(**kwargs)
                return self._combine_factors_with_weights(weights)
                
            elif method == 'multivariate':
                weights = self.calculate_weights_multivariate_regression(**kwargs)
                return self._combine_factors_with_weights(weights)
                
            else:
                raise ValueError(f"不支持的复合方法: {method}")
        except Exception as e:
            print(f"创建复合因子时出现错误: {str(e)}")
            # 返回一个空的复合因子
            return pd.DataFrame(0, index=self.returns_wide.index, columns=self.returns_wide.columns)
    
    def _combine_factors_with_weights(self, weights: pd.Series) -> pd.DataFrame:
        """使用权重组合因子"""
        print("正在使用权重组合因子...")
        
        # 标准化所有因子
        standardized_factors = {}
        for factor_name in self.factor_names:
            factor_wide = self.factor_wide_data[factor_name]
            # 按行标准化，处理标准差为0的情况
            def safe_standardize(x):
                if x.std() == 0:
                    return x - x.mean()  # 如果标准差为0，只减去均值
                else:
                    return (x - x.mean()) / x.std()
            
            standardized_factor = factor_wide.apply(safe_standardize, axis=1)
            standardized_factors[factor_name] = standardized_factor
        
        # 加权组合
        combined_factor = pd.DataFrame(0, index=self.returns_wide.index, columns=self.returns_wide.columns)
        
        for factor_name in self.factor_names:
            if factor_name in weights.index and not pd.isna(weights[factor_name]):
                weight = weights[factor_name]
                factor_data = standardized_factors[factor_name]
                # 处理NaN值
                factor_data_clean = factor_data.fillna(0)
                combined_factor += weight * factor_data_clean
        
        return combined_factor
    
    def _combine_ranked_factors(self, ranked_factors: Dict[str, pd.DataFrame], 
                               method: str = 'add') -> pd.DataFrame:
        """组合排序因子"""
        print("正在组合排序因子...")
        
        if method == 'add':
            # 排序相加：等权相加
            combined_factor = pd.DataFrame(0, index=self.returns_wide.index, columns=self.returns_wide.columns)
            for factor_name, ranked_factor in ranked_factors.items():
                # 处理NaN值
                ranked_factor_clean = ranked_factor.fillna(0)
                combined_factor += ranked_factor_clean
            combined_factor = combined_factor / len(ranked_factors)
            
        elif method == 'multiply':
            # 排序相乘：直接相乘
            combined_factor = pd.DataFrame(1, index=self.returns_wide.index, columns=self.returns_wide.columns)
            for factor_name, ranked_factor in ranked_factors.items():
                # 处理NaN值，将NaN替换为1（乘法单位元）
                ranked_factor_clean = ranked_factor.fillna(1)
                combined_factor *= ranked_factor_clean
        
        return combined_factor
    
    def analyze_combined_factor(self, combined_factor: pd.DataFrame) -> Dict:
        """分析复合因子"""
        print("正在分析复合因子...")
        
        try:
            # 转换为长格式数据
            combined_factor_long = combined_factor.stack().reset_index()
            combined_factor_long.columns = ['date', 'order_book_id', 'factor_value']
            
            # 合并收益率数据
            returns_long = self.returns_wide.stack().reset_index()
            returns_long.columns = ['date', 'order_book_id', 'return']
            
            merged_data = pd.merge(combined_factor_long, returns_long, on=['date', 'order_book_id'])
            merged_data = merged_data.dropna()
            
            # 计算IC
            merged_data['future_return'] = merged_data.groupby('order_book_id')['return'].shift(-self.rebalance_period)
            
            def calculate_ic_for_date(group):
                if len(group) < 10:
                    return np.nan
                x = group['factor_value'].dropna()
                y = group['future_return'].dropna()
                common_idx = x.index.intersection(y.index)
                if len(common_idx) < 10:
                    return np.nan
                x = x.loc[common_idx]
                y = y.loc[common_idx]
                return stats.spearmanr(x, y)[0]
            
            ic_series = merged_data.groupby('date').apply(calculate_ic_for_date)
            
            # 计算统计指标
            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            icir = ic_mean / ic_std if ic_std != 0 else np.nan
            ic_positive_ratio = (ic_series > 0).mean()
            
            # 计算分组收益率
            merged_data['future_return'] = merged_data.groupby('order_book_id')['return'].shift(-self.rebalance_period)
            
            def create_groups_for_date(group):
                factor = group['factor_value'].dropna()
                if len(factor) < 10:
                    return pd.Series(index=group.index, dtype=float)
                try:
                    # 尝试创建10个分组
                    groups = pd.qcut(factor, 10, labels=False, duplicates='drop')
                except ValueError:
                    # 如果无法创建10个分组，尝试创建更少的分组
                    try:
                        n_groups = min(5, len(factor) // 2)  # 至少2个数据点一组
                        if n_groups >= 2:
                            groups = pd.qcut(factor, n_groups, labels=False, duplicates='drop')
                        else:
                            # 如果数据太少，创建2个分组
                            groups = pd.qcut(factor, 2, labels=False, duplicates='drop')
                    except ValueError:
                        # 如果还是失败，使用排名方法
                        ranks = factor.rank(method='first')
                        group_size = max(1, len(ranks) // 5)  # 至少5个分组
                        groups = (ranks - 1) // group_size
                        groups = groups.clip(upper=4)  # 最多5个分组
                
                result = pd.Series(index=group.index, dtype=float)
                result.loc[factor.index] = groups
                return result
            
            group_series = merged_data.groupby('date').apply(create_groups_for_date)
            merged_data['group'] = group_series.values
            merged_data = merged_data.dropna()
            
            # 计算分组收益率
            group_returns = merged_data.groupby(['date', 'group'])['future_return'].mean().unstack(fill_value=np.nan)
            
            # 计算多空组合收益率 - 安全地获取最高和最低分组
            available_groups = group_returns.columns.tolist()
            if len(available_groups) >= 2:
                # 获取最高分组和最低分组
                max_group = max(available_groups)
                min_group = min(available_groups)
                
                long_returns = group_returns[max_group].dropna()  # 最高分组
                short_returns = group_returns[min_group].dropna()  # 最低分组
                long_short_returns = long_returns - short_returns
            else:
                # 如果没有足够的分组，创建空的收益率序列
                long_short_returns = pd.Series(dtype=float)
            
            # 计算年化统计指标
            if len(long_short_returns) > 0:
                total_return = (1 + long_short_returns).prod() - 1
                years = len(long_short_returns) / 252
                annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan
                annual_volatility = long_short_returns.std() * np.sqrt(252)
                sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else np.nan
            else:
                annual_return = np.nan
                annual_volatility = np.nan
                sharpe_ratio = np.nan
            
            return {
                'ic_series': ic_series,
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'icir': icir,
                'ic_positive_ratio': ic_positive_ratio,
                'group_returns': group_returns,
                'long_short_returns': long_short_returns,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio
            }
            
        except Exception as e:
            print(f"分析复合因子时出现错误: {str(e)}")
            # 返回默认的分析结果
            return {
                'ic_series': pd.Series(dtype=float),
                'ic_mean': np.nan,
                'ic_std': np.nan,
                'icir': np.nan,
                'ic_positive_ratio': np.nan,
                'group_returns': pd.DataFrame(),
                'long_short_returns': pd.Series(dtype=float),
                'annual_return': np.nan,
                'annual_volatility': np.nan,
                'sharpe_ratio': np.nan
            }
    
    def generate_combined_factors_report(self, N_values: List[int], M_values: List[int],
                                       methods: List[str] = None,
                                       save_path: Optional[str] = None) -> Dict:
        """
        生成复合因子分析报告
        
        Args:
            N_values: 滚动加权窗口数量列表
            M_values: 多元回归滚动加权窗口数量列表
            methods: 要分析的复合方法列表，可选值：['univariate', 'ranking', 'ic', 'multivariate']
                    如果为None，则分析所有方法
            save_path: 保存路径
        """
        print("=" * 60)
        print("因子复合分析报告")
        print(f"N值列表: {N_values}")
        print(f"M值列表: {M_values}")
        if methods:
            print(f"选择的方法: {methods}")
        else:
            print("分析所有方法")
        print("=" * 60)
        
        # 定义所有可用的方法
        all_methods = ['univariate', 'ranking', 'ic', 'multivariate']
        
        # 如果没有指定方法，则使用所有方法
        if methods is None:
            methods = all_methods
        else:
            # 验证方法名称
            invalid_methods = [m for m in methods if m not in all_methods]
            if invalid_methods:
                raise ValueError(f"不支持的方法: {invalid_methods}。支持的方法: {all_methods}")
        
        results = {}
        
        # 1. 一元回归加权
        if 'univariate' in methods:
            print("\n1. 一元回归加权分析...")
            univariate_results = {}
            
            # 固定权重
            combined_factor = self.create_combined_factor('univariate', window_type='fixed')
            analysis = self.analyze_combined_factor(combined_factor)
            univariate_results['fixed'] = {
                'combined_factor': combined_factor,
                'analysis': analysis
            }
            
            # 扩展权重
            combined_factor = self.create_combined_factor('univariate', window_type='expanding')
            analysis = self.analyze_combined_factor(combined_factor)
            univariate_results['expanding'] = {
                'combined_factor': combined_factor,
                'analysis': analysis
            }
            
            # 滚动权重
            for N in N_values:
                combined_factor = self.create_combined_factor('univariate', window_type='rolling', window_size=N)
                analysis = self.analyze_combined_factor(combined_factor)
                univariate_results[f'rolling_{N}'] = {
                    'combined_factor': combined_factor,
                    'analysis': analysis
                }
            
            results['univariate'] = univariate_results
        
        # 2. 排序加权
        if 'ranking' in methods:
            print("\n2. 排序加权分析...")
            ranking_results = {}
            
            # 排序相加
            combined_factor = self.create_combined_factor('ranking', ranking_method='add')
            analysis = self.analyze_combined_factor(combined_factor)
            ranking_results['add'] = {
                'combined_factor': combined_factor,
                'analysis': analysis
            }
            
            # 排序相乘
            combined_factor = self.create_combined_factor('ranking', ranking_method='multiply')
            analysis = self.analyze_combined_factor(combined_factor)
            ranking_results['multiply'] = {
                'combined_factor': combined_factor,
                'analysis': analysis
            }
            
            results['ranking'] = ranking_results
        
        # 3. IC加权
        if 'ic' in methods:
            print("\n3. IC加权分析...")
            ic_results = {}
            
            # 固定权重
            combined_factor = self.create_combined_factor('ic', window_type='fixed')
            analysis = self.analyze_combined_factor(combined_factor)
            ic_results['fixed'] = {
                'combined_factor': combined_factor,
                'analysis': analysis
            }
            
            # 扩展权重
            combined_factor = self.create_combined_factor('ic', window_type='expanding')
            analysis = self.analyze_combined_factor(combined_factor)
            ic_results['expanding'] = {
                'combined_factor': combined_factor,
                'analysis': analysis
            }
            
            # 滚动权重
            for N in N_values:
                combined_factor = self.create_combined_factor('ic', window_type='rolling', window_size=N)
                analysis = self.analyze_combined_factor(combined_factor)
                ic_results[f'rolling_{N}'] = {
                    'combined_factor': combined_factor,
                    'analysis': analysis
                }
            
            results['ic'] = ic_results
        
        # 4. 多元回归加权
        if 'multivariate' in methods:
            print("\n4. 多元回归加权分析...")
            multivariate_results = {}
            
            # 固定权重
            combined_factor = self.create_combined_factor('multivariate', window_type='fixed')
            analysis = self.analyze_combined_factor(combined_factor)
            multivariate_results['fixed'] = {
                'combined_factor': combined_factor,
                'analysis': analysis
            }
            
            # 扩展权重
            combined_factor = self.create_combined_factor('multivariate', window_type='expanding')
            analysis = self.analyze_combined_factor(combined_factor)
            multivariate_results['expanding'] = {
                'combined_factor': combined_factor,
                'analysis': analysis
            }
            
            # 滚动权重
            for M in M_values:
                combined_factor = self.create_combined_factor('multivariate', window_type='rolling', window_size=M)
                analysis = self.analyze_combined_factor(combined_factor)
                multivariate_results[f'rolling_{M}'] = {
                    'combined_factor': combined_factor,
                    'analysis': analysis
                }
            
            results['multivariate'] = multivariate_results
        
        # 5. 生成图表
        if save_path:
            self.plot_combined_factors_analysis(results, save_path)
        
        # 6. 打印报告
        self.print_combined_factors_report(results)
        
        return results
    
    def plot_combined_factors_analysis(self, results: Dict, save_path: str):
        """绘制复合因子分析图表"""
        print("正在生成复合因子分析图表...")
        
        fig = plt.figure(figsize=(20, 24))
        gs = fig.add_gridspec(4, 2, height_ratios=[1.5, 2, 2, 2])
        
        # Sheet1: 复合因子性能对比热力图
        ax1 = fig.add_subplot(gs[0, :])
        
        # 收集所有复合因子的性能指标
        performance_data = []
        method_names = []
        
        for method, method_results in results.items():
            for weight_type, result in method_results.items():
                analysis = result['analysis']
                performance_data.append([
                    analysis['ic_mean'],
                    analysis['icir'],
                    analysis['ic_positive_ratio'],
                    analysis['annual_return'],
                    analysis['sharpe_ratio']
                ])
                method_names.append(f"{method}_{weight_type}")
        
        performance_df = pd.DataFrame(
            performance_data,
            index=method_names,
            columns=['IC均值', 'ICIR', 'IC正比例', '年化收益', '夏普比率']
        )
        
        sns.heatmap(performance_df.T, annot=True, fmt='.4f', cmap='RdYlBu_r', 
                   center=0, ax=ax1, cbar_kws={'label': '数值'})
        ax1.set_title('复合因子性能对比', fontsize=16, pad=20)
        ax1.set_xlabel('复合方法')
        ax1.set_ylabel('性能指标')
        
        # Sheet2: 累计IC对比
        ax2 = fig.add_subplot(gs[1, 0])
        for method, method_results in results.items():
            for weight_type, result in method_results.items():
                ic_series = result['analysis']['ic_series']
                if len(ic_series.dropna()) > 0:
                    cum_ic = ic_series.cumsum()
                    ax2.plot(cum_ic.index, cum_ic.values, 
                            label=f"{method}_{weight_type}", linewidth=2, alpha=0.8)
        ax2.set_title('复合因子累计IC对比', fontsize=14)
        ax2.set_xlabel('日期')
        ax2.set_ylabel('累计IC')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Sheet3: 多空组合累计收益率对比
        ax3 = fig.add_subplot(gs[1, 1])
        for method, method_results in results.items():
            for weight_type, result in method_results.items():
                ls_returns = result['analysis']['long_short_returns']
                if len(ls_returns) > 0:
                    cum_returns = (1 + ls_returns).cumprod()
                    ax3.plot(cum_returns.index, cum_returns.values, 
                            label=f"{method}_{weight_type}", linewidth=2, alpha=0.8)
        ax3.set_title('复合因子多空组合累计收益率对比', fontsize=14)
        ax3.set_xlabel('日期')
        ax3.set_ylabel('累计收益率')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Sheet4: 分组收益率热力图（选择表现最好的方法）
        ax4 = fig.add_subplot(gs[2, :])
        
        # 找到表现最好的方法
        best_method = None
        best_sharpe = -np.inf
        
        for method, method_results in results.items():
            for weight_type, result in method_results.items():
                sharpe = result['analysis']['sharpe_ratio']
                if not pd.isna(sharpe) and sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_method = (method, weight_type)
        
        if best_method:
            best_result = results[best_method[0]][best_method[1]]
            group_returns = best_result['analysis']['group_returns']
            
            # 计算平均分组收益率
            avg_group_returns = group_returns.mean()
            
            # 绘制分组收益率柱状图 - 使用实际的分组数量
            available_groups = sorted(avg_group_returns.index.tolist())
            groups = available_groups
            returns = [avg_group_returns.get(i, 0) for i in groups]
            
            bars = ax4.bar(groups, returns, alpha=0.7)
            ax4.set_title(f'最佳复合因子 ({best_method[0]}_{best_method[1]}) 分组收益率 ({len(groups)}组)', fontsize=14)
            ax4.set_xlabel('分组')
            ax4.set_ylabel('平均收益率')
            ax4.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, ret in zip(bars, returns):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{ret:.4f}', ha='center', va='bottom')
        
        # Sheet5: 性能对比表格
        ax5 = fig.add_subplot(gs[3, :])
        ax5.axis('off')
        
        # 创建性能对比表格
        table_data = []
        for method, method_results in results.items():
            for weight_type, result in method_results.items():
                analysis = result['analysis']
                table_data.append([
                    f"{method}_{weight_type}",
                    f"{analysis['ic_mean']:.4f}" if not pd.isna(analysis['ic_mean']) else "N/A",
                    f"{analysis['icir']:.4f}" if not pd.isna(analysis['icir']) else "N/A",
                    f"{analysis['ic_positive_ratio']:.4f}" if not pd.isna(analysis['ic_positive_ratio']) else "N/A",
                    f"{analysis['annual_return']:.2%}" if not pd.isna(analysis['annual_return']) else "N/A",
                    f"{analysis['sharpe_ratio']:.2f}" if not pd.isna(analysis['sharpe_ratio']) else "N/A"
                ])
        
        col_labels = ['复合方法', 'IC均值', 'ICIR', 'IC正比例', '年化收益', '夏普比率']
        
        table = ax5.table(cellText=table_data, colLabels=col_labels, 
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2.0)
        ax5.set_title('复合因子性能对比表', fontsize=16, pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"复合因子分析图表已保存到: {save_path}")
        
        plt.show()
    
    def print_combined_factors_report(self, results: Dict):
        """打印复合因子分析报告"""
        print("\n" + "="*60)
        print("因子复合分析详细报告")
        print("="*60)
        
        # 收集所有方法的性能指标
        all_performance = []
        
        for method, method_results in results.items():
            print(f"\n{method.upper()} 方法:")
            for weight_type, result in method_results.items():
                analysis = result['analysis']
                print(f"  {weight_type}:")
                print(f"    IC均值: {analysis['ic_mean']:.4f}" if not pd.isna(analysis['ic_mean']) else "    IC均值: N/A")
                print(f"    ICIR: {analysis['icir']:.4f}" if not pd.isna(analysis['icir']) else "    ICIR: N/A")
                print(f"    IC正比例: {analysis['ic_positive_ratio']:.4f}" if not pd.isna(analysis['ic_positive_ratio']) else "    IC正比例: N/A")
                print(f"    年化收益: {analysis['annual_return']:.2%}" if not pd.isna(analysis['annual_return']) else "    年化收益: N/A")
                print(f"    夏普比率: {analysis['sharpe_ratio']:.2f}" if not pd.isna(analysis['sharpe_ratio']) else "    夏普比率: N/A")
                
                all_performance.append({
                    'method': f"{method}_{weight_type}",
                    'ic_mean': analysis['ic_mean'],
                    'icir': analysis['icir'],
                    'ic_positive_ratio': analysis['ic_positive_ratio'],
                    'annual_return': analysis['annual_return'],
                    'sharpe_ratio': analysis['sharpe_ratio']
                })
        
        # 性能排名
        performance_df = pd.DataFrame(all_performance)
        
        print("\n" + "="*60)
        print("性能排名")
        print("="*60)
        
        # 按夏普比率排名
        sharpe_ranking = performance_df.sort_values('sharpe_ratio', ascending=False, na_position='last')
        print("\n按夏普比率排名:")
        for i, row in sharpe_ranking.iterrows():
            sharpe_val = row['sharpe_ratio']
            if pd.isna(sharpe_val):
                print(f"{i+1:2d}. {row['method']}: N/A")
            else:
                print(f"{i+1:2d}. {row['method']}: {sharpe_val:.2f}")
        
        # 按年化收益排名
        return_ranking = performance_df.sort_values('annual_return', ascending=False, na_position='last')
        print("\n按年化收益排名:")
        for i, row in return_ranking.iterrows():
            return_val = row['annual_return']
            if pd.isna(return_val):
                print(f"{i+1:2d}. {row['method']}: N/A")
            else:
                print(f"{i+1:2d}. {row['method']}: {return_val:.2%}")
        
        # 按ICIR排名
        icir_ranking = performance_df.sort_values('icir', ascending=False, na_position='last')
        print("\n按ICIR排名:")
        for i, row in icir_ranking.iterrows():
            icir_val = row['icir']
            if pd.isna(icir_val):
                print(f"{i+1:2d}. {row['method']}: N/A")
            else:
                print(f"{i+1:2d}. {row['method']}: {icir_val:.4f}")

    def print_performance_summary(self, results: Dict):
        """打印性能总结"""
        print("\n" + "="*60)
        print("因子复合分析简要总结")
        print("="*60)
        
        # 收集所有方法的性能指标
        all_performance = []
        for method, method_results in results.items():
            for weight_type, result in method_results.items():
                analysis = result['analysis']
                all_performance.append({
                    'method': f"{method}_{weight_type}",
                    'ic_mean': analysis['ic_mean'],
                    'icir': analysis['icir'],
                    'annual_return': analysis['annual_return'],
                    'sharpe_ratio': analysis['sharpe_ratio']
                })
        
        performance_df = pd.DataFrame(all_performance)
        
        # IC均值排名
        ic_ranking = performance_df.sort_values('ic_mean', ascending=False, na_position='last')
        print("\nIC均值排名 (前10):")
        for i, row in ic_ranking.head(10).iterrows():
            ic_val = row['ic_mean']
            if pd.isna(ic_val):
                print(f"{i+1:2d}. {row['method']}: N/A")
            else:
                print(f"{i+1:2d}. {row['method']}: {ic_val:.4f}")
        
        # 夏普比率排名
        sharpe_ranking = performance_df.sort_values('sharpe_ratio', ascending=False, na_position='last')
        print("\n夏普比率排名 (前10):")
        for i, row in sharpe_ranking.head(10).iterrows():
            sharpe_val = row['sharpe_ratio']
            if pd.isna(sharpe_val):
                print(f"{i+1:2d}. {row['method']}: N/A")
            else:
                print(f"{i+1:2d}. {row['method']}: {sharpe_val:.2f}")
        
        # 年化收益排名
        return_ranking = performance_df.sort_values('annual_return', ascending=False, na_position='last')
        print("\n年化收益排名 (前10):")
        for i, row in return_ranking.head(10).iterrows():
            return_val = row['annual_return']
            if pd.isna(return_val):
                print(f"{i+1:2d}. {row['method']}: N/A")
            else:
                print(f"{i+1:2d}. {row['method']}: {return_val:.2%}")
        
        # 最佳方法总结
        best_ic = ic_ranking.iloc[0]
        best_sharpe = sharpe_ranking.iloc[0]
        best_return = return_ranking.iloc[0]
        
        print("\n最佳方法总结:")
        ic_val = best_ic['ic_mean']
        sharpe_val = best_sharpe['sharpe_ratio']
        return_val = best_return['annual_return']
        
        print(f"最佳IC均值: {best_ic['method']} ({ic_val:.4f})" if not pd.isna(ic_val) else f"最佳IC均值: {best_ic['method']} (N/A)")
        print(f"最佳夏普比率: {best_sharpe['method']} ({sharpe_val:.2f})" if not pd.isna(sharpe_val) else f"最佳夏普比率: {best_sharpe['method']} (N/A)")
        print(f"最佳年化收益: {best_return['method']} ({return_val:.2%})" if not pd.isna(return_val) else f"最佳年化收益: {best_return['method']} (N/A)")

    def save_combined_factors_to_pkl(self, results: Dict, output_dir: str):
        """保存复合因子值到pkl文件"""
        print("正在保存复合因子值到pkl文件...")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 遍历所有复合方法
        for method, method_results in results.items():
            for weight_type, result in method_results.items():
                combined_factor = result['combined_factor']
                
                # 创建文件名
                filename = f"{method}_{weight_type}_复合因子.pkl"
                filepath = os.path.join(output_dir, filename)
                
                # 保存为pkl文件
                with open(filepath, 'wb') as f:
                    pickle.dump(combined_factor, f)
                
                print(f"  已保存: {filename} (形状: {combined_factor.shape})")
        
        print(f"复合因子值已保存到: {output_dir}")

    def save_analysis_report_to_pkl(self, results: Dict, output_dir: str):
        """保存复合因子分析报告到pkl文件"""
        print("正在保存复合因子分析报告到pkl文件...")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 收集所有分析结果
        report_data = []
        
        for method, method_results in results.items():
            for weight_type, result in method_results.items():
                analysis = result['analysis']
                
                report_data.append({
                    '复合方法': f"{method}_{weight_type}",
                    'IC均值': analysis['ic_mean'],
                    'IC标准差': analysis['ic_std'],
                    'ICIR': analysis['icir'],
                    'IC正比例': analysis['ic_positive_ratio'],
                    '年化收益率': analysis['annual_return'],
                    '年化波动率': analysis['annual_volatility'],
                    '夏普比率': analysis['sharpe_ratio']
                })
        
        # 创建报告DataFrame
        report_df = pd.DataFrame(report_data)
        
        # 保存性能汇总到pkl
        performance_path = os.path.join(output_dir, "复合因子性能汇总.pkl")
        with open(performance_path, 'wb') as f:
            pickle.dump(report_df, f)
        print(f"性能汇总已保存到: {performance_path}")
        
        # 保存详细分析结果到pkl
        for method, method_results in results.items():
            for weight_type, result in method_results.items():
                analysis = result['analysis']
                
                # 创建详细分析结果字典
                detailed_analysis = {
                    'ic_series': analysis['ic_series'],
                    'group_returns': analysis['group_returns'],
                    'long_short_returns': analysis['long_short_returns']
                }
                
                # 保存到pkl文件
                filename = f"{method}_{weight_type}_详细分析.pkl"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'wb') as f:
                    pickle.dump(detailed_analysis, f)
                
                print(f"  已保存: {filename}")
        
        print(f"复合因子分析报告已保存到: {output_dir}")

    def print_performance_summary(self, results: Dict):
        """打印性能总结"""
        print("\n" + "="*60)
        print("因子复合分析简要总结")
        print("="*60)
        
        # 收集所有方法的性能指标
        all_performance = []
        for method, method_results in results.items():
            for weight_type, result in method_results.items():
                analysis = result['analysis']
                all_performance.append({
                    'method': f"{method}_{weight_type}",
                    'ic_mean': analysis['ic_mean'],
                    'icir': analysis['icir'],
                    'annual_return': analysis['annual_return'],
                    'sharpe_ratio': analysis['sharpe_ratio']
                })
        
        performance_df = pd.DataFrame(all_performance)
        
        # IC均值排名
        ic_ranking = performance_df.sort_values('ic_mean', ascending=False, na_position='last')
        print("\nIC均值排名 (前10):")
        for i, row in ic_ranking.head(10).iterrows():
            ic_val = row['ic_mean']
            if pd.isna(ic_val):
                print(f"{i+1:2d}. {row['method']}: N/A")
            else:
                print(f"{i+1:2d}. {row['method']}: {ic_val:.4f}")
        
        # 夏普比率排名
        sharpe_ranking = performance_df.sort_values('sharpe_ratio', ascending=False, na_position='last')
        print("\n夏普比率排名 (前10):")
        for i, row in sharpe_ranking.head(10).iterrows():
            sharpe_val = row['sharpe_ratio']
            if pd.isna(sharpe_val):
                print(f"{i+1:2d}. {row['method']}: N/A")
            else:
                print(f"{i+1:2d}. {row['method']}: {sharpe_val:.2f}")
        
        # 年化收益排名
        return_ranking = performance_df.sort_values('annual_return', ascending=False, na_position='last')
        print("\n年化收益排名 (前10):")
        for i, row in return_ranking.head(10).iterrows():
            return_val = row['annual_return']
            if pd.isna(return_val):
                print(f"{i+1:2d}. {row['method']}: N/A")
            else:
                print(f"{i+1:2d}. {row['method']}: {return_val:.2%}")
        
        # 最佳方法总结
        best_ic = ic_ranking.iloc[0]
        best_sharpe = sharpe_ranking.iloc[0]
        best_return = return_ranking.iloc[0]
        
        print("\n最佳方法总结:")
        ic_val = best_ic['ic_mean']
        sharpe_val = best_sharpe['sharpe_ratio']
        return_val = best_return['annual_return']
        
        print(f"最佳IC均值: {best_ic['method']} ({ic_val:.4f})" if not pd.isna(ic_val) else f"最佳IC均值: {best_ic['method']} (N/A)")
        print(f"最佳夏普比率: {best_sharpe['method']} ({sharpe_val:.2f})" if not pd.isna(sharpe_val) else f"最佳夏普比率: {best_sharpe['method']} (N/A)")
        print(f"最佳年化收益: {best_return['method']} ({return_val:.2%})" if not pd.isna(return_val) else f"最佳年化收益: {best_return['method']} (N/A)")


def analyze_factor_combination(factors_data: Dict[str, pd.DataFrame], 
                             returns_data: pd.DataFrame,
                             N_values: List[int] = [20, 60, 120],
                             M_values: List[int] = [20, 60, 120],
                             methods: List[str] = None,
                             rebalance_period: int = 1,
                             save_path: Optional[str] = None) -> Dict:
    """
    因子复合分析主函数
    
    Args:
        factors_data: 字典，键为因子名称，值为因子数据框
        returns_data: 收益率数据框
        N_values: 滚动加权窗口数量列表
        M_values: 多元回归滚动加权窗口数量列表
        methods: 要分析的复合方法列表，可选值：['univariate', 'ranking', 'ic', 'multivariate']
                如果为None，则分析所有方法
        rebalance_period: 调仓周期
        save_path: 图表保存路径
        
    Returns:
        包含所有分析结果的字典
    """
    analyzer = FactorCombinationAnalyzer(factors_data, returns_data, rebalance_period)
    return analyzer.generate_combined_factors_report(N_values, M_values, methods, save_path)


# 数据加载和保存函数
def load_data_from_files(work_dir: str, factor_names: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    从文件加载行情数据和多个因子数据
    
    Args:
        work_dir: 工作目录
        factor_names: 因子名称列表，如果为None则自动检测
        
    Returns:
        (行情数据, 因子数据字典)
    """
    print("正在加载数据...")
    
    # 自动检测因子文件
    if factor_names is None:
        factor_dir = os.path.join(work_dir, "../因子库")
        if not os.path.exists(factor_dir):
            os.makedirs(factor_dir, exist_ok=True)
        factor_names = []
        for file in os.listdir(factor_dir):
            if file.endswith('.pkl'):
                factor_name = os.path.splitext(file)[0]
                factor_names.append(factor_name)
        print(f"自动检测到因子文件: {factor_names}")
    
    # 加载行情数据
    data_path = os.path.join(work_dir, "../行情数据库/data.pkl")
    data_dir = os.path.dirname(data_path)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        print(f"行情数据加载完成，数据形状: {data.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"行情数据文件不存在: {data_path}")
    
    # 加载多个因子数据
    factors_data = {}
    for factor_name in factor_names:
        factor_path = os.path.join(work_dir, f"../因子库/{factor_name}.pkl")
        factor_dir = os.path.dirname(factor_path)
        if not os.path.exists(factor_dir):
            os.makedirs(factor_dir, exist_ok=True)
        try:
            with open(factor_path, 'rb') as f:
                factor_data = pickle.load(f)
            factors_data[factor_name] = factor_data
            print(f"因子 {factor_name} 数据加载完成，数据形状: {factor_data.shape}")
        except FileNotFoundError:
            print(f"警告：因子 {factor_name} 数据文件不存在，跳过")
            continue
    
    if not factors_data:
        raise ValueError("没有找到任何有效的因子数据文件")
    
    print(f"成功加载 {len(factors_data)} 个因子数据")
    return data, factors_data


def prepare_data_for_analysis(data: pd.DataFrame, factors_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    准备分析所需的数据格式
    
    Args:
        data: 原始行情数据
        factors_data: 原始因子数据字典
        
    Returns:
        (处理后的收益率数据, 处理后的因子数据字典)
    """
    print("正在准备数据...")
    
    # 处理行情数据
    data_reset = None
    if 'close' not in data.columns:
        if isinstance(data.index, pd.MultiIndex):
            data_reset = data.reset_index()
            if len(data_reset.columns) == 3:
                data_reset.columns = ['date', 'order_book_id', 'close']
            else:
                print("数据格式需要调整，请检查数据结构")
                return None, None
        else:
            print("请检查行情数据格式，需要包含close列")
            return None, None
    else:
        if isinstance(data.index, pd.MultiIndex):
            data_reset = data.reset_index()
        else:
            data_reset = data.copy()
    
    # 准备多个因子数据
    factors_data_ready = {}
    for factor_name, factor_data in factors_data.items():
        factor_reset = None
        if isinstance(factor_data.index, pd.MultiIndex):
            factor_reset = factor_data.reset_index()
            if len(factor_reset.columns) == 3:
                factor_reset.columns = ['date', 'order_book_id', 'factor_value']
            else:
                print(f"因子 {factor_name} 数据格式需要调整，请检查数据结构")
                continue
        elif isinstance(factor_data, pd.DataFrame) and not isinstance(factor_data.index, pd.MultiIndex):
            factor_reset = factor_data.stack().reset_index()
            factor_reset.columns = ['date', 'order_book_id', 'factor_value']
        else:
            print(f"因子 {factor_name} 数据格式错误，跳过")
            continue
        
        factors_data_ready[factor_name] = factor_reset
    
    if not factors_data_ready:
        raise ValueError("没有有效的因子数据")
    
    print(f"数据准备完成")
    print(f"行情数据: {data_reset.shape}")
    print(f"因子数据数量: {len(factors_data_ready)}")
    
    return data_reset, factors_data_ready


def save_combined_factors_to_pkl(results: Dict, output_dir: str):
    """保存复合因子值到pkl文件"""
    print("正在保存复合因子值到pkl文件...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历所有复合方法
    for method, method_results in results.items():
        for weight_type, result in method_results.items():
            combined_factor = result['combined_factor']
            
            # 创建文件名
            filename = f"{method}_{weight_type}_复合因子.pkl"
            filepath = os.path.join(output_dir, filename)
            
            # 保存为pkl文件
            with open(filepath, 'wb') as f:
                pickle.dump(combined_factor, f)
            
            print(f"  已保存: {filename} (形状: {combined_factor.shape})")
    
    print(f"复合因子值已保存到: {output_dir}")


def save_analysis_report_to_pkl(results: Dict, output_dir: str):
    """保存复合因子分析报告到pkl文件"""
    print("正在保存复合因子分析报告到pkl文件...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有分析结果
    report_data = []
    
    for method, method_results in results.items():
        for weight_type, result in method_results.items():
            analysis = result['analysis']
            
            report_data.append({
                '复合方法': f"{method}_{weight_type}",
                'IC均值': analysis['ic_mean'],
                'IC标准差': analysis['ic_std'],
                'ICIR': analysis['icir'],
                'IC正比例': analysis['ic_positive_ratio'],
                '年化收益率': analysis['annual_return'],
                '年化波动率': analysis['annual_volatility'],
                '夏普比率': analysis['sharpe_ratio']
            })
    
    # 创建报告DataFrame
    report_df = pd.DataFrame(report_data)
    
    # 保存性能汇总到pkl
    performance_path = os.path.join(output_dir, "复合因子性能汇总.pkl")
    with open(performance_path, 'wb') as f:
        pickle.dump(report_df, f)
    print(f"性能汇总已保存到: {performance_path}")
    
    # 保存详细分析结果到pkl
    for method, method_results in results.items():
        for weight_type, result in method_results.items():
            analysis = result['analysis']
            
            # 创建详细分析结果字典
            detailed_analysis = {
                'ic_series': analysis['ic_series'],
                'group_returns': analysis['group_returns'],
                'long_short_returns': analysis['long_short_returns']
            }
            
            # 保存到pkl文件
            filename = f"{method}_{weight_type}_详细分析.pkl"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(detailed_analysis, f)
            
            print(f"  已保存: {filename}")
    
    print(f"复合因子分析报告已保存到: {output_dir}")


def load_combined_factor_from_pkl(filepath: str) -> pd.DataFrame:
    """从pkl文件加载复合因子"""
    with open(filepath, 'rb') as f:
        combined_factor = pickle.load(f)
    return combined_factor


def load_performance_summary_from_pkl(filepath: str) -> pd.DataFrame:
    """从pkl文件加载性能汇总"""
    with open(filepath, 'rb') as f:
        performance_df = pickle.load(f)
    return performance_df


def load_detailed_analysis_from_pkl(filepath: str) -> Dict:
    """从pkl文件加载详细分析结果"""
    with open(filepath, 'rb') as f:
        detailed_analysis = pickle.load(f)
    return detailed_analysis


def run_complete_factor_analysis(work_dir: str, 
                                factor_names: List[str] = None,
                                N_values: List[int] = None,
                                M_values: List[int] = None,
                                methods: List[str] = None,
                                rebalance_period: int = 1) -> Dict:
    """
    运行完整的因子复合分析流程
    
    Args:
        work_dir: 工作目录
        factor_names: 因子名称列表
        N_values: 滚动加权窗口数量列表
        M_values: 多元回归滚动加权窗口数量列表
        methods: 要分析的复合方法列表
        rebalance_period: 调仓周期
        
    Returns:
        分析结果字典
    """
    # 设置默认参数
    if N_values is None:
        N_values = [20, 60, 120]
    if M_values is None:
        M_values = [20, 60, 120]
    if methods is None:
        methods = ['univariate', 'ranking', 'ic', 'multivariate']
    
    print("=" * 60)
    print(f"因子复合分析")
    if factor_names:
        print(f"分析因子: {', '.join(factor_names)}")
    print(f"N值列表: {N_values}")
    print(f"M值列表: {M_values}")
    print(f"选择的方法: {methods}")
    print("=" * 60)
    
    try:
        # 加载数据
        data, factors_data = load_data_from_files(work_dir, factor_names)
        
        # 准备数据
        returns_data, factors_data_ready = prepare_data_for_analysis(data, factors_data)
        
        if returns_data is None or not factors_data_ready:
            print("数据准备失败，请检查数据格式")
            return {}
        
        # 显示数据基本信息
        print("\n数据基本信息:")
        print(f"时间范围: {returns_data['date'].min()} 到 {returns_data['date'].max()}")
        print(f"股票数量: {returns_data['order_book_id'].nunique()}")
        print(f"有效因子数量: {len(factors_data_ready)}")
        
        # 进行因子复合分析
        print("\n开始因子复合分析...")
        
        # 准备图片保存路径
        image_save_path = os.path.join(work_dir, "../测试结果/因子复合分析结果.png")
        image_save_dir = os.path.dirname(image_save_path)
        os.makedirs(image_save_dir, exist_ok=True)
        
        results = analyze_factor_combination(
            factors_data=factors_data_ready,
            returns_data=returns_data,
            N_values=N_values,
            M_values=M_values,
            methods=methods,
            rebalance_period=rebalance_period,
            save_path=image_save_path
        )
        
        print("\n因子复合分析完成！")
        
        # 保存结果
        output_dir = os.path.join(work_dir, "../测试结果/因子复合结果")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存复合因子值到pkl文件
        save_combined_factors_to_pkl(results, output_dir)
        
        # 保存分析报告到pkl文件
        save_analysis_report_to_pkl(results, output_dir)
        
        print(f"\n结果已保存到: {output_dir}")
        print(f"图表已保存到: {image_save_path}")
        
        # 打印性能总结
        analyzer = FactorCombinationAnalyzer(factors_data_ready, returns_data, rebalance_period)
        analyzer.print_performance_summary(results)
        
        return results
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}
