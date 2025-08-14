#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
遗传算法挖因子 - 测试版本
使用遗传算法自动构造和优化因子，以IC均值和IC波动为优化目标

测试配置：
- 数据采样：10%（快速验证）
- 种群大小：5（小规模测试）
- 进化代数：3（快速收敛）
- 目标：验证代码功能和多进程稳定性
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
import warnings
import random
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import gc
import psutil
from scipy import stats

# 移除对single_factor_analysis的依赖
# 直接计算IC作为评价标准

warnings.filterwarnings('ignore')

class GeneticFactorMiner:
    """遗传算法因子挖掘器"""
    
    def __init__(self, data_path, population_size=20, n_generations=15, n_jobs=None):
        self.data_path = data_path
        self.population_size = population_size
        self.n_generations = n_generations
        self.n_jobs = n_jobs if n_jobs is not None else mp.cpu_count()
        
        # 加载数据
        self.load_data()
        
        # 因子构造操作符
        self.operators = {
            'price': ['close', 'open', 'high', 'low', 'volume'],
            'returns': ['return_1d', 'return_5d', 'return_20d'],
            'volatility': ['volatility_5d', 'volatility_20d'],
            'momentum': ['momentum_5d', 'momentum_20d'],
            'mean_reversion': ['mean_reversion_5d', 'mean_reversion_20d']
        }
        
        # 技术指标参数范围
        self.param_ranges = {
            'window': [5, 10, 20, 30, 60],
            'alpha': [0.1, 0.2, 0.5, 1.0, 2.0],
            'threshold': [0.1, 0.2, 0.5, 1.0, 2.0]
        }
    
    def load_data(self):
        """加载行情数据"""
        print("正在加载行情数据...")
        try:
            with open(self.data_path, 'rb') as f:
                self.data = pickle.load(f)
            print(f"数据加载成功，数据形状: {self.data.shape}")
            
            # 验证数据完整性
            self._validate_data()
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            # 尝试加载其他数据文件
            try:
                close_path = os.path.join(os.path.dirname(self.data_path), 'close.pkl')
                with open(close_path, 'rb') as f:
                    self.data = pickle.load(f)
                print(f"使用close.pkl数据，数据形状: {self.data.shape}")
                self._validate_data()
            except Exception as e2:
                print(f"所有数据文件加载失败: {e2}")
                raise
    
    def _validate_data(self):
        """验证数据完整性"""
        print("验证数据完整性...")
        
        # 检查必需的列
        required_columns = ['date', 'order_book_id', 'close']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"数据缺少必需的列: {missing_columns}")
        
        # 检查数据类型
        if not pd.api.types.is_datetime64_any_dtype(self.data['date']):
            print("转换日期列为datetime类型...")
            self.data['date'] = pd.to_datetime(self.data['date'])
        
        # 检查数据范围
        print(f"数据时间范围: {self.data['date'].min()} 到 {self.data['date'].max()}")
        print(f"股票数量: {self.data['order_book_id'].nunique()}")
        print(f"总记录数: {len(self.data)}")
        
        # 检查缺失值
        missing_counts = self.data[required_columns].isnull().sum()
        if missing_counts.sum() > 0:
            print(f"缺失值统计: {missing_counts}")
            print("清理缺失值...")
            self.data = self.data.dropna(subset=required_columns)
            print(f"清理后记录数: {len(self.data)}")
        
        # 数据采样（测试阶段使用10%数据）
        print("测试阶段：使用10%数据进行快速验证...")
        sample_ratio = 0.1  # 采样10%
        self.data = self.data.sample(frac=sample_ratio, random_state=42).reset_index(drop=True)
        print(f"采样后记录数: {len(self.data)}")
        
        print("数据验证完成！")
    
    def create_individual(self):
        """创建个体（因子构造规则）"""
        # 随机选择因子类型
        factor_type = random.choice(['price', 'returns', 'volatility', 'momentum', 'mean_reversion'])
        
        # 随机选择操作符
        operator = random.choice(self.operators[factor_type])
        
        # 随机选择参数
        window = random.choice(self.param_ranges['window'])
        alpha = random.choice(self.param_ranges['alpha'])
        threshold = random.choice(self.param_ranges['threshold'])
        
        # 随机选择技术指标
        technical_indicator = random.choice(['sma', 'ema', 'rsi', 'macd', 'bollinger'])
        
        return [factor_type, operator, window, alpha, threshold, technical_indicator]
    
    def construct_factor(self, individual):
        """根据个体构造因子"""
        try:
            factor_type, operator, window, alpha, threshold, technical_indicator = individual
            
            # 获取基础数据
            if operator in self.data.columns:
                base_data = self.data[['date', 'order_book_id', operator]].copy()
            else:
                # 如果操作符不存在，使用close价格
                base_data = self.data[['date', 'order_book_id', 'close']].copy()
                base_data = base_data.rename(columns={'close': operator})
            
            # 按股票分组计算技术指标
            factor_data = []
            
            for stock in base_data['order_book_id'].unique():
                stock_data = base_data[base_data['order_book_id'] == stock].copy()
                stock_data = stock_data.sort_values('date').reset_index(drop=True)
                
                if len(stock_data) < window:
                    continue
                
                # 计算技术指标
                if technical_indicator == 'sma':
                    factor_value = stock_data[operator].rolling(window=window).mean()
                elif technical_indicator == 'ema':
                    factor_value = stock_data[operator].ewm(span=window).mean()
                elif technical_indicator == 'rsi':
                    factor_value = self.calculate_rsi(stock_data[operator], window)
                elif technical_indicator == 'macd':
                    factor_value = self.calculate_macd(stock_data[operator], window)
                elif technical_indicator == 'bollinger':
                    factor_value = self.calculate_bollinger(stock_data[operator], window, alpha)
                else:
                    factor_value = stock_data[operator].rolling(window=window).mean()
                
                # 标准化因子值
                factor_value = (factor_value - factor_value.rolling(window=window).mean()) / factor_value.rolling(window=window).std()
                
                # 添加阈值过滤
                factor_value = np.where(np.abs(factor_value) > threshold, factor_value, 0)
                
                # 构建结果
                result = pd.DataFrame({
                    'date': stock_data['date'],
                    'order_book_id': stock,
                    'factor_value': factor_value
                })
                
                factor_data.append(result)
            
            if not factor_data:
                return None
            
            # 合并所有股票的结果
            final_factor = pd.concat(factor_data, ignore_index=True)
            final_factor = final_factor.dropna()
            
            return final_factor
            
        except Exception as e:
            print(f"因子构造失败: {e}")
            return None
    
    def construct_factor_mp_safe(self, individual):
        """多进程安全的因子构造方法"""
        try:
            # 重新加载数据（多进程环境下需要）
            if not hasattr(self, '_mp_data') or self._mp_data is None:
                with open(self.data_path, 'rb') as f:
                    self._mp_data = pickle.load(f)
            
            factor_type, operator, window, alpha, threshold, technical_indicator = individual
            
            # 获取基础数据
            if operator in self._mp_data.columns:
                base_data = self._mp_data[['date', 'order_book_id', operator]].copy()
            else:
                # 如果操作符不存在，使用close价格
                base_data = self._mp_data[['date', 'order_book_id', 'close']].copy()
                base_data = base_data.rename(columns={'close': operator})
            
            # 按股票分组计算技术指标
            factor_data = []
            
            for stock in base_data['order_book_id'].unique():
                stock_data = base_data[base_data['order_book_id'] == stock].copy()
                stock_data = stock_data.sort_values('date').reset_index(drop=True)
                
                if len(stock_data) < window:
                    continue
                
                # 计算技术指标
                if technical_indicator == 'sma':
                    factor_value = stock_data[operator].rolling(window=window).mean()
                elif technical_indicator == 'ema':
                    factor_value = stock_data[operator].ewm(span=window).mean()
                elif technical_indicator == 'rsi':
                    factor_value = self.calculate_rsi(stock_data[operator], window)
                elif technical_indicator == 'macd':
                    factor_value = self.calculate_macd(stock_data[operator], window)
                elif technical_indicator == 'bollinger':
                    factor_value = self.calculate_bollinger(stock_data[operator], window, alpha)
                else:
                    factor_value = stock_data[operator].rolling(window=window).mean()
                
                # 标准化因子值
                factor_value = (factor_value - factor_value.rolling(window=window).mean()) / factor_value.rolling(window=window).std()
                
                # 添加阈值过滤
                factor_value = np.where(np.abs(factor_value) > threshold, factor_value, 0)
                
                # 构建结果
                result = pd.DataFrame({
                    'date': stock_data['date'],
                    'order_book_id': stock,
                    'factor_value': factor_value
                })
                
                factor_data.append(result)
            
            if not factor_data:
                return None
            
            # 合并所有股票的结果
            final_factor = pd.concat(factor_data, ignore_index=True)
            final_factor = final_factor.dropna()
            
            return final_factor
            
        except Exception as e:
            print(f"多进程因子构造失败: {e}")
            return None
    
    def calculate_rsi(self, prices, window):
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, window):
        """计算MACD指标"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=window).mean()
        return macd - signal
    
    def calculate_bollinger(self, prices, window, alpha):
        """计算布林带指标"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + alpha * std
        lower = sma - alpha * std
        return (prices - sma) / (upper - lower)
    
    def evaluate_individual(self, individual):
        """评估个体适应度"""
        try:
            # 构造因子
            factor_data = self.construct_factor(individual)
            
            if factor_data is None or len(factor_data) == 0:
                return -1000.0, 1000.0  # 惩罚值
            
            # 准备收益率数据
            returns_data = self.data[['date', 'order_book_id', 'close']].copy()
            returns_data = returns_data.merge(factor_data[['date', 'order_book_id']], 
                                           on=['date', 'order_book_id'], how='inner')
            
            # 计算未来1、5、20日收益率
            returns_data = returns_data.sort_values(['order_book_id', 'date'])
            returns_data['return_1d'] = returns_data.groupby('order_book_id')['close'].pct_change().shift(-1)
            returns_data['return_5d'] = returns_data.groupby('order_book_id')['close'].pct_change().shift(-5)
            returns_data['return_20d'] = returns_data.groupby('order_book_id')['close'].pct_change().shift(-20)
            
            # 计算IC统计（1日、5日、20日）
            ic_1d = self._calculate_ic(factor_data, returns_data, 'return_1d')
            ic_5d = self._calculate_ic(factor_data, returns_data, 'return_5d')
            ic_20d = self._calculate_ic(factor_data, returns_data, 'return_20d')
            
            # 综合IC均值（加权平均）
            ic_mean = (ic_1d['mean'] * 0.5 + ic_5d['mean'] * 0.3 + ic_20d['mean'] * 0.2)
            
            # 综合IC波动（加权平均）
            ic_std = (ic_1d['std'] * 0.5 + ic_5d['std'] * 0.3 + ic_20d['std'] * 0.2)
            
            return ic_mean, ic_std
            
        except Exception as e:
            print(f"个体评估失败: {e}")
            return -1000.0, 1000.0
    
    def _calculate_ic(self, factor_data, returns_data, return_col):
        """计算IC统计"""
        try:
            # 合并因子数据和收益率数据
            merged_data = pd.merge(
                factor_data[['date', 'order_book_id', 'factor_value']],
                returns_data[['date', 'order_book_id', return_col]],
                on=['date', 'order_book_id'],
                how='inner'
            )
            
            # 去除缺失值
            merged_data = merged_data.dropna()
            
            if len(merged_data) == 0:
                return {'mean': 0.0, 'std': 1.0}
            
            # 按日期分组计算IC
            ic_values = []
            for date in merged_data['date'].unique():
                date_data = merged_data[merged_data['date'] == date]
                
                if len(date_data) < 10:  # 至少需要10只股票
                    continue
                
                # 计算Spearman相关系数
                factor_values = date_data['factor_value'].values
                return_values = date_data[return_col].values
                
                # 去除无穷值和异常值
                valid_mask = np.isfinite(factor_values) & np.isfinite(return_values)
                if np.sum(valid_mask) < 10:
                    continue
                
                factor_values = factor_values[valid_mask]
                return_values = return_values[valid_mask]
                
                try:
                    ic = stats.spearmanr(factor_values, return_values)[0]
                    if np.isfinite(ic):
                        ic_values.append(ic)
                except:
                    continue
            
            if len(ic_values) == 0:
                return {'mean': 0.0, 'std': 1.0}
            
            ic_values = np.array(ic_values)
            
            return {
                'mean': np.mean(ic_values),
                'std': np.std(ic_values)
            }
            
        except Exception as e:
            print(f"IC计算失败: {e}")
            return {'mean': 0.0, 'std': 1.0}
    
    def evaluate_population_parallel(self, population):
        """多线程并行评估种群"""
        print(f"使用 {self.n_jobs} 个线程并行评估种群...")
        
        # 检查内存使用情况
        self._check_memory_usage()
        
        try:
            # 创建线程池
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                # 提交所有任务，直接传递个体（多线程可以共享内存）
                future_to_individual = {
                    executor.submit(self._evaluate_individual_thread, individual): individual 
                    for individual in population
                }
                
                # 收集结果
                fitnesses = []
                with tqdm(total=len(population), desc="评估进度") as pbar:
                    for future in as_completed(future_to_individual, timeout=300):  # 5分钟超时
                        individual = future_to_individual[future]
                        try:
                            fitness = future.result(timeout=60)  # 1分钟超时
                            fitnesses.append((individual, fitness))
                        except Exception as e:
                            print(f"个体评估异常: {e}")
                            fitnesses.append((individual, (-1000.0, 1000.0)))
                        finally:
                            pbar.update(1)
                            
                        # 定期清理内存
                        if len(fitnesses) % 5 == 0:
                            gc.collect()
            
            return fitnesses
            
        except Exception as e:
            print(f"多线程评估失败: {e}")
            print("切换到单线程模式...")
            return self.evaluate_population_single(population)
    
    def _check_memory_usage(self):
        """检查内存使用情况"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            print(f"当前内存使用: {memory_mb:.1f} MB")
            
            if memory_mb > 2000:  # 超过2GB
                print("内存使用较高，建议减少并行进程数")
                gc.collect()
        except ImportError:
            pass  # psutil不可用时跳过
    
    def _evaluate_individual_thread(self, individual):
        """多线程个体评估函数（可以直接访问self.data）"""
        try:
            # 构造因子
            factor_data = self.construct_factor(individual)
            
            if factor_data is None or len(factor_data) == 0:
                return -1000.0, 1000.0  # 惩罚值
            
            # 准备收益率数据
            returns_data = self.data[['date', 'order_book_id', 'close']].copy()
            returns_data = returns_data.merge(factor_data[['date', 'order_book_id']], 
                                           on=['date', 'order_book_id'], how='inner')
            
            # 计算未来1、5、20日收益率
            returns_data = returns_data.sort_values(['order_book_id', 'date'])
            returns_data['return_1d'] = returns_data.groupby('order_book_id')['close'].pct_change().shift(-1)
            returns_data['return_5d'] = returns_data.groupby('order_book_id')['close'].pct_change().shift(-5)
            returns_data['return_20d'] = returns_data.groupby('order_book_id')['close'].pct_change().shift(-20)
            
            # 计算IC统计（1日、5日、20日）
            ic_1d = self._calculate_ic(factor_data, returns_data, 'return_1d')
            ic_5d = self._calculate_ic(factor_data, returns_data, 'return_5d')
            ic_20d = self._calculate_ic(factor_data, returns_data, 'return_20d')
            
            # 综合IC均值（加权平均）
            ic_mean = (ic_1d['mean'] * 0.5 + ic_5d['mean'] * 0.3 + ic_20d['mean'] * 0.2)
            
            # 综合IC波动（加权平均）
            ic_std = (ic_1d['std'] * 0.5 + ic_5d['std'] * 0.3 + ic_20d['std'] * 0.2)
            
            return ic_mean, ic_std
            
        except Exception as e:
            print(f"多线程个体评估失败: {e}")
            return -1000.0, 1000.0
    
    @staticmethod
    def _evaluate_individual_mp_simple(individual, data_path):
        """简化的多进程个体评估函数（静态方法）"""
        try:
            # 重新加载数据
            with open(data_path, 'rb') as f:
                mp_data = pickle.load(f)
            
            # 数据采样（多进程环境下也使用10%数据）
            if len(mp_data) > 1000000:
                mp_data = mp_data.sample(frac=0.1, random_state=42).reset_index(drop=True)
            
            # 构造因子
            factor_data = GeneticFactorMiner._construct_factor_static(individual, mp_data)
            
            if factor_data is None or len(factor_data) == 0:
                return -1000.0, 1000.0  # 惩罚值
            
            # 准备收益率数据
            returns_data = mp_data[['date', 'order_book_id', 'close']].copy()
            returns_data = returns_data.merge(factor_data[['date', 'order_book_id']], 
                                           on=['date', 'order_book_id'], how='inner')
            
            # 计算未来1、5、20日收益率
            returns_data = returns_data.sort_values(['order_book_id', 'date'])
            returns_data['return_1d'] = returns_data.groupby('order_book_id')['close'].pct_change().shift(-1)
            ic_5d = GeneticFactorMiner._calculate_ic_static(factor_data, returns_data, 'return_5d')
            ic_20d = GeneticFactorMiner._calculate_ic_static(factor_data, returns_data, 'return_20d')
            
            # 综合IC均值（加权平均）
            ic_mean = (ic_1d['mean'] * 0.5 + ic_5d['mean'] * 0.3 + ic_20d['mean'] * 0.2)
            
            # 综合IC波动（加权平均）
            ic_std = (ic_1d['std'] * 0.5 + ic_5d['std'] * 0.3 + ic_20d['std'] * 0.2)
            
            return ic_mean, ic_std
            
        except Exception as e:
            print(f"多进程个体评估失败: {e}")
            return -1000.0, 1000.0
        finally:
            # 清理内存
            if 'mp_data' in locals():
                del mp_data
            if 'factor_data' in locals():
                del factor_data
            if 'returns_data' in locals():
                del returns_data
    
    @staticmethod
    def _construct_factor_static(individual, mp_data):
        """静态的因子构造方法"""
        try:
            factor_type, operator, window, alpha, threshold, technical_indicator = individual
            
            # 获取基础数据
            if operator in mp_data.columns:
                base_data = mp_data[['date', 'order_book_id', operator]].copy()
            else:
                # 如果操作符不存在，使用close价格
                base_data = mp_data[['date', 'order_book_id', 'close']].copy()
                base_data = base_data.rename(columns={'close': operator})
            
            # 按股票分组计算技术指标
            factor_data = []
            
            for stock in base_data['order_book_id'].unique():
                stock_data = base_data[base_data['order_book_id'] == stock].copy()
                stock_data = stock_data.sort_values('date').reset_index(drop=True)
                
                if len(stock_data) < window:
                    continue
                
                # 计算技术指标
                if technical_indicator == 'sma':
                    factor_value = stock_data[operator].rolling(window=window).mean()
                elif technical_indicator == 'ema':
                    factor_value = stock_data[operator].ewm(span=window).mean()
                elif technical_indicator == 'rsi':
                    factor_value = GeneticFactorMiner._calculate_rsi_static(stock_data[operator], window)
                elif technical_indicator == 'macd':
                    factor_value = GeneticFactorMiner._calculate_macd_static(stock_data[operator], window)
                elif technical_indicator == 'bollinger':
                    factor_value = GeneticFactorMiner._calculate_bollinger_static(stock_data[operator], window, alpha)
                else:
                    factor_value = stock_data[operator].rolling(window=window).mean()
                
                # 标准化因子值
                factor_value = (factor_value - factor_value.rolling(window=window).mean()) / factor_value.rolling(window=window).std()
                
                # 添加阈值过滤
                factor_value = np.where(np.abs(factor_value) > threshold, factor_value, 0)
                
                # 构建结果
                result = pd.DataFrame({
                    'date': stock_data['date'],
                    'order_book_id': stock,
                    'factor_value': factor_value
                })
                
                factor_data.append(result)
            
            if not factor_data:
                return None
            
            # 合并所有股票的结果
            final_factor = pd.concat(factor_data, ignore_index=True)
            final_factor = final_factor.dropna()
            
            return final_factor
            
        except Exception as e:
            print(f"静态因子构造失败: {e}")
            return None
    
    @staticmethod
    def _calculate_rsi_static(prices, window):
        """静态RSI计算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _calculate_macd_static(prices, window):
        """静态MACD计算"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=window).mean()
        return macd - signal
    
    @staticmethod
    def _calculate_bollinger_static(prices, window, alpha):
        """静态布林带计算"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + alpha * std
        lower = sma - alpha * std
        return (prices - sma) / (upper - lower)
    
    @staticmethod
    def _calculate_ic_static(factor_data, returns_data, return_col):
        """静态IC计算方法"""
        try:
            # 合并因子数据和收益率数据
            merged_data = pd.merge(
                factor_data[['date', 'order_book_id', 'factor_value']],
                returns_data[['date', 'order_book_id', return_col]],
                on=['date', 'order_book_id'],
                how='inner'
            )
            
            # 去除缺失值
            merged_data = merged_data.dropna()
            
            if len(merged_data) == 0:
                return {'mean': 0.0, 'std': 1.0}
            
            # 按日期分组计算IC
            ic_values = []
            for date in merged_data['date'].unique():
                date_data = merged_data[merged_data['date'] == date]
                
                if len(date_data) < 10:  # 至少需要10只股票
                    continue
                
                # 计算Spearman相关系数
                factor_values = date_data['factor_value'].values
                return_values = date_data[return_col].values
                
                # 去除无穷值和异常值
                valid_mask = np.isfinite(factor_values) & np.isfinite(return_values)
                if np.sum(valid_mask) < 10:
                    continue
                
                factor_values = factor_values[valid_mask]
                return_values = return_values[valid_mask]
                
                try:
                    ic = stats.spearmanr(factor_values, return_values)[0]
                    if np.isfinite(ic):
                        ic_values.append(ic)
                except:
                    continue
            
            if len(ic_values) == 0:
                return {'mean': 0.0, 'std': 1.0}
            
            ic_values = np.array(ic_values)
            
            return {
                'mean': np.mean(ic_values),
                'std': np.std(ic_values)
            }
            
        except Exception as e:
            print(f"静态IC计算失败: {e}")
            return {'mean': 0.0, 'std': 1.0}
    
    def _evaluate_individual_mp(self, individual):
        """多进程安全的个体评估函数"""
        try:
            # 重新加载数据（多进程环境下需要）
            with open(self.data_path, 'rb') as f:
                mp_data = pickle.load(f)
            
            # 构造因子
            factor_data = self._construct_factor_mp_safe(individual, mp_data)
            
            if factor_data is None or len(factor_data) == 0:
                return -1000.0, 1000.0  # 惩罚值
            
            # 准备收益率数据
            returns_data = mp_data[['date', 'order_book_id', 'close']].copy()
            returns_data = returns_data.merge(factor_data[['date', 'order_book_id']], 
                                           on=['date', 'order_book_id'], how='inner')
            
            # 计算未来1、5、20日收益率
            returns_data = returns_data.sort_values(['order_book_id', 'date'])
            returns_data['return_1d'] = returns_data.groupby('order_book_id')['close'].pct_change().shift(-1)
            returns_data['return_5d'] = returns_data.groupby('order_book_id')['close'].pct_change().shift(-5)
            returns_data['return_20d'] = returns_data.groupby('order_book_id')['close'].pct_change().shift(-20)
            
            # 计算IC统计（1日、5日、20日）
            ic_1d = self._calculate_ic(factor_data, returns_data, 'return_1d')
            ic_5d = self._calculate_ic(factor_data, returns_data, 'return_5d')
            ic_20d = self._calculate_ic(factor_data, returns_data, 'return_20d')
            
            # 综合IC均值（加权平均）
            ic_mean = (ic_1d['mean'] * 0.5 + ic_5d['mean'] * 0.3 + ic_20d['mean'] * 0.2)
            
            # 综合IC波动（加权平均）
            ic_std = (ic_1d['std'] * 0.5 + ic_5d['std'] * 0.3 + ic_20d['std'] * 0.2)
            
            return ic_mean, ic_std
            
        except Exception as e:
            print(f"多进程个体评估失败: {e}")
            return -1000.0, 1000.0
    
    def _construct_factor_mp_safe(self, individual, mp_data):
        """多进程安全的因子构造方法（静态方法）"""
        try:
            factor_type, operator, window, alpha, threshold, technical_indicator = individual
            
            # 获取基础数据
            if operator in mp_data.columns:
                base_data = mp_data[['date', 'order_book_id', operator]].copy()
            else:
                # 如果操作符不存在，使用close价格
                base_data = mp_data[['date', 'order_book_id', 'close']].copy()
                base_data = base_data.rename(columns={'close': operator})
            
            # 按股票分组计算技术指标
            factor_data = []
            
            for stock in base_data['order_book_id'].unique():
                stock_data = base_data[base_data['order_book_id'] == stock].copy()
                stock_data = stock_data.sort_values('date').reset_index(drop=True)
                
                if len(stock_data) < window:
                    continue
                
                # 计算技术指标
                if technical_indicator == 'sma':
                    factor_value = stock_data[operator].rolling(window=window).mean()
                elif technical_indicator == 'ema':
                    factor_value = stock_data[operator].ewm(span=window).mean()
                elif technical_indicator == 'rsi':
                    factor_value = self._calculate_rsi_mp(stock_data[operator], window)
                elif technical_indicator == 'macd':
                    factor_value = self._calculate_macd_mp(stock_data[operator], window)
                elif technical_indicator == 'bollinger':
                    factor_value = self._calculate_bollinger_mp(stock_data[operator], window, alpha)
                else:
                    factor_value = stock_data[operator].rolling(window=window).mean()
                
                # 标准化因子值
                factor_value = (factor_value - factor_value.rolling(window=window).mean()) / factor_value.rolling(window=window).std()
                
                # 添加阈值过滤
                factor_value = np.where(np.abs(factor_value) > threshold, factor_value, 0)
                
                # 构建结果
                result = pd.DataFrame({
                    'date': stock_data['date'],
                    'order_book_id': stock,
                    'factor_value': factor_value
                })
                
                factor_data.append(result)
            
            if not factor_data:
                return None
            
            # 合并所有股票的结果
            final_factor = pd.concat(factor_data, ignore_index=True)
            final_factor = final_factor.dropna()
            
            return final_factor
            
        except Exception as e:
            print(f"多进程因子构造失败: {e}")
            return None
    
    def _calculate_rsi_mp(self, prices, window):
        """多进程安全的RSI计算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd_mp(self, prices, window):
        """多进程安全的MACD计算"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=window).mean()
        return macd - signal
    
    def _calculate_bollinger_mp(self, prices, window, alpha):
        """多进程安全的布林带计算"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + alpha * std
        lower = sma - alpha * std
        return (prices - sma) / (upper - lower)
    
    def crossover(self, ind1, ind2):
        """交叉操作"""
        if random.random() < 0.7:  # 交叉率70%
            cxpoint = random.randint(1, len(ind1) - 1)
            ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]
        return ind1, ind2
    
    def mutate_individual(self, individual):
        """变异操作"""
        for i in range(len(individual)):
            if random.random() < 0.1:  # 变异率10%
                if i == 0:  # 因子类型
                    individual[i] = random.choice(list(self.operators.keys()))
                elif i == 1:  # 操作符
                    factor_type = individual[0]
                    individual[i] = random.choice(self.operators[factor_type])
                elif i == 2:  # 窗口
                    individual[i] = random.choice(self.param_ranges['window'])
                elif i == 3:  # alpha
                    individual[i] = random.choice(self.param_ranges['alpha'])
                elif i == 4:  # threshold
                    individual[i] = random.choice(self.param_ranges['threshold'])
                elif i == 5:  # 技术指标
                    individual[i] = random.choice(['sma', 'ema', 'rsi', 'macd', 'bollinger'])
        return individual
    
    def run_evolution(self):
        """运行遗传算法进化"""
        print("开始遗传算法进化...")
        print(f"种群大小: {self.population_size}")
        print(f"进化代数: {self.n_generations}")
        print(f"并行线程数: {self.n_jobs}")
        
        # 创建初始种群
        population = [self.create_individual() for _ in range(self.population_size)]
        
        # 记录最佳个体
        best_individual = None
        best_fitness = (-float('inf'), float('inf'))
        
        # 进化循环
        total_start_time = time.time()
        generation_times = []
        
        for generation in range(self.n_generations):
            print(f"\n第 {generation + 1} 代进化...")
            start_time = time.time()
            
            # 评估种群（根据n_jobs选择并行或单线程）
            if self.n_jobs > 1:
                try:
                    fitnesses = self.evaluate_population_parallel(population)
                except Exception as e:
                    print(f"多线程评估失败: {e}")
                    print("切换到单线程模式...")
                    self.n_jobs = 1
                    fitnesses = self.evaluate_population_single(population)
            else:
                fitnesses = self.evaluate_population_single(population)
            
            # 更新最佳个体
            for individual, fitness in fitnesses:
                if fitness[0] > best_fitness[0] or (fitness[0] == best_fitness[0] and fitness[1] < best_fitness[1]):
                    best_fitness = fitness
                    best_individual = individual[:]
            
            # 选择下一代（锦标赛选择）
            new_population = []
            for _ in range(self.population_size):
                # 锦标赛选择
                tournament = random.sample(fitnesses, 3)
                winner = max(tournament, key=lambda x: (x[1][0], -x[1][1]))
                new_population.append(winner[0][:])
            
            population = new_population
            
            # 交叉和变异
            for i in range(0, len(population), 2):
                if i + 1 < len(population):
                    # 交叉
                    population[i], population[i+1] = self.crossover(population[i], population[i+1])
                    
                    # 变异
                    population[i] = self.mutate_individual(population[i])
                    population[i+1] = self.mutate_individual(population[i+1])
            
            # 计算耗时
            generation_time = time.time() - start_time
            generation_times.append(generation_time)
            
            # 打印当前代最佳适应度和统计信息
            current_best = max(fitnesses, key=lambda x: (x[1][0], -x[1][1]))
            avg_generation_time = np.mean(generation_times)
            remaining_generations = self.n_generations - generation - 1
            estimated_remaining_time = remaining_generations * avg_generation_time
            
            print(f"当前代最佳适应度: IC均值={current_best[1][0]:.4f}, IC波动={current_best[1][1]:.4f}")
            print(f"历史最佳适应度: IC均值={best_fitness[0]:.4f}, IC波动={best_fitness[1]:.4f}")
            print(f"第{generation + 1}代耗时: {generation_time:.2f}秒")
            print(f"平均代耗时: {avg_generation_time:.2f}秒")
            print(f"预计剩余时间: {estimated_remaining_time/60:.1f}分钟")
        
        total_time = time.time() - total_start_time
        print(f"\n总耗时: {total_time/60:.1f}分钟")
        print(f"平均每代耗时: {np.mean(generation_times):.2f}秒")
        if self.n_jobs > 1:
            print(f"多线程加速比: {np.mean(generation_times) * self.n_generations / total_time:.2f}x")
        
        print(f"\n进化完成！")
        print(f"最佳个体: {best_individual}")
        print(f"最佳适应度: IC均值={best_fitness[0]:.4f}, IC波动={best_fitness[1]:.4f}")
        
        return best_individual, best_fitness
    
    def evaluate_population_single(self, population):
        """单线程评估种群"""
        print("使用单线程评估种群...")
        fitnesses = []
        
        with tqdm(total=len(population), desc="评估进度") as pbar:
            for individual in population:
                try:
                    fitness = self.evaluate_individual(individual)
                    fitnesses.append((individual, fitness))
                except Exception as e:
                    print(f"个体评估失败: {e}")
                    fitnesses.append((individual, (-1000.0, 1000.0)))
                finally:
                    pbar.update(1)
        
        return fitnesses
    
    def analyze_best_factor(self, best_individual):
        """分析最佳因子"""
        print("\n正在分析最佳因子...")
        
        # 构造最佳因子
        factor_data = self.construct_factor(best_individual)
        
        if factor_data is None:
            print("最佳因子构造失败")
            return None
        
        # 准备收益率数据
        returns_data = self.data[['date', 'order_book_id', 'close']].copy()
        returns_data = returns_data.merge(factor_data[['date', 'order_book_id']], 
                                       on=['date', 'order_book_id'], how='inner')
        
        # 计算未来1、5、20日收益率
        returns_data = returns_data.sort_values(['order_book_id', 'date'])
        returns_data['return_1d'] = returns_data.groupby('order_book_id')['close'].pct_change().shift(-1)
        returns_data['return_5d'] = returns_data.groupby('order_book_id')['close'].pct_change().shift(-5)
        returns_data['return_20d'] = returns_data.groupby('order_book_id')['close'].pct_change().shift(-20)
        
        # 计算IC统计（1日、5日、20日）
        ic_1d = self._calculate_ic(factor_data, returns_data, 'return_1d')
        ic_5d = self._calculate_ic(factor_data, returns_data, 'return_5d')
        ic_20d = self._calculate_ic(factor_data, returns_data, 'return_20d')
        
        # 打印分析结果
        print(f"\n最佳因子分析结果:")
        print(f"因子类型: {best_individual[0]}")
        print(f"操作符: {best_individual[1]}")
        print(f"参数: {best_individual[2:5]}")
        print(f"技术指标: {best_individual[5]}")
        print(f"\nIC分析结果:")
        print(f"1日IC均值: {ic_1d['mean']:.4f}, 标准差: {ic_1d['std']:.4f}")
        print(f"5日IC均值: {ic_5d['mean']:.4f}, 标准差: {ic_5d['std']:.4f}")
        print(f"20日IC均值: {ic_20d['mean']:.4f}, 标准差: {ic_20d['std']:.4f}")
        
        # 计算综合IC
        ic_mean = (ic_1d['mean'] * 0.5 + ic_5d['mean'] * 0.3 + ic_20d['mean'] * 0.2)
        ic_std = (ic_1d['std'] * 0.5 + ic_5d['std'] * 0.3 + ic_20d['std'] * 0.2)
        print(f"综合IC均值: {ic_mean:.4f}, 标准差: {ic_std:.4f}")
        
        return {
            'factor_data': factor_data,
            'ic_1d': ic_1d,
            'ic_5d': ic_5d,
            'ic_20d': ic_20d,
            'ic_mean': ic_mean,
            'ic_std': ic_std
        }

def test_basic_functionality():
    """测试基本功能"""
    print("=" * 60)
    print("测试基本功能...")
    print("=" * 60)
    
    # 数据路径
    data_path = os.path.join(os.path.dirname(__file__), '..', '行情数据库', 'data.pkl')
    
    if not os.path.exists(data_path):
        print(f"数据文件不存在: {data_path}")
        return False
    
    try:
        # 创建小规模测试实例
        miner = GeneticFactorMiner(
            data_path=data_path,
            population_size=2,       # 极小种群
            n_generations=1,         # 极少代数
            n_jobs=1                 # 单进程
        )
        
        # 测试个体创建
        individual = miner.create_individual()
        print(f"✓ 个体创建成功: {individual}")
        
        # 测试因子构造
        factor_data = miner.construct_factor(individual)
        if factor_data is not None:
            print(f"✓ 因子构造成功，数据形状: {factor_data.shape}")
        else:
            print("✗ 因子构造失败")
            return False
        
        # 测试个体评估
        fitness = miner.evaluate_individual(individual)
        print(f"✓ 个体评估成功: IC均值={fitness[0]:.4f}, IC波动={fitness[1]:.4f}")
        
        print("✓ 基本功能测试通过！")
        return True
        
    except Exception as e:
        print(f"✗ 基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main(
    data_sample_desc="10%（快速验证）",
    population_size=16,
    n_generations=3,
    n_jobs=16,  # 默认使用4个线程
    data_dir=None
):
    """主函数"""
    print("=" * 60)
    print("遗传算法因子挖掘器 - 测试模式")
    print("=" * 60)
    print("测试配置：")
    print(f"- 数据采样：{data_sample_desc}")
    print(f"- 种群大小：{population_size}（小规模测试）")
    print(f"- 进化代数：{n_generations}（快速收敛）")
    print(f"- 线程模式：{'单线程（确保稳定性）' if n_jobs == 1 else f'{n_jobs}线程并行'}")
    print("=" * 60)
    
    # 先运行基本功能测试
    if not test_basic_functionality():
        print("基本功能测试失败，退出程序")
        return
    
    print("\n基本功能测试通过，开始遗传算法...")
    print("=" * 60)
    
    # 数据路径
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', '行情数据库')
    data_path = os.path.join(data_dir, 'data.pkl')
    
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        print(f"数据文件不存在: {data_path}")
        # 尝试其他数据文件
        close_path = os.path.join(data_dir, 'close.pkl')
        if os.path.exists(close_path):
            data_path = close_path
            print(f"使用数据文件: {data_path}")
        else:
            print("未找到可用的数据文件")
            return
    
    # 获取CPU核心数并设置合理的线程数
    cpu_count = mp.cpu_count()
    recommended_jobs = min(8, cpu_count)  # 线程数可以等于CPU核心数，最多8个
    
    print(f"检测到CPU核心数: {cpu_count}")
    print(f"建议并行线程数: {recommended_jobs}")
    
    # 创建遗传算法因子挖掘器（测试阶段使用小规模参数）
    try:
        miner = GeneticFactorMiner(
            data_path=data_path,
            population_size=population_size,       # 可调参数
            n_generations=n_generations,           # 可调参数
            n_jobs=n_jobs                          # 可调参数（多线程）
        )
        
        # 运行遗传算法进化
        best_individual, best_fitness = miner.run_evolution()
        
        # 分析最佳因子
        report = miner.analyze_best_factor(best_individual)
        
        print("\n" + "=" * 60)
        print("遗传算法因子挖掘完成！")
        print("=" * 60)
        print(f"最佳因子: {best_individual}")
        print(f"最佳适应度: IC均值={best_fitness[0]:.4f}, IC波动={best_fitness[1]:.4f}")
        print(f"详细IC分析结果已在上方显示")
        
    except Exception as e:
        print(f"运行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 尝试使用单线程模式
        print("\n尝试使用单线程模式...")
        try:
            miner = GeneticFactorMiner(
                data_path=data_path,
                population_size=population_size,
                n_generations=n_generations,
                n_jobs=1  # 单线程
            )
            
            best_individual, best_fitness = miner.run_evolution()
            print(f"\n单线程模式成功！最佳因子: {best_individual}")
            
            # 分析最佳因子
            report = miner.analyze_best_factor(best_individual)
            
        except Exception as e2:
            print(f"单线程模式也失败: {e2}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
