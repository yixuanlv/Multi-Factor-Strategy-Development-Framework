import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import os
import warnings
import time
warnings.filterwarnings('ignore')

class STRFactorCalculator:
    """STR因子计算器类 - 最终优化版本"""
    
    def __init__(self, data_path: str = None, output_dir: str = None):
        """
        初始化STR因子计算器
        
        Args:
            data_path: 数据文件路径，如果为None则使用默认路径
            output_dir: 输出目录，如果为None则使用默认路径
        """
        if data_path is None:
            # 使用相对路径，数据文件在行情数据库目录中
            self.data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "行情数据库", "data.pkl")
        else:
            self.data_path = data_path
            
        if output_dir is None:
            # 输出到因子库目录
            self.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "因子库")
        else:
            self.output_dir = output_dir
            
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 检查输出目录的写入权限，如果没有权限则使用当前目录
        test_file = os.path.join(self.output_dir, "test_write.tmp")
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"使用输出目录: {self.output_dir}")
        except Exception as e:
            print(f"警告：输出目录 {self.output_dir} 没有写入权限: {e}")
            # 使用当前目录作为备选
            self.output_dir = os.path.dirname(__file__)
            print(f"使用备选输出目录: {self.output_dir}")
        
        # 初始化数据
        self.df = None
        self.return_matrix = None
        self.salience_matrix = None
        self.cov_matrix = None
        
        # 性能监控
        self.timings = {}
        
    def _time_function(self, func_name, func, *args, **kwargs):
        """计时装饰器"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        self.timings[func_name] = end_time - start_time
        print(f"{func_name} 耗时: {self.timings[func_name]:.2f}秒")
        return result
        
    def load_data(self):
        """加载数据文件"""
        def _load():
            try:
                print(f"正在加载数据文件: {self.data_path}")
                self.df = pd.read_pickle(self.data_path)
                self.df['date'] = pd.to_datetime(self.df['date'])
                print(f"数据加载成功，共 {len(self.df)} 条记录")
            except Exception as e:
                print(f"数据加载失败: {e}")
                raise
        
        return self._time_function("数据加载", _load)
    
    def calculate_returns(self):
        """计算收益率矩阵"""
        def _calculate():
            print("正在计算收益率矩阵...")
            
            # 计算收益率
            returns = self.df.groupby("order_book_id")['close'].apply(
                lambda x: x.pct_change(fill_method=None)
            ).reset_index(level=0, drop=True)
            
            self.df["return"] = returns
            
            # 转换为矩阵形式
            self.return_matrix = self.df.pivot(
                index='date', 
                columns='order_book_id', 
                values='return'
            )
            
            print(f"收益率矩阵计算完成，形状: {self.return_matrix.shape}")
        
        return self._time_function("收益率计算", _calculate)
    
    def calculate_sigma(self):
        """计算sigma值"""
        def _calculate():
            print("正在计算sigma值...")
            
            def calc_sigma(r):
                if r.dropna().empty:
                    return pd.Series([np.nan] * len(r), index=r.index)
                median_return = r.median()
                sigma = abs(r - median_return) / (abs(r) + abs(median_return) + 0.1)
                return sigma
            
            # 按日期分组计算sigma
            self.df['sigma'] = self.df.groupby('date')['return'].apply(calc_sigma).reset_index(level=0, drop=True)
            
            # 转换为矩阵形式
            sigma_matrix = self.df.pivot(
                index='date', 
                columns='order_book_id', 
                values='sigma'
            )
            
            # 删除全为空的列
            sigma_matrix = sigma_matrix.dropna(axis=1, how='all')
            
            print(f"Sigma矩阵计算完成，形状: {sigma_matrix.shape}")
            return sigma_matrix
        
        return self._time_function("Sigma计算", _calculate)
    
    def calculate_salience(self, sigma_matrix: pd.DataFrame, delta: float = 0.8, n_jobs: int = -1):
        """计算salience权重矩阵 - 使用向量化操作优化性能"""
        def _calculate():
            print("正在计算salience权重矩阵...")
            
            # 尝试使用优化的向量化方法
            try:
                print("尝试使用向量化salience计算方法...")
                self.salience_matrix = self._calculate_salience_vectorized(sigma_matrix, delta)
                print(f"向量化salience矩阵计算完成，形状: {self.salience_matrix.shape}")
            except Exception as e:
                print(f"向量化方法失败: {e}，回退到并行计算方法...")
                self.salience_matrix = self._calculate_salience_parallel(sigma_matrix, delta, n_jobs)
        
        return self._time_function("Salience计算", _calculate)
    
    def _calculate_salience_vectorized(self, sigma_matrix: pd.DataFrame, delta: float = 0.8):
        """使用向量化操作计算salience权重 - 最高性能"""
        print("使用向量化操作计算salience权重...")
        
        # 转换为numpy数组以提高性能
        sigma_array = sigma_matrix.values.astype(float)
        dates = sigma_matrix.index
        stocks = sigma_matrix.columns
        
        n_dates, n_stocks = sigma_array.shape
        window = 20
        salience_matrix = np.full((n_dates, n_stocks), np.nan)
        
        # 预计算delta的幂次，避免重复计算
        delta_powers = delta ** np.arange(1, window + 1)
        
        # 使用更高效的向量化操作
        # 批量处理所有列，减少循环开销
        for i in range(window - 1, n_dates):
            # 获取当前窗口的数据
            start_idx = i - window + 1
            end_idx = i + 1
            
            # 窗口内的数据
            window_data = sigma_array[start_idx:end_idx, :]
            
            # 创建有效值掩码
            valid_mask = ~np.isnan(window_data)
            
            # 对每列计算salience
            for j in range(n_stocks):
                col_data = window_data[:, j]
                col_valid_mask = valid_mask[:, j]
                
                if np.sum(col_valid_mask) >= 1:  # min_periods=1
                    valid_data = col_data[col_valid_mask]
                    current_val = col_data[-1]  # 当前值
                    
                    if not np.isnan(current_val):
                        # 使用更高效的排名计算方法
                        # 直接计算排名，避免复杂的排序操作
                        rank = np.sum(valid_data > current_val) + 1
                        
                        # 计算分母
                        n = len(valid_data)
                        denom = np.sum(delta_powers[:n]) / n
                        
                        # 计算salience权重
                        if rank <= n:
                            salience = (delta ** rank) / denom
                            salience_matrix[i, j] = salience
        
        # 转换回DataFrame
        return pd.DataFrame(salience_matrix, index=dates, columns=stocks)
    
    def _calculate_salience_parallel(self, sigma_matrix: pd.DataFrame, delta: float = 0.8, n_jobs: int = -1):
        """并行计算salience权重 - 备用方法"""
        print("使用并行方法计算salience权重...")
        
        def process_column(col_data: pd.Series, delta: float) -> pd.Series:
            def compute_salience(window):
                current_val = window.iloc[-1]
                if pd.isna(current_val):
                    return np.nan
                
                non_nan = window.dropna()
                n = len(non_nan)
                if n == 0:
                    return np.nan
                
                # 当前值在非空值中的排名
                try:
                    rank = non_nan.rank(ascending=False, method='min')[window.index[-1]]
                except KeyError:
                    return np.nan
                
                denom = np.sum(delta ** np.arange(1, n + 1)) / n
                return (delta ** rank) / denom
            
            return col_data.rolling(window=20, min_periods=1).apply(compute_salience, raw=False)
        
        def salience_transform_parallel(sigma_df: pd.DataFrame, delta: float, n_jobs: int = -1) -> pd.DataFrame:
            """并行计算salience权重"""
            sigma_df = sigma_df.apply(pd.to_numeric, errors='coerce')
            
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_column)(sigma_df[col], delta)
                for col in tqdm(sigma_df.columns, desc="Salience Parallel")
            )
            
            transformed = pd.DataFrame(
                dict(zip(sigma_df.columns, results)), 
                index=sigma_df.index
            )
            return transformed
        
        # 计算salience矩阵
        return salience_transform_parallel(sigma_matrix, delta, n_jobs)
    
    def calculate_covariance(self, window: int = 20, n_jobs: int = -1):
        """计算协方差矩阵 - 使用向量化操作优化性能"""
        def _calculate():
            print("正在计算协方差矩阵...")
            
            # 确保数据类型一致
            self.salience_matrix = self.salience_matrix.astype(float)
            self.return_matrix = self.return_matrix.astype(float)
            
            # 找到共同的日期和股票
            common_dates = self.salience_matrix.index.intersection(self.return_matrix.index)
            common_stocks = self.salience_matrix.columns.intersection(self.return_matrix.columns)
            
            # 对齐数据
            self.salience_matrix = self.salience_matrix.loc[common_dates, common_stocks]
            self.return_matrix = self.return_matrix.loc[common_dates, common_stocks]
            
            print(f"数据对齐完成，共同日期: {len(common_dates)}, 共同股票: {len(common_stocks)}")
            
            # 使用向量化操作计算rolling covariance
            print("使用向量化操作计算rolling covariance...")
            
            # 优先使用pandas的rolling().cov()，这是最快的向量化方法
            try:
                print("尝试使用pandas rolling().cov()方法...")
                self.cov_matrix = self.salience_matrix.rolling(
                    window=window, 
                    min_periods=2
                ).cov(self.return_matrix)
                
                # 检查结果是否有效
                if not self.cov_matrix.isna().all().all():
                    print(f"pandas rolling().cov()方法成功，形状: {self.cov_matrix.shape}")
                    return
                else:
                    print("pandas方法结果全为空，尝试numpy向量化方法...")
                    self.cov_matrix = self._calculate_covariance_vectorized(window)
                    print(f"numpy向量化协方差矩阵计算完成，形状: {self.cov_matrix.shape}")
                    
            except Exception as e:
                print(f"向量化方法失败: {e}，回退到并行计算方法...")
                self._calculate_covariance_parallel(window, n_jobs)
        
        return self._time_function("协方差计算", _calculate)
    
    def _calculate_covariance_vectorized(self, window: int = 20):
        """使用numpy向量化操作计算rolling covariance - 最高性能"""
        print("使用numpy向量化操作计算rolling covariance...")
        
        # 转换为numpy数组以提高性能
        salience_array = self.salience_matrix.values
        return_array = self.return_matrix.values
        dates = self.salience_matrix.index
        stocks = self.salience_matrix.columns
        
        n_dates, n_stocks = salience_array.shape
        cov_matrix = np.full((n_dates, n_stocks), np.nan)
        
        # 使用numpy的向量化操作计算rolling covariance
        # 预分配内存以提高性能
        for i in range(window - 1, n_dates):
            # 获取当前窗口的数据
            start_idx = i - window + 1
            end_idx = i + 1
            
            # 窗口内的数据
            salience_window = salience_array[start_idx:end_idx, :]
            return_window = return_array[start_idx:end_idx, :]
            
            # 创建有效值掩码
            valid_mask = ~(np.isnan(salience_window) | np.isnan(return_window))
            
            # 对每列计算协方差
            for j in range(n_stocks):
                col_valid_mask = valid_mask[:, j]
                if np.sum(col_valid_mask) >= 2:  # min_periods=2
                    salience_valid = salience_window[col_valid_mask, j]
                    return_valid = return_window[col_valid_mask, j]
                    
                    # 使用numpy的向量化操作计算协方差
                    n = len(salience_valid)
                    if n > 1:
                        # 计算均值
                        salience_mean = np.mean(salience_valid)
                        return_mean = np.mean(return_valid)
                        
                        # 向量化计算协方差
                        cov = np.sum((salience_valid - salience_mean) * (return_valid - return_mean)) / (n - 1)
                        cov_matrix[i, j] = cov
        
        # 转换回DataFrame
        return pd.DataFrame(cov_matrix, index=dates, columns=stocks)
    
    def _calculate_covariance_parallel(self, window: int = 20, n_jobs: int = -1):
        """并行计算协方差矩阵 - 备用方法"""
        print("使用并行方法计算协方差矩阵...")
        
        def compute_stock_cov_vectorized(stock: str) -> pd.Series:
            s_series = self.salience_matrix[stock]
            r_series = self.return_matrix[stock]
            cov_series = s_series.rolling(window, min_periods=2).cov(r_series)
            return cov_series
        
        # 并行计算协方差
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(compute_stock_cov_vectorized)(stock) 
            for stock in tqdm(self.salience_matrix.columns, desc="计算每只股票的rolling cov")
        )
        
        # 组装结果
        self.cov_matrix = pd.DataFrame(
            dict(zip(self.salience_matrix.columns, results)), 
            index=self.salience_matrix.index
        )
        
        print(f"并行协方差矩阵计算完成，形状: {self.cov_matrix.shape}")
    
    def save_results(self):
        """保存结果"""
        def _save():
            print("正在保存结果...")
            
            # 保存最终的STR因子
            output_path = os.path.join(self.output_dir, "str.pkl")
            self.cov_matrix.to_pickle(output_path)
            
            print(f"STR因子已保存到: {output_path}")
            print(f"最终结果形状: {self.cov_matrix.shape}")
            
            # 显示一些统计信息
            print(f"非空值数量: {self.cov_matrix.count().sum()}")
            print(f"总元素数量: {self.cov_matrix.size}")
            print(f"数据完整度: {self.cov_matrix.count().sum() / self.cov_matrix.size:.2%}")
        
        return self._time_function("结果保存", _save)
    
    def run(self, delta: float = 0.8, window: int = 20, n_jobs: int = -1):
        """运行完整的STR因子计算流程"""
        print("开始STR因子计算流程...")
        total_start_time = time.time()
        
        try:
            # 1. 加载数据
            self.load_data()
            
            # 2. 计算收益率
            self.calculate_returns()
            
            # 3. 计算sigma
            sigma_matrix = self.calculate_sigma()
            
            # 4. 计算salience
            self.calculate_salience(sigma_matrix, delta, n_jobs)
            
            # 5. 计算协方差
            self.calculate_covariance(window, n_jobs)
            
            # 6. 保存结果
            self.save_results()
            
            total_time = time.time() - total_start_time
            print(f"\nSTR因子计算流程完成！总耗时: {total_time:.2f}秒")
            
            # 显示性能统计
            print("\n性能统计:")
            for step, timing in self.timings.items():
                print(f"  {step}: {timing:.2f}秒")
            
        except Exception as e:
            print(f"计算过程中出现错误: {e}")
            raise

def main():
    """主函数"""
    # 创建计算器实例
    calculator = STRFactorCalculator()
    
    # 运行计算流程
    calculator.run(delta=0.8, window=20, n_jobs=4)  # 限制并行数以提高稳定性

if __name__ == "__main__":
    main()
