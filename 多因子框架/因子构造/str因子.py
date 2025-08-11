import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import os
import warnings
warnings.filterwarnings('ignore')

class STRFactorCalculator:
    """STR因子计算器类"""
    
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
        
        # 初始化数据
        self.df = None
        self.return_matrix = None
        self.salience_matrix = None
        self.cov_matrix = None
        
    def load_data(self):
        """加载数据文件"""
        try:
            print(f"正在加载数据文件: {self.data_path}")
            self.df = pd.read_pickle(self.data_path)
            self.df['date'] = pd.to_datetime(self.df['date'])
            print(f"数据加载成功，共 {len(self.df)} 条记录")
        except Exception as e:
            print(f"数据加载失败: {e}")
            raise
    
    def calculate_returns(self):
        """计算收益率矩阵"""
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
    
    def calculate_sigma(self):
        """计算sigma值"""
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
    
    def calculate_salience(self, sigma_matrix: pd.DataFrame, delta: float = 0.8, n_jobs: int = -1):
        """计算salience权重矩阵"""
        print("正在计算salience权重矩阵...")
        
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
        self.salience_matrix = salience_transform_parallel(sigma_matrix, delta, n_jobs)
        
        print(f"Salience矩阵计算完成，形状: {self.salience_matrix.shape}")
    
    def calculate_covariance(self, window: int = 20, n_jobs: int = -1):
        """计算协方差矩阵"""
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
        
        def compute_stock_cov_vectorized(stock: str) -> pd.Series:
            s_series = self.salience_matrix[stock]
            r_series = self.return_matrix[stock]
            cov_series = s_series.rolling(window, min_periods=2).cov(r_series)
            return cov_series
        
        # 并行计算协方差
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(compute_stock_cov_vectorized)(stock) 
            for stock in tqdm(common_stocks, desc="计算每只股票的rolling cov")
        )
        
        # 组装结果
        self.cov_matrix = pd.DataFrame(
            dict(zip(common_stocks, results)), 
            index=common_dates
        )
        
        print(f"协方差矩阵计算完成，形状: {self.cov_matrix.shape}")
    
    def save_results(self):
        """保存结果"""
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
    
    def run(self, delta: float = 0.8, window: int = 20, n_jobs: int = -1):
        """运行完整的STR因子计算流程"""
        print("开始STR因子计算流程...")
        
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
            
            print("STR因子计算流程完成！")
            
        except Exception as e:
            print(f"计算过程中出现错误: {e}")
            raise

def main():
    """主函数"""
    # 创建计算器实例
    calculator = STRFactorCalculator()
    
    # 运行计算流程
    calculator.run(delta=0.8, window=20, n_jobs=-1)

def test_basic_functionality():
    """测试基本功能"""
    print("开始测试基本功能...")
    
    try:
        # 创建计算器实例
        calculator = STRFactorCalculator()
        
        # 测试数据加载
        print("测试数据加载...")
        calculator.load_data()
        print(f"数据加载成功，数据形状: {calculator.df.shape}")
        
        # 测试收益率计算
        print("测试收益率计算...")
        calculator.calculate_returns()
        print(f"收益率矩阵形状: {calculator.return_matrix.shape}")
        
        # 测试sigma计算
        print("测试sigma计算...")
        sigma_matrix = calculator.calculate_sigma()
        print(f"Sigma矩阵形状: {sigma_matrix.shape}")
        
        print("基本功能测试完成！")
        return True
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        return False

if __name__ == "__main__":
    # 先运行基本功能测试
    if test_basic_functionality():
        print("\n基本功能测试通过，开始完整计算...")
        # 运行完整的STR因子计算流程
        main()
    else:
        print("基本功能测试失败，请检查数据文件")
