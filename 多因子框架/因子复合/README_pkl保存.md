# 因子复合模块 - PKL保存功能说明

## 概述

本模块已将因子复合分析结果从Excel格式改为PKL格式保存，避免了Excel文件过大的问题，同时提供了自动文件夹创建功能。

## 主要功能

### 1. 自动文件夹创建
- 所有保存函数都会自动检查并创建必要的文件夹
- 使用 `os.makedirs(output_dir, exist_ok=True)` 确保文件夹存在
- 支持多层嵌套文件夹的创建

### 2. PKL文件保存
- **复合因子值**: 每个复合方法都会生成独立的pkl文件
- **性能汇总**: 所有方法的性能指标汇总保存
- **详细分析**: IC序列、分组收益、多空收益等详细数据

## 文件结构

```
../测试结果/因子复合结果/
├── 复合因子性能汇总.pkl                    # 性能指标汇总
├── univariate_fixed_复合因子.pkl           # 一元回归固定窗口复合因子
├── univariate_expanding_复合因子.pkl       # 一元回归扩展窗口复合因子
├── univariate_rolling_20_复合因子.pkl      # 一元回归滚动窗口(20)复合因子
├── univariate_rolling_60_复合因子.pkl      # 一元回归滚动窗口(60)复合因子
├── univariate_rolling_120_复合因子.pkl     # 一元回归滚动窗口(120)复合因子
├── ranking_add_复合因子.pkl                # 排序相加复合因子
├── ranking_multiply_复合因子.pkl           # 排序相乘复合因子
├── ic_fixed_复合因子.pkl                   # IC加权固定窗口复合因子
├── ic_expanding_复合因子.pkl               # IC加权扩展窗口复合因子
├── ic_rolling_20_复合因子.pkl              # IC加权滚动窗口(20)复合因子
├── ic_rolling_60_复合因子.pkl              # IC加权滚动窗口(60)复合因子
├── ic_rolling_120_复合因子.pkl             # IC加权滚动窗口(120)复合因子
├── multivariate_20_复合因子.pkl            # 多元回归(20)复合因子
├── multivariate_60_复合因子.pkl            # 多元回归(60)复合因子
├── multivariate_120_复合因子.pkl           # 多元回归(120)复合因子
├── univariate_fixed_详细分析.pkl           # 详细分析结果
├── univariate_expanding_详细分析.pkl       # 详细分析结果
└── ... (其他方法的详细分析文件)
```

## 使用方法

### 1. 运行主程序
```python
python main.py
```

### 2. 加载保存的数据
```python
from main import (
    load_combined_factor_from_pkl,
    load_performance_summary_from_pkl,
    load_detailed_analysis_from_pkl
)

# 加载复合因子
combined_factor = load_combined_factor_from_pkl("path/to/univariate_fixed_复合因子.pkl")

# 加载性能汇总
performance_df = load_performance_summary_from_pkl("path/to/复合因子性能汇总.pkl")

# 加载详细分析
detailed_analysis = load_detailed_analysis_from_pkl("path/to/univariate_fixed_详细分析.pkl")
```

### 3. 查看性能汇总
```python
import pandas as pd

# 加载性能汇总
performance_df = load_performance_summary_from_pkl("path/to/复合因子性能汇总.pkl")

# 按IC均值排序
ic_ranking = performance_df.sort_values('IC均值', ascending=False)
print("IC均值排名:")
print(ic_ranking[['复合方法', 'IC均值', 'ICIR', '夏普比率']].head(10))

# 按夏普比率排序
sharpe_ranking = performance_df.sort_values('夏普比率', ascending=False)
print("\n夏普比率排名:")
print(sharpe_ranking[['复合方法', '夏普比率', '年化收益率']].head(10))
```

### 4. 分析详细结果
```python
# 加载详细分析
detailed = load_detailed_analysis_from_pkl("path/to/univariate_fixed_详细分析.pkl")

# 查看IC序列
ic_series = detailed['ic_series']
print(f"IC序列长度: {len(ic_series)}")
print(f"IC均值: {ic_series.mean():.4f}")
print(f"IC标准差: {ic_series.std():.4f}")

# 查看分组收益
group_returns = detailed['group_returns']
print(f"分组收益形状: {group_returns.shape}")

# 查看多空组合收益
long_short_returns = detailed['long_short_returns']
print(f"多空组合收益长度: {len(long_short_returns)}")
```

## 优势

### 1. 避免Excel限制
- 不再受Excel行数限制(1,048,576行)
- 支持任意大小的数据保存
- 文件读写速度更快

### 2. 数据完整性
- 保持原始数据格式和精度
- 支持复杂的数据结构(DataFrame、Series等)
- 避免数据转换过程中的精度损失

### 3. 自动文件夹管理
- 无需手动创建文件夹
- 支持多层嵌套路径
- 避免因文件夹不存在导致的保存失败

### 4. 模块化设计
- 每个复合方法独立保存
- 便于后续分析和使用
- 支持选择性加载

## 注意事项

1. **文件大小**: PKL文件可能比Excel文件更大，但读写速度更快
2. **兼容性**: 确保使用相同版本的Python和pandas加载数据
3. **路径**: 使用相对路径时注意当前工作目录
4. **内存**: 加载大型PKL文件时注意内存使用情况

## 测试验证

运行测试脚本验证功能：
```python
python test_folder_creation.py
```

该脚本会测试：
- 文件夹自动创建功能
- PKL文件读写功能
- 多层嵌套路径创建
- 文件清理功能

