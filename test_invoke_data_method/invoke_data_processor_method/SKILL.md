---
name: invoke_data_processor_method
description: 数据处理方法调用工具。提供7大类数据处理算子，包括缺失值处理、滤波降噪、数据替换、异常处理、差分算子、重采样和标准化/归一化。支持分步调用每个算子，通过DataFrame方式传递数据，最后可导出CSV文件。使用此skill来处理各种数据清洗和预处理任务。
---

# invoke_data_processor_method 数据处理Skill

本skill提供完整的数据处理能力，包含7大类算子，适用于各种数据集的清洗和预处理。

## 快速开始

### 1. 导入模块

#### 方式一：通过DataProcessor统一入口
```python
import sys
sys.path.append('/path/to/invoke_data_processor_method/scripts')
from data_processor import DataProcessor

# 创建处理器实例
processor = DataProcessor()
```

#### 方式二：单独导入各算子（推荐）
```python
import sys
sys.path.append('/path/to/invoke_data_processor_method/scripts')

# 单独导入需要的算子
from missing_value import MissingValueHandler
from filter_denoiser import FilterDenoiser
from data_transformer import DataTransformer
from outlier_handler import OutlierHandler
from differential_operator import DifferentialOperator
from resampler import Resampler
from normalizer import Normalizer

# 创建各算子实例
missing_handler = MissingValueHandler()
filter_handler = FilterDenoiser()
# ... 其他算子同理
```

### 2. 加载数据

```python
# 从CSV加载
df = processor.load_csv('your_data.csv')

# 或直接使用DataFrame
import pandas as pd
df = pd.read_csv('your_data.csv')
```

### 3. 分步处理数据

#### 缺失值处理
```python
# 前向填充
df = processor.missing_value.fill_forward(df)

# 后向填充
df = processor.missing_value.fill_backward(df)

# 均值填充
df = processor.missing_value.fill_mean(df)

# 中位数填充
df = processor.missing_value.fill_median(df)

# 固定值填充
df = processor.missing_value.fill_constant(df, value=-1)

# 众数填充
df = processor.missing_value.fill_mode(df)

# 线性插值
df = processor.missing_value.interpolate_linear(df)

# 样条插值
df = processor.missing_value.interpolate_spline(df, order=3)

# 行删除
df = processor.missing_value.delete_row(df)

# 列删除（缺失率>40%）
df = processor.missing_value.delete_column(df, threshold=0.4)
```

#### 滤波降噪
```python
# 中值滤波
df = processor.filter_denoiser.median_filter(df, kernel_size=3)

# 移动平均
df = processor.filter_denoiser.moving_average(df, window=5)

# 指数加权移动平均
df = processor.filter_denoiser.exponential_moving_average(df, span=5)

# 低通滤波
df = processor.filter_denoiser.lowpass_filter(df, cutoff=0.1)

# 傅里叶变换
df = processor.filter_denoiser.fourier_transform(df, low_freq_ratio=0.1)
```

#### 数据替换
```python
# 对数变换
df = processor.data_transformer.log_transform(df)

# 差分变换
df = processor.data_transformer.diff_transform(df)

# 标准化
df = processor.data_transformer.standardize(df)

# 类别编码（标签编码）
df = processor.data_transformer.category_encode(df, method='label')

# 类别编码（独热编码）
df = processor.data_transformer.category_encode(df, method='onehot')
```

#### 异常处理
```python
# 3-Sigma方法（截断）
df = processor.outlier_handler.three_sigma(df, n_sigma=3.0, method='clip')

# 3-Sigma方法（替换为NaN）
df = processor.outlier_handler.three_sigma(df, method='nan')

# 四分位距法
df = processor.outlier_handler.quartile_method(df, k=1.5, method='clip')

# 聚类检测
df = processor.outlier_handler.clustering_detection(df, n_clusters=3, threshold=2.0)

# 移动标准差法
df = processor.outlier_handler.moving_std(df, window=10, threshold=3.0)

# Z-score方法
df = processor.outlier_handler.zscore(df, threshold=3.0, method='clip')
```

#### 差分算子
```python
# 一阶差分
df = processor.differential_operator.first_order_diff(df)

# 滞后k阶差分
df = processor.differential_operator.lag_diff(df, k=4)

# 二阶差分
df = processor.differential_operator.second_order_diff(df)

# n阶差分
df = processor.differential_operator.n_order_diff(df, n=3)

# 季节性差分
df = processor.differential_operator.seasonal_diff(df, period=12)

# 对数一阶差分
df = processor.differential_operator.log_diff(df)

# 百分比变化
df = processor.differential_operator.percent_diff(df)
```

#### 重采样
```python
# 上采样（线性插值）
df = processor.resampler.upsample_linear(df, factor=2)

# 上采样（多项式插值）
df = processor.resampler.upsample_polynomial(df, factor=2, degree=3)

# 上采样（样条插值）
df = processor.resampler.upsample_spline(df, factor=2, order=3)

# 下采样（均值聚合）
df = processor.resampler.downsample_mean(df, factor=2)

# 下采样（求和聚合）
df = processor.resampler.downsample_sum(df, factor=2)
```

#### 标准化/归一化
```python
# Z-Score标准化
df = processor.normalizer.zscore_standardize(df)

# 均值中心化
df = processor.normalizer.mean_center(df)

# 稳健标准化
df = processor.normalizer.robust_standardize(df)

# Min-Max归一化
df = processor.normalizer.minmax_normalize(df)

# 指定区间缩放
df = processor.normalizer.range_normalize(df, feature_range=(-1, 1))

# 最大绝对值归一化
df = processor.normalizer.maxabs_normalize(df)

# 对数归一化
df = processor.normalizer.log_normalize(df)

# Box-Cox变换
df = processor.normalizer.boxcox_transform(df)
```

### 4. 指定列处理

```python
# 只处理特定列
df = processor.missing_value.fill_mean(df, columns=['age', 'income'])
df = processor.normalizer.zscore_standardize(df, columns=['price', 'quantity'])
```

### 5. 保存结果

```python
# 保存为CSV
processor.save_to_csv(df, 'output.csv')
```

## 7大类算子详细说明

### 1. 缺失值处理 (MissingValueHandler)
- `delete_row`: 行删除
- `delete_column`: 列删除
- `fill_forward`: 前项填充
- `fill_backward`: 后向填充
- `fill_mean`: 均值填充
- `fill_median`: 中位数填充
- `fill_constant`: 固定值填充
- `fill_mode`: 众数填充
- `interpolate_linear`: 线性插值
- `interpolate_spline`: 样条插值

### 2. 滤波降噪 (FilterDenoiser)
- `median_filter`: 中值滤波
- `moving_average`: 移动平均
- `exponential_moving_average`: 指数加权移动平均
- `lowpass_filter`: 低通滤波
- `fourier_transform`: 傅里叶变换

### 3. 数据替换 (DataTransformer)
- `log_transform`: 对数变换
- `diff_transform`: 差分变换
- `standardize`: 标准化
- `category_encode`: 类别编码替换

### 4. 异常处理 (OutlierHandler)
- `three_sigma`: 3-Sigma方法
- `quartile_method`: 四分位距法
- `clustering_detection`: 聚类检测
- `moving_std`: 移动标准差法
- `zscore`: Z-score方法

### 5. 差分算子 (DifferentialOperator)
- `first_order_diff`: 普通一阶差分
- `lag_diff`: 滞后k阶差分
- `second_order_diff`: 二阶差分
- `n_order_diff`: n阶差分
- `seasonal_diff`: 季节性差分
- `log_diff`: 对数一阶差分
- `percent_diff`: 相对变化率

### 6. 重采样 (Resampler)
- `upsample_linear`: 线性插值上采样
- `upsample_polynomial`: 多项式插值上采样
- `upsample_spline`: 样条插值上采样
- `downsample_mean`: 均值聚合下采样
- `downsample_sum`: 求和聚合下采样

### 7. 标准化/归一化 (Normalizer)
- `zscore_standardize`: Z-Score标准化
- `mean_center`: 均值中心化
- `robust_standardize`: 稳健标准化
- `minmax_normalize`: Min-Max归一化
- `range_normalize`: 指定区间缩放
- `maxabs_normalize`: 最大绝对值归一化
- `log_normalize`: 对数变换归一化
- `boxcox_transform`: Box-Cox变换

## 工具函数

```python
# 获取数值列
numeric_cols = processor.get_numeric_columns(df)

# 获取缺失值信息
missing_info = processor.get_missing_info(df)
```

## 数据处理流程示例

```python
import sys
sys.path.append('/path/to/invoke_data_processor_method/scripts')
from data_processor import DataProcessor
import pandas as pd

# 创建处理器
processor = DataProcessor()

# 加载数据
df = processor.load_csv('sales_data.csv')

# 查看缺失值
print(processor.get_missing_info(df))

# 1. 缺失值填充
df = processor.missing_value.fill_forward(df)
df = processor.missing_value.interpolate_linear(df)

# 2. 滤波降噪
df = processor.filter_denoiser.moving_average(df, window=5)
df = processor.filter_denoiser.median_filter(df, kernel_size=3)

# 3. 异常处理
df = processor.outlier_handler.quartile_method(df, method='clip')

# 4. 标准化
df = processor.normalizer.zscore_standardize(df)

# 保存结果
processor.save_to_csv(df, 'processed_data.csv')
```

## 单独使用各算子文件

每个算子都封装在独立的文件中，可以单独导入使用：

```python
import sys
sys.path.append('/path/to/invoke_data_processor_method/scripts')

# 缺失值处理 - missing_value.py
from missing_value import MissingValueHandler
handler = MissingValueHandler()
df = handler.fill_forward(df)
df = handler.interpolate_linear(df)

# 滤波降噪 - filter_denoiser.py
from filter_denoiser import FilterDenoiser
filter_handler = FilterDenoiser()
df = filter_handler.moving_average(df, window=5)
df = filter_handler.median_filter(df, kernel_size=3)

# 数据替换 - data_transformer.py
from data_transformer import DataTransformer
transformer = DataTransformer()
df = transformer.log_transform(df)
df = transformer.standardize(df)

# 异常处理 - outlier_handler.py
from outlier_handler import OutlierHandler
outlier = OutlierHandler()
df = outlier.three_sigma(df, method='clip')
df = outlier.quartile_method(df, method='clip')

# 差分算子 - differential_operator.py
from differential_operator import DifferentialOperator
diff_op = DifferentialOperator()
df = diff_op.first_order_diff(df)
df = diff_op.seasonal_diff(df, period=12)

# 重采样 - resampler.py
from resampler import Resampler
resampler = Resampler()
df = resampler.upsample_linear(df, factor=2)
df = resampler.downsample_mean(df, factor=2)

# 标准化/归一化 - normalizer.py
from normalizer import Normalizer
normalizer = Normalizer()
df = normalizer.zscore_standardize(df)
df = normalizer.minmax_normalize(df)
```

## 依赖

- pandas
- numpy
- scipy
- scikit-learn

## 代码位置

- 主入口: `scripts/data_processor.py`
- 缺失值处理: `scripts/missing_value.py`
- 滤波降噪: `scripts/filter_denoiser.py`
- 数据替换: `scripts/data_transformer.py`
- 异常处理: `scripts/outlier_handler.py`
- 差分算子: `scripts/differential_operator.py`
- 重采样: `scripts/resampler.py`
- 标准化/归一化: `scripts/normalizer.py`
