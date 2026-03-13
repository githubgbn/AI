# invoke_data_processor_method Skill 设计文档

## 概述

创建一个数据处理Python库，支持7大类数据处理算子，包括缺失值处理、滤波降噪、数据替换、异常处理、差分算子、重采样和标准化/归一化。

## 架构设计

### 核心类结构
```
DataProcessor (主类)
├── MissingValueHandler (缺失值处理)
├── FilterDenoiser (滤波降噪)
├── DataTransformer (数据替换)
├── OutlierHandler (异常处理)
├── DifferentialOperator (差分算子)
├── Resampler (重采样)
└── Normalizer (标准化/归一化)
```

### 数据流
1. 输入：原始DataFrame
2. 处理：通过各算子方法逐步处理
3. 输出：处理后的DataFrame
4. 保存：调用save_to_csv方法导出CSV

## 算子详情

### 1. 缺失值处理 (MissingValueHandler)
- delete_row: 行删除
- delete_column: 列删除
- fill_forward: 前项填充
- fill_backward: 后向填充
- fill_mean: 均值填充
- fill_median: 中位数填充
- fill_constant: 固定值填充
- fill_mode: 众数填充
- interpolate_linear: 线性插值
- interpolate_spline: 样条插值

### 2. 滤波降噪 (FilterDenoiser)
- median_filter: 中值滤波
- moving_average: 移动平均
- exponential_moving_average: 指数加权移动平均
- lowpass_filter: 低通滤波
- fourier_transform: 傅里叶变换

### 3. 数据替换 (DataTransformer)
- log_transform: 对数变换
- diff_transform: 差分变换
- standardize: 标准化
- category_encode: 类别编码替换

### 4. 异常处理 (OutlierHandler)
- three_sigma: 3-Sigma方法
- quartile_method: 四分位距法
- clustering_detection: 聚类检测
- moving_std: 移动标准差法
- zscore: Z-score

### 5. 差分算子 (DifferentialOperator)
- first_order_diff: 普通一阶差分
- lag_diff: 滞后k阶差分
- second_order_diff: 二阶差分
- n_order_diff: n阶差分
- seasonal_diff: 季节性差分
- log_diff: 对数一阶差分
- percent_diff: 相对变化率

### 6. 重采样 (Resampler)
- upsample_linear: 线性插值上采样
- upsample_polynomial: 多项式插值上采样
- upsample_spline: 样条插值上采样
- downsample_mean: 均值聚合下采样
- downsample_sum: 求和聚合下采样

### 7. 标准化/归一化 (Normalizer)
- zscore_standardize: Z-Score标准化
- mean_center: 均值中心化
- robust_standardize: 稳健标准化
- minmax_normalize: Min-Max归一化
- range_normalize: 指定区间缩放
- maxabs_normalize: 最大绝对值归一化
- log_normalize: 对数变换归一化
- boxcox_transform: Box-Cox变换

## 使用示例

```python
processor = DataProcessor()

# 缺失值填充
df = processor.missing_value.fill_forward(df)
df = processor.missing_value.interpolate_linear(df)

# 滤波降噪
df = processor.filter_denoiser.moving_average(df, window=5)
df = processor.filter_denoiser.median_filter(df, kernel_size=3)

# 数据替换
df = processor.data_transformer.log_transform(df)

# 异常处理
df = processor.outlier_handler.three_sigma(df)

# 差分算子
df = processor.differential_operator.first_order_diff(df)

# 重采样
df = processor.resampler.upsample_linear(df, factor=2)

# 标准化
df = processor.normalizer.zscore_standardize(df)

# 保存结果
processor.save_to_csv(df, 'output.csv')
```

## 验收标准

1. 所有算子方法都能接收DataFrame并返回DataFrame
2. 支持指定列处理或自动处理所有数值列
3. 每个方法包含详细的使用场景docstring
4. 最后可通过save_to_csv导出CSV
5. 代码兼容pandas和numpy
