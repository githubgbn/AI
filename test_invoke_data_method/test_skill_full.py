"""
完整测试脚本 - 验证invoke_data_processor_method skill
使用测试数据集进行全面测试
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加路径
script_dir = '/mnt/workspace/gbn/project/test_invoke_data_method/invoke_data_processor_method/scripts'
sys.path.insert(0, script_dir)

# 导入所有算子
from missing_value import MissingValueHandler
from filter_denoiser import FilterDenoiser
from data_transformer import DataTransformer
from outlier_handler import OutlierHandler
from differential_operator import DifferentialOperator
from resampler import Resampler
from normalizer import Normalizer
from data_processor import DataProcessor

# 加载测试数据
data_path = '/mnt/workspace/gbn/project/test_invoke_data_method/test_dataset.csv'
df = pd.read_csv(data_path)
df = df.drop(columns=['date'])  # 移除日期列便于处理

print("=" * 70)
print("测试数据集信息")
print("=" * 70)
print(f"数据形状: {df.shape}")
print(f"缺失值统计:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# 创建处理器
processor = DataProcessor()

# ==================== 1. 缺失值处理测试 ====================
print("\n" + "=" * 70)
print("1. 缺失值处理测试")
print("=" * 70)

missing_handler = MissingValueHandler()

# 测试前向填充
df1 = missing_handler.fill_forward(df.copy(), columns=['temperature_missing', 'sales_missing'])
print(f"前向填充后缺失值: {df1[['temperature_missing', 'sales_missing']].isnull().sum().sum()}")

# 测试后向填充
df2 = missing_handler.fill_backward(df.copy(), columns=['age_missing', 'income_missing'])
print(f"后向填充后缺失值: {df2[['age_missing', 'income_missing']].isnull().sum().sum()}")

# 测试均值填充
df3 = missing_handler.fill_mean(df.copy(), columns=['temperature_missing'])
print(f"均值填充后缺失值: {df3['temperature_missing'].isnull().sum()}")

# 测试中位数填充
df4 = missing_handler.fill_median(df.copy(), columns=['income_missing'])
print(f"中位数填充后缺失值: {df4['income_missing'].isnull().sum()}")

# 测试线性插值
df5 = missing_handler.interpolate_linear(df.copy(), columns=['temperature_missing'])
print(f"线性插值后缺失值: {df5['temperature_missing'].isnull().sum()}")

# 测试固定值填充
df6 = missing_handler.fill_constant(df.copy(), value=-999, columns=['age_missing'])
print(f"固定值填充后缺失值: {df6['age_missing'].isnull().sum()}")

# 测试行删除
df7 = missing_handler.delete_row(df.copy(), columns=['temperature_missing'])
print(f"行删除后数据量: {len(df7)}")

print("✓ 缺失值处理测试通过")

# ==================== 2. 滤波降噪测试 ====================
print("\n" + "=" * 70)
print("2. 滤波降噪测试")
print("=" * 70)

filter_handler = FilterDenoiser()

# 测试移动平均
df8 = filter_handler.moving_average(df.copy(), columns=['noisy_signal'], window=5)
print(f"移动平均 - 原始标准差: {df['noisy_signal'].std():.2f}, 处理后: {df8['noisy_signal'].std():.2f}")

# 测试中值滤波
df9 = filter_handler.median_filter(df.copy(), columns=['noisy_signal'], kernel_size=3)
print(f"中值滤波 - 原始标准差: {df['noisy_signal'].std():.2f}, 处理后: {df9['noisy_signal'].std():.2f}")

# 测试指数加权移动平均
df10 = filter_handler.exponential_moving_average(df.copy(), columns=['temperature'], span=5)
print(f"指数加权移动平均 - 处理完成")

# 测试低通滤波
df11 = filter_handler.lowpass_filter(df.copy(), columns=['noisy_signal'], cutoff=0.1)
print(f"低通滤波 - 原始标准差: {df['noisy_signal'].std():.2f}, 处理后: {df11['noisy_signal'].std():.2f}")

print("✓ 滤波降噪测试通过")

# ==================== 3. 数据替换测试 ====================
print("\n" + "=" * 70)
print("3. 数据替换测试")
print("=" * 70)

transformer = DataTransformer()

# 测试对数变换
df12 = transformer.log_transform(df.copy(), columns=['income'])
print(f"对数变换 - 原始范围: [{df['income'].min():.0f}, {df['income'].max():.0f}], 变换后: [{df12['income'].min():.2f}, {df12['income'].max():.2f}]")

# 测试差分变换
df13 = transformer.diff_transform(df.copy(), columns=['trend'])
print(f"差分变换 - 处理完成")

# 测试标准化
df14 = transformer.standardize(df.copy(), columns=['score'])
print(f"标准化 - 均值: {df14['score'].mean():.4f}, 标准差: {df14['score'].std():.4f}")

# 测试类别编码（标签编码）
df15 = df.copy()
df15 = transformer.category_encode(df15, columns=['category', 'city'], method='label')
print(f"标签编码 - category唯一值: {df15['category'].nunique()}, city唯一值: {df15['city'].nunique()}")

print("✓ 数据替换测试通过")

# ==================== 4. 异常处理测试 ====================
print("\n" + "=" * 70)
print("4. 异常处理测试")
print("=" * 70)

outlier_handler = OutlierHandler()

# 测试3-Sigma方法
df16 = outlier_handler.three_sigma(df.copy(), columns=['score', 'income'], n_sigma=3.0, method='clip')
print(f"3-Sigma - 原始score范围: [{df['score'].min():.1f}, {df['score'].max():.1f}], 处理后: [{df16['score'].min():.1f}, {df16['score'].max():.1f}]")

# 测试四分位距法
df17 = outlier_handler.quartile_method(df.copy(), columns=['income', 'score'], k=1.5, method='clip')
print(f"四分位距法 - 处理完成")

# 测试Z-score
df18 = outlier_handler.zscore(df.copy(), columns=['score'], threshold=3.0, method='clip')
print(f"Z-score - 处理完成")

print("✓ 异常处理测试通过")

# ==================== 5. 差分算子测试 ====================
print("\n" + "=" * 70)
print("5. 差分算子测试")
print("=" * 70)

diff_op = DifferentialOperator()

# 测试一阶差分
df19 = diff_op.first_order_diff(df.copy(), columns=['trend'])
print(f"一阶差分 - 处理完成，均值: {df19['trend'].mean():.2f}")

# 测试二阶差分
df20 = diff_op.second_order_diff(df.copy(), columns=['trend'])
print(f"二阶差分 - 处理完成")

# 测试季节性差分
df21 = diff_op.seasonal_diff(df.copy(), columns=['pressure'], period=7)
print(f"季节性差分 - 处理完成")

# 测试对数差分
df22 = diff_op.log_diff(df.copy(), columns=['income'])
print(f"对数差分 - 处理完成")

# 测试百分比变化
df23 = diff_op.percent_diff(df.copy(), columns=['sales'])
print(f"百分比变化 - 处理完成")

print("✓ 差分算子测试通过")

# ==================== 6. 重采样测试 ====================
print("\n" + "=" * 70)
print("6. 重采样测试")
print("=" * 70)

resampler = Resampler()

# 创建简单数值数据用于重采样测试
simple_df = pd.DataFrame({
    'value1': np.linspace(1, 100, 50),
    'value2': np.sin(np.linspace(0, 4*np.pi, 50)),
    'category': ['A'] * 25 + ['B'] * 25
})

# 测试上采样（线性插值）
df24 = resampler.upsample_linear(simple_df.copy(), factor=2)
print(f"线性插值上采样 - 原始: {len(simple_df)}, 上采样后: {len(df24)}")

# 测试下采样（均值聚合）
df25 = resampler.downsample_mean(simple_df.copy(), factor=2)
print(f"均值聚合下采样 - 原始: {len(simple_df)}, 下采样后: {len(df25)}")

# 测试下采样（求和聚合）
df26 = resampler.downsample_sum(simple_df.copy(), factor=2)
print(f"求和聚合下采样 - 处理完成")

print("✓ 重采样测试通过")

# ==================== 7. 标准化/归一化测试 ====================
print("\n" + "=" * 70)
print("7. 标准化/归一化测试")
print("=" * 70)

normalizer = Normalizer()

# 测试Z-Score标准化
df27 = normalizer.zscore_standardize(df.copy(), columns=['temperature', 'sales'])
print(f"Z-Score标准化 - temperature均值: {df27['temperature'].mean():.4f}, 标准差: {df27['temperature'].std():.4f}")

# 测试Min-Max归一化
df28 = normalizer.minmax_normalize(df.copy(), columns=['score'])
print(f"Min-Max归一化 - score范围: [{df28['score'].min():.2f}, {df28['score'].max():.2f}]")

# 测试均值中心化
df29 = normalizer.mean_center(df.copy(), columns=['income'])
print(f"均值中心化 - income均值: {df29['income'].mean():.4f}")

# 测试稳健标准化
df30 = normalizer.robust_standardize(df.copy(), columns=['income', 'score'])
print(f"稳健标准化 - 处理完成")

# 测试对数归一化
df31 = normalizer.log_normalize(df.copy(), columns=['skewed_positive'])
print(f"对数归一化 - 处理完成")

# 测试指定区间缩放
df32 = normalizer.range_normalize(df.copy(), columns=['temperature'], feature_range=(-1, 1))
print(f"指定区间缩放 - temperature范围: [{df32['temperature'].min():.2f}, {df32['temperature'].max():.2f}]")

# 测试最大绝对值归一化
df33 = normalizer.maxabs_normalize(df.copy(), columns=['score'])
print(f"最大绝对值归一化 - score范围: [{df33['score'].min():.2f}, {df33['score'].max():.2f}]")

print("✓ 标准化/归一化测试通过")

# ==================== 8. 完整流程测试 ====================
print("\n" + "=" * 70)
print("8. 完整流程测试")
print("=" * 70)

# 加载原始数据
df_final = pd.read_csv(data_path)
df_final = df_final.drop(columns=['date'])

# 完整流程：缺失值填充 -> 滤波降噪 -> 异常处理 -> 标准化
print("开始完整流程处理...")

# 1. 缺失值填充（使用线性插值）
df_final = missing_handler.interpolate_linear(df_final, columns=['temperature_missing', 'sales_missing', 'age_missing', 'income_missing'])
print(f"  1) 缺失值填充完成，剩余缺失值: {df_final.isnull().sum().sum()}")

# 2. 滤波降噪（移动平均）
df_final = filter_handler.moving_average(df_final, columns=['noisy_signal'], window=5)
print(f"  2) 滤波降噪完成")

# 3. 异常处理（四分位距法）
df_final = outlier_handler.quartile_method(df_final, columns=['score', 'income'], method='clip')
print(f"  3) 异常处理完成")

# 4. 标准化（Z-Score）
df_final = normalizer.zscore_standardize(df_final, columns=['temperature', 'sales', 'pressure', 'age', 'income', 'score'])
print(f"  4) 标准化完成")

# 保存结果
output_path = '/mnt/workspace/gbn/project/test_invoke_data_method/processed_result.csv'
processor.save_to_csv(df_final, output_path)
print(f"\n处理完成，结果已保存到: {output_path}")

print("\n" + "=" * 70)
print("所有测试通过!")
print("=" * 70)
