#!/usr/bin/env python3
"""创建带缺失值、噪声、异常值的模拟数据集用于验证"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)
n_samples = 200

# 生成特征数据
temperature = np.random.normal(25, 5, n_samples)
pressure = np.random.normal(1013, 20, n_samples)
humidity = np.random.normal(60, 10, n_samples)
vibration = np.random.normal(0.5, 0.1, n_samples)
current = np.random.normal(10, 2, n_samples)

# 生成目标标签（模拟二分类：设备正常/异常）
# 基于温度和压力的组合逻辑
y = ((temperature > 28) | (pressure < 995)).astype(int)

# 创建 DataFrame
df = pd.DataFrame({
    'temperature': temperature,
    'pressure': pressure,
    'humidity': humidity,
    'vibration': vibration,
    'current': current,
    'status': y
})

# 1. 添加缺失值（约 5%）
missing_cols = ['temperature', 'humidity', 'current']
for col in missing_cols:
    missing_idx = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
    df.loc[missing_idx, col] = np.nan

# 2. 添加噪声（随机抖动）
noise_idx = np.random.choice(df.index, size=20, replace=False)
df.loc[noise_idx, 'vibration'] += np.random.normal(0, 0.3, len(noise_idx))

# 3. 添加异常值
outlier_idx = np.random.choice(df.index, size=10, replace=False)
df.loc[outlier_idx, 'pressure'] = np.random.uniform(800, 900, len(outlier_idx))
outlier_idx2 = np.random.choice(df.index, size=8, replace=False)
df.loc[outlier_idx2, 'temperature'] = np.random.uniform(50, 60, len(outlier_idx2))

# 保存
df.to_csv('/mnt/workspace/gbn/project/infer_image_build/test_data.csv', index=False, encoding='utf-8-sig')
print(f"测试数据集已创建: test_data.csv")
print(f"数据形状: {df.shape}")
print(f"缺失值统计:\n{df.isnull().sum()}")
print(f"\n前5行:\n{df.head()}")
