"""
创建测试数据集 - 模拟真实数据场景
包含：缺失值、噪声、异常值、分类变量等
"""

import pandas as pd
import numpy as np
import os

# 设置随机种子
np.random.seed(42)

# 创建测试数据
n = 200  # 样本数量

# 1. 基础时间序列数据
dates = pd.date_range('2024-01-01', periods=n, freq='D')

# 2. 先创建基础特征
temperature = np.linspace(10, 30, n) + np.random.normal(0, 2, n)
sales = np.exp(np.linspace(0, 3, n)) * 100 + np.random.normal(0, 50, n)
age = np.random.randint(18, 80, n).astype(float)
income = np.random.lognormal(10, 1, n)
score = np.random.normal(70, 15, n)

# 3. 创建包含各种特征的数据
data = {
    # 时间序列特征
    'date': dates,
    'temperature': temperature,
    'sales': sales,
    'pressure': 100 + np.sin(np.linspace(0, 10*np.pi, n)) * 10 + np.random.normal(0, 1, n),

    # 数值特征（用于填充测试）
    'age': age,
    'income': income,
    'score': score,

    # 添加缺失值的副本
    'temperature_missing': temperature.copy(),
    'sales_missing': sales.copy(),
    'age_missing': age.copy(),
    'income_missing': income.copy(),
}

df = pd.DataFrame(data)

# 3. 添加缺失值
# 随机缺失（10%）
missing_indices_temp = np.random.choice(n, 20, replace=False)
missing_indices_sales = np.random.choice(n, 15, replace=False)
missing_indices_age = np.random.choice(n, 25, replace=False)
missing_indices_income = np.random.choice(n, 30, replace=False)

df.loc[missing_indices_temp, 'temperature_missing'] = np.nan
df.loc[missing_indices_sales, 'sales_missing'] = np.nan
df.loc[missing_indices_age, 'age_missing'] = np.nan
df.loc[missing_indices_income, 'income_missing'] = np.nan

# 4. 添加异常值
outlier_indices_1 = np.random.choice(n, 10, replace=False)
outlier_indices_2 = np.random.choice(n, 8, replace=False)

# 极端高值
df.loc[outlier_indices_1[:5], 'score'] = df.loc[outlier_indices_1[:5], 'score'] * 5
# 极端低值
df.loc[outlier_indices_2[:4], 'income'] = df.loc[outlier_indices_2[:4], 'income'] * 0.01

# 5. 添加分类变量
df['category'] = np.random.choice(['A', 'B', 'C', 'D'], n)
df['city'] = np.random.choice(['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen'], n)
df['status'] = np.random.choice(['Active', 'Inactive', 'Pending'], n)

# 6. 添加偏态分布数据（用于归一化测试）
df['skewed_positive'] = np.random.exponential(10, n)  # 右偏
df['skewed_negative'] = -np.random.exponential(10, n)  # 左偏

# 7. 添加强噪声数据（用于滤波测试）
df['noisy_signal'] = np.sin(np.linspace(0, 4*np.pi, n)) * 50 + np.random.normal(0, 20, n)

# 8. 添加需要差分的数据（趋势数据）
df['trend'] = np.linspace(0, 1000, n) + np.random.normal(0, 10, n)

# 保存测试数据
output_path = '/mnt/workspace/gbn/project/test_invoke_data_method/test_dataset.csv'
df.to_csv(output_path, index=False)

print(f"测试数据集已创建: {output_path}")
print(f"数据形状: {df.shape}")
print(f"\n数据列:")
for col in df.columns:
    missing = df[col].isnull().sum()
    print(f"  {col}: {missing} 缺失值 ({missing/len(df)*100:.1f}%)")

print(f"\n前5行数据:")
print(df.head())
