#!/usr/bin/env python3
"""训练 sklearn 模型并保存为 .pkl 文件"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import sys
import os

# 添加 references 目录到路径（使 Python 能找到 data_process_policy.py 模块）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data_process_infer_image_build', 'references'))
import data_process_policy
deal_data = data_process_policy.deal_data

# 读取原始数据
df = pd.read_csv('/mnt/workspace/gbn/project/infer_image_build/test_data.csv')
print(f"原始数据形状: {df.shape}")
print(f"原始缺失值:\n{df.isnull().sum()}")

# 数据清洗
cleaned_df = deal_data(df)
print(f"\n清洗后数据形状: {cleaned_df.shape}")

# 分离特征和标签
X = cleaned_df.drop(columns=['status'])
y = cleaned_df['status']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# 评估
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"\n训练集准确率: {train_score:.4f}")
print(f"测试集准确率: {test_score:.4f}")

# 保存模型
model_path = '/mnt/workspace/gbn/project/infer_image_build/data_process_infer_image_build/references/rf_model.pkl'
joblib.dump(model, model_path)
print(f"\n模型已保存: {model_path}")

# 保存清洗后的测试数据（用于和推理结果对比）
cleaned_df.to_csv('/mnt/workspace/gbn/project/infer_image_build/test_data_cleaned.csv', index=False, encoding='utf-8-sig')
print(f"清洗后数据已保存: test_data_cleaned.csv")
