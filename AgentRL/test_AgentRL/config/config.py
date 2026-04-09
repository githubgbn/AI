"""
AgentRL 配置文件
定义所有数据处理算法和模型配置
"""

# 缺失值处理算法
MISSING_VALUE_METHODS = {
    'delete': '删除缺失行',
    'mean': '均值填充',
    'median': '中位数填充',
    'forward_fill': '前向填充',
    'backward_fill': '后向填充',
    'interpolate': '插值填充',
    'knn': 'KNN填充',
    'multivariate_impute': '多重插补'
}

# 异常处理算法
ANOMALY_METHODS = {
    'zscore': 'Z-score方法',
    'iqr': 'IQR方法',
    'isolation_forest': '孤立森林',
    'percentile': '百分位法'
}

# 滤波降噪算法
FILTER_METHODS = {
    'median_filter': '中值滤波',
    'mean_filter': '均值滤波',
    'gaussian_filter': '高斯滤波',
    'kalman_filter': '卡尔曼滤波',
    'wavelet_denoise': '小波降噪'
}

# 差分算子算法
DIFF_METHODS = {
    'first_order': '一阶差分',
    'second_order': '二阶差分',
    'seasonal_diff': '季节性差分'
}

# 重采样算法
RESAMPLE_METHODS = {
    'upsample': '上采样',
    'downsample': '下采样',
    'interpolate_resample': '插值重采样',
    'smote': 'SMOTE过采样'
}

# 标准化/归一化算法
NORMALIZE_METHODS = {
    'minmax': 'MinMax归一化',
    'standard': 'Standard标准化',
    'zscore': 'Z-score标准化',
    'l2': 'L2归一化',
    'robust': 'RobustScaler'
}

# 预测模型
MODELS = {
    'random_forest': '随机森林',
    'lightgbm': 'LightGBM',
    'xgboost': 'XGBoost'
}

# 动作空间映射
ACTION_MAPPING = {
    'missing': list(MISSING_VALUE_METHODS.keys()),
    'anomaly': list(ANOMALY_METHODS.keys()),
    'filter': list(FILTER_METHODS.keys()),
    'diff': list(DIFF_METHODS.keys()),
    'resample': list(RESAMPLE_METHODS.keys()),
    'normalize': list(NORMALIZE_METHODS.keys())
}

# 强化学习配置
RL_CONFIG = {
    'n_steps': 512,
    'batch_size': 64,
    'n_epochs': 5,
    'gamma': 0.99,
    'learning_rate': 3e-4,
    'total_timesteps': 5000
}

# 模型训练配置
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'n_estimators': 100
}
