"""
强化学习环境
实现符合Gymnasium接口的DataProcessingEnv
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class DataProcessingEnv(gym.Env):
    """数据处理强化学习环境

    本环境将数据处理流水线的每个步骤建模为一个决策点，
    Agent通过选择不同算法来优化数据处理流程，最终提升预测精度。

    动作空间设计：
    - 步骤0 (缺失值处理): 8种算法 -> action 0-7
    - 步骤1 (异常处理): 4种算法 -> action 0-3
    - 步骤2 (滤波降噪): 5种算法 -> action 0-4
    - 步骤3 (差分算子): 3种算法 -> action 0-2
    - 步骤4 (重采样): 4种算法 -> action 0-3
    - 步骤5 (标准化): 5种算法 -> action 0-4
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, df, target_column, max_steps=6):
        """初始化环境

        Args:
            df: 原始数据框
            target_column: 目标变量列名
            max_steps: 处理步骤数量
        """
        super().__init__()

        self.original_df = df.copy()
        self.df = df.copy()
        self.target_column = target_column
        self.max_steps = max_steps

        # 各步骤的算法数量
        self.n_methods = [8, 4, 5, 3, 4, 5]

        # 动作空间：使用MultiDiscrete表示每个步骤的选择
        self.action_space = spaces.Discrete(29)  # 总共29种离散动作

        # 观察空间：数据统计特征 + 历史决策
        # 基础统计特征: 20维
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32
        )

        # 算法映射
        self.method_mappings = {
            0: ['delete', 'mean', 'median', 'forward_fill', 'backward_fill',
                'interpolate', 'knn', 'multivariate_impute'],
            1: ['zscore', 'iqr', 'isolation_forest', 'percentile'],
            2: ['median_filter', 'mean_filter', 'gaussian_filter',
                'kalman_filter', 'wavelet_denoise'],
            3: ['first_order', 'second_order', 'seasonal_diff'],
            4: ['downsample', 'upsample', 'interpolate_resample', 'smote'],
            5: ['minmax', 'standard', 'zscore', 'l2', 'robust']
        }

        # 状态变量
        self.current_step = 0
        self.config = {}
        self.baseline_accuracy = None
        self.current_obs = None

        # 导入处理器
        from data.processor import DataProcessor
        self.processor = DataProcessor()

    def _compute_data_features(self):
        """计算数据统计特征"""
        df = self.df
        features = []

        # 数值列统计
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_data = df[numeric_cols].values.flatten()
        else:
            numeric_data = np.array([0])

        # 基础统计
        features.append(np.nanmean(numeric_data) if len(numeric_data) > 0 else 0)
        features.append(np.nanstd(numeric_data) if len(numeric_data) > 0 else 0)
        features.append(np.nanmin(numeric_data) if len(numeric_data) > 0 else 0)
        features.append(np.nanmax(numeric_data) if len(numeric_data) > 0 else 0)

        # 缺失率
        missing_rate = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        features.append(missing_rate)

        # 偏度和峰度
        if len(numeric_cols) > 0 and len(df) > 2:
            from scipy.stats import skew, kurtosis
            try:
                features.append(skew(df[numeric_cols[0]].dropna()))
                features.append(kurtosis(df[numeric_cols[0]].dropna()))
            except:
                features.append(0)
                features.append(0)
        else:
            features.append(0)
            features.append(0)

        # 数据形状
        features.append(df.shape[0])
        features.append(df.shape[1])

        # 目标变量分布
        if self.target_column in df.columns:
            y = df[self.target_column]
            if len(y.unique()) > 1:
                features.append(y.value_counts().iloc[0] / len(y))
            else:
                features.append(0.5)
        else:
            features.append(0.5)

        # 数据量变化
        features.append(len(df) / len(self.original_df))

        # 当前步骤
        features.append(self.current_step)

        # 历史决策编码
        history = np.zeros(6)
        for i, step_config in enumerate(self.config.get('history', [])):
            if step_config < self.n_methods[i]:
                history[i] = step_config
        features.extend(history)

        # 填充到26维
        while len(features) < 26:
            features.append(0)

        return np.array(features[:26], dtype=np.float32)

    def _compute_baseline(self):
        """计算baseline精度"""
        if self.baseline_accuracy is not None:
            return self.baseline_accuracy

        try:
            df = self.processor.process(self.original_df.copy(),
                                       self.processor.get_default_config(),
                                       self.target_column)
            if len(df) < 10:
                self.baseline_accuracy = 0.5
                return self.baseline_accuracy

            X = df.drop(columns=[self.target_column], errors='ignore')
            y = df[self.target_column] if self.target_column in df.columns else df.iloc[:, -1]

            if len(X.columns) == 0 or len(np.unique(y)) < 2:
                self.baseline_accuracy = 0.5
                return self.baseline_accuracy

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            clf.fit(X_train, y_train)
            self.baseline_accuracy = accuracy_score(y_test, clf.predict(X_test))
        except:
            self.baseline_accuracy = 0.5

        return self.baseline_accuracy

    def _compute_reward(self):
        """计算奖励（测试集精度 - baseline精度）"""
        try:
            df = self.processor.process(self.original_df.copy(), self.config, self.target_column)
            if df is None or len(df) < 10:
                return -1.0

            X = df.drop(columns=[self.target_column], errors='ignore')
            y = df[self.target_column] if self.target_column in df.columns else df.iloc[:, -1]

            if len(X.columns) == 0 or len(np.unique(y)) < 2:
                return -1.0

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            clf.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, clf.predict(X_test))

            baseline = self._compute_baseline()
            reward = accuracy - baseline

            # 稀疏奖励设计：只在最后一步返回奖励
            return 0 if self.current_step < self.max_steps else reward

        except Exception as e:
            return -1.0

    def reset(self, seed=None):
        """重置环境"""
        super().reset(seed=seed)

        self.df = self.original_df.copy()
        self.current_step = 0
        self.config = {'history': []}
        self.baseline_accuracy = None

        self.current_obs = self._compute_data_features()

        return self.current_obs, {}

    def step(self, action):
        """执行动作

        Args:
            action: 动作索引 (0-28)

        Returns:
            observation: 新状态
            reward: 奖励
            terminated: 是否结束
            truncated: 是否截断
            info: 额外信息
        """
        # 将动作映射到当前步骤的算法
        step_methods = self.method_mappings[self.current_step]
        method_index = action % len(step_methods)
        method_name = step_methods[method_index]

        # 更新配置
        step_keys = ['missing_method', 'anomaly_method', 'filter_method',
                    'diff_method', 'resample_method', 'normalize_method']
        self.config[step_keys[self.current_step]] = method_name
        self.config['history'] = self.config.get('history', []) + [method_index]

        # 计算中间奖励（基于当前数据状态）
        intermediate_reward = 0
        if self.current_step > 0:
            try:
                temp_df = self.processor.process(self.original_df.copy(), self.config, self.target_column)
                if temp_df is not None and len(temp_df) >= 10:
                    X = temp_df.drop(columns=[self.target_column], errors='ignore')
                    y = temp_df[self.target_column] if self.target_column in temp_df.columns else temp_df.iloc[:, -1]
                    if len(X.columns) > 0 and len(np.unique(y)) >= 2:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )
                        clf = RandomForestClassifier(n_estimators=50, random_state=42)
                        clf.fit(X_train, y_train)
                        acc = accuracy_score(y_test, clf.predict(X_test))
                        baseline = self._compute_baseline()
                        intermediate_reward = (acc - baseline) * 0.1  # 小权重
            except:
                pass

        # 移动到下一步
        self.current_step += 1

        # 更新观察
        self.current_obs = self._compute_data_features()

        # 判断是否结束
        if self.current_step >= self.max_steps:
            final_reward = self._compute_reward()
            reward = final_reward
            terminated = True
        else:
            reward = intermediate_reward
            terminated = False

        truncated = False
        info = {'step': self.current_step, 'config': self.config}

        return self.current_obs, reward, terminated, truncated, info

    def get_best_config(self):
        """获取最优配置"""
        step_keys = ['missing_method', 'anomaly_method', 'filter_method',
                    'diff_method', 'resample_method', 'normalize_method']
        return {k: v for k, v in self.config.items() if k in step_keys}

    def render(self, mode='human'):
        """渲染环境状态"""
        print(f"Step: {self.current_step}/{self.max_steps}")
        print(f"Config: {self.get_best_config()}")
        print(f"Data shape: {self.df.shape}")
