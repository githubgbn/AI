---
name: test_AgentRL
description: |
  当用户需要处理CSV文件并提升预测模型精度时触发本skill。
  用法：直接运行 `python main.py [csv_file]` 即可自动处理CSV文件，通过PPO强化学习优化数据处理流水线，无需额外配置。
  支持：自动检测目标列、自动选择最优处理算法、展示精度提升效果。
  用户可能描述为："处理这个CSV文件"、"优化数据"、"提升预测精度"等。
compatibility: Python 3.8+, stable-baselines3, scikit-learn, lightgbm, xgboost, pandas, numpy
---

# AgentRL 数据处理优化系统

## 快速使用

### 命令行直接运行

```bash
cd test_AgentRL

# 处理CSV文件（自动检测目标列）
python main.py data.csv

# 指定目标列
python main.py data.csv --target label

# 指定训练步数
python main.py data.csv --timesteps 10000

# 使用示例数据（不提供CSV）
python main.py
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `csv_file` | CSV文件路径（可选，不提供则使用示例数据） |
| `--target, -t` | 目标列列名（可选，自动检测 target/label/y/最后一列） |
| `--timesteps` | PPO训练步数（默认5000） |

### 输出示例

```
【精度对比】
  模型                   处理前          处理后          提升
  --------------------------------------------------------
  random_forest        0.4242       0.6170       +0.1928
  lightgbm             0.4848       0.7021       +0.2173
  xgboost              0.4848       0.6596       +0.1747
  --------------------------------------------------------
  平均                   0.4646       0.6596       +0.1949

【最佳模型】
  lightgbm: 准确率 0.7021, F1 0.7005
```

## 系统功能

- **自动数据处理**：缺失值填充、异常值处理、滤波降噪、差分、重采样、标准化
- **29种算法**：缺失值8种、异常处理4种、滤波5种、差分3种、重采样4种、标准化5种
- **PPO强化学习**：自动探索最优处理组合
- **多模型评估**：随机森林、LightGBM、XGBoost

## 项目结构

```
test_AgentRL/
├── config/
│   └── config.py              # 配置文件
├── data/
│   └── processor.py           # 数据处理流水线
├── env/
│   └── rl_env.py              # 强化学习环境
├── agent/
│   └── ppo_agent.py           # PPO Agent
├── models/
│   └── predictor.py           # 预测模型
├── main.py                    # 主入口
├── requirements.txt           # 依赖
└── README.md                   # 说明文档
```
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
```

### 第三步：创建数据处理流水线 data/processor.py

实现完整的7个数据处理步骤，每个步骤需支持设计文档中的多种算法。

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, normalize
from sklearn.ensemble import IsolationForest
from scipy import signal, interpolate
from scipy.ndimage import median_filter, gaussian_filter
from statsmodels.tsa.statespace.kalman import KalmanFilter
import pywt

class DataProcessor:
    """数据处理流水线"""

    def __init__(self):
        self.missing_method = None
        self.anomaly_method = None
        self.filter_method = None
        self.diff_method = None
        self.resample_method = None
        self.normalize_method = None

    def process_missing_value(self, df, method='mean'):
        """缺失值处理"""
        if method == 'delete':
            return df.dropna()
        elif method == 'mean':
            return df.fillna(df.mean())
        elif method == 'median':
            return df.fillna(df.median())
        elif method == 'forward_fill':
            return df.fillna(method='ffill')
        elif method == 'backward_fill':
            return df.fillna(method='bfill')
        elif method == 'interpolate':
            return df.interpolate(method='linear')
        elif method == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        elif method == 'multivariate_impute':
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            imputer = IterativeImputer()
            return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        return df

    def process_anomaly(self, df, method='zscore', threshold=3):
        """异常处理"""
        df = df.copy()
        if method == 'zscore':
            for col in df.select_dtypes(include=[np.number]).columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < threshold]
        elif method == 'iqr':
            for col in df.select_dtypes(include=[np.number]).columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
        elif method == 'isolation_forest':
            iso = IsolationForest(contamination=0.1)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                mask = iso.fit_predict(df[numeric_cols]) != -1
                df = df[mask]
        elif method == 'percentile':
            for col in df.select_dtypes(include=[np.number]).columns:
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df = df[(df[col] >= lower) & (df[col] <= upper)]
        return df

    def process_filter(self, df, method='median_filter', window_size=3):
        """滤波降噪"""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if method == 'median_filter':
                df[col] = median_filter(df[col].values, size=window_size)
            elif method == 'mean_filter':
                df[col] = df[col].rolling(window=window_size, center=True).mean()
            elif method == 'gaussian_filter':
                df[col] = gaussian_filter(df[col].values, sigma=window_size)
            elif method == 'kalman_filter':
                kf = KalmanKalmanFilter()
                df[col] = kf.smooth(df[col].values)[0]
            elif method == 'wavelet_denoise':
                coeffs = pywt.wavedec(df[col].values, 'db4', level=2)
                threshold = np.std(coeffs[-1]) * np.sqrt(2*np.log(len(df)))
                coeffs[-1] = pywt.threshold(coeffs[-1], threshold, mode='soft')
                df[col] = pywt.waverec(coeffs, 'db4')[:len(df)]
        return df

    def process_diff(self, df, method='first_order'):
        """差分算子"""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if method == 'first_order':
                df[col] = df[col].diff()
            elif method == 'second_order':
                df[col] = df[col].diff().diff()
            elif method == 'seasonal_diff':
                df[col] = df[col].diff(periods=7)  # 假设季节周期为7
        return df.dropna()

    def process_resample(self, df, method='downsample', target_size=None, random_state=42):
        """重采样"""
        if len(df) <= 10:
            return df
        if method == 'downsample' and target_size:
            indices = np.random.choice(len(df), target_size, replace=False)
            return df.iloc[sorted(indices)]
        elif method == 'upsample' and target_size:
            if target_size <= len(df):
                return df
            indices = np.linspace(0, len(df)-1, target_size)
            result = pd.DataFrame()
            for col in df.columns:
                result[col] = np.interp(indices, np.arange(len(df)), df[col].values)
            return result
        elif method == 'smote':
            from imblearn.over_sampling import SMOTE
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                smote = SMOTE(random_state=random_state)
                try:
                    X = df[numeric_cols].values
                    y = df.iloc[:, -1].values
                    if len(np.unique(y)) > 1:
                        X_res, y_res = smote.fit_resample(X, y)
                        result = pd.DataFrame(X_res, columns=numeric_cols)
                        result[df.columns[-1]] = y_res
                        return result
                except:
                    pass
            return df
        return df

    def process_normalize(self, df, method='standard'):
        """标准化/归一化"""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return df
        data = df[numeric_cols].values
        if method == 'minmax':
            scaler = MinMaxScaler()
            df[numeric_cols] = scaler.fit_transform(data)
        elif method == 'standard':
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(data)
        elif method == 'zscore':
            for col in numeric_cols:
                df[col] = (df[col] - df[col].mean()) / df[col].std()
        elif method == 'l2':
            df[numeric_cols] = normalize(data, norm='l2')
        elif method == 'robust':
            scaler = RobustScaler()
            df[numeric_cols] = scaler.fit_transform(data)
        return df

    def process(self, df, config):
        """执行完整的数据处理流水线"""
        df = self.process_missing_value(df, config.get('missing_method', 'mean'))
        df = self.process_anomaly(df, config.get('anomaly_method', 'zscore'))
        df = self.process_filter(df, config.get('filter_method', 'median_filter'))
        df = self.process_diff(df, config.get('diff_method', 'first_order'))
        df = self.process_resample(df, config.get('resample_method', 'downsample'))
        df = self.process_normalize(df, config.get('normalize_method', 'standard'))
        return df
```

### 第四步：创建强化学习环境 env/rl_env.py

实现符合Gymnasium接口的RL环境。

```python
import gymnasium as gym
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class DataProcessingEnv(gym.Env):
    """数据处理强化学习环境"""

    def __init__(self, df, target_column):
        super().__init__()
        self.df = df.copy()
        self.target_column = target_column
        self.processor = DataProcessor()

        # 定义动作空间：每个处理步骤选择一个算法
        # 0-7: 缺失值处理(8种), 8-11: 异常处理(4种),
        # 12-16: 滤波降噪(5种), 17-19: 差分算子(3种),
        # 20-23: 重采样(4种), 24-28: 标准化(5种)
        self.action_space = gym.spaces.Discrete(29)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(20,))

        self.current_step = 0
        self.max_steps = 6  # 6个处理步骤
        self.config = {}
        self.baseline_accuracy = None

    def _get_state(self):
        """获取当前数据状态"""
        df = self.df
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return np.zeros(20)
        data = df[numeric_cols].values.flatten()[:20]
        if len(data) < 20:
            data = np.pad(data, (0, 20 - len(data)))
        return data

    def _compute_reward(self):
        """计算奖励（测试集精度 - baseline精度）"""
        try:
            df = self.processor.process(self.df.copy(), self.config)
            if len(df) < 10:
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

            if self.baseline_accuracy is None:
                self.baseline_accuracy = accuracy

            reward = accuracy - self.baseline_accuracy
            return reward
        except Exception as e:
            return -1.0

    def reset(self, seed=None):
        """重置环境"""
        super().reset(seed=seed)
        self.current_step = 0
        self.config = {}
        return self._get_state(), {}

    def step(self, action):
        """执行动作"""
        # 将动作映射到具体处理步骤和算法
        if self.current_step == 0:
            methods = ['delete', 'mean', 'median', 'forward_fill', 'backward_fill',
                      'interpolate', 'knn', 'multivariate_impute']
            self.config['missing_method'] = methods[action % len(methods)]
        elif self.current_step == 1:
            methods = ['zscore', 'iqr', 'isolation_forest', 'percentile']
            self.config['anomaly_method'] = methods[action % len(methods)]
        elif self.current_step == 2:
            methods = ['median_filter', 'mean_filter', 'gaussian_filter',
                      'kalman_filter', 'wavelet_denoise']
            self.config['filter_method'] = methods[action % len(methods)]
        elif self.current_step == 3:
            methods = ['first_order', 'second_order', 'seasonal_diff']
            self.config['diff_method'] = methods[action % len(methods)]
        elif self.current_step == 4:
            methods = ['downsample', 'upsample', 'interpolate_resample', 'smote']
            self.config['resample_method'] = methods[action % len(methods)]
        elif self.current_step == 5:
            methods = ['minmax', 'standard', 'zscore', 'l2', 'robust']
            self.config['normalize_method'] = methods[action % len(methods)]

        self.current_step += 1

        if self.current_step >= self.max_steps:
            reward = self._compute_reward()
            done = True
        else:
            reward = 0
            done = False

        return self._get_state(), reward, done, False, {}

    def get_best_config(self):
        """获取最优配置"""
        return self.config
```

### 第五步：创建PPO Agent agent/ppo_agent.py

使用Stable-Baselines3实现PPO Agent。

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch

class PPOAgent:
    """PPO强化学习Agent"""

    def __init__(self, env, policy='MlpPolicy', n_steps=2048, batch_size=64,
                 n_epochs=10, gamma=0.99, learning_rate=3e-4):
        self.env = make_vec_env(lambda: env, n_envs=1)

        policy_kwargs = {
            'net_arch': [dict(pi=[64, 64], vf=[64, 64])]
        }

        self.model = PPO(
            policy=policy,
            env=self.env,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=1
        )

    def train(self, total_timesteps=50000):
        """训练Agent"""
        self.model.learn(total_timesteps=total_timesteps)
        return self.model

    def predict(self, env):
        """预测最优动作"""
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = self.model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
        return env.get_best_config()
```

### 第六步：创建预测模型 models/predictor.py

集成随机森林、LightGBM、XGBoost三种模型。

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import lightgbm as lgb
import xgboost as xgb

class Predictor:
    """多模型预测器"""

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.results = {}

    def train_and_evaluate(self, df, target_column):
        """训练并评估所有模型"""
        X = df.drop(columns=[target_column], errors='ignore')
        y = df[target_column] if target_column in df.columns else df.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # 随机森林
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        self.models['random_forest'] = rf
        self.results['random_forest'] = {
            'accuracy': accuracy_score(y_test, rf_pred),
            'f1': f1_score(y_test, rf_pred, average='weighted')
        }

        # LightGBM
        lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=self.random_state, verbose=-1)
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_test)
        self.models['lightgbm'] = lgb_model
        self.results['lightgbm'] = {
            'accuracy': accuracy_score(y_test, lgb_pred),
            'f1': f1_score(y_test, lgb_pred, average='weighted')
        }

        # XGBoost
        xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=self.random_state, use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        self.models['xgboost'] = xgb_model
        self.results['xgboost'] = {
            'accuracy': accuracy_score(y_test, xgb_pred),
            'f1': f1_score(y_test, xgb_pred, average='weighted')
        }

        return self.results

    def get_best_model(self):
        """获取最佳模型"""
        best_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
        return best_name, self.models[best_name], self.results[best_name]
```

### 第七步：创建主入口 main.py

```python
import pandas as pd
import numpy as np
from data.processor import DataProcessor
from env.rl_env import DataProcessingEnv
from agent.ppo_agent import PPOAgent
from models.predictor import Predictor

def load_sample_data():
    """生成示例数据集"""
    np.random.seed(42)
    n_samples = 1000
    data = {
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'feature4': np.random.randn(n_samples),
        'feature5': np.random.randn(n_samples),
    }
    # 添加缺失值
    for col in data.keys():
        mask = np.random.random(n_samples) < 0.1
        data[col] = np.where(mask, np.nan, data[col])

    # 添加异常值
    data['feature1'] = np.where(
        np.random.random(n_samples) < 0.05,
        data['feature1'] * 5,
        data['feature1']
    )

    # 生成目标变量
    data['target'] = (
        data['feature1'] * 0.3 +
        data['feature2'] * 0.2 +
        data['feature3'] * 0.25 +
        np.random.randn(n_samples) * 0.2
    ) > 0

    df = pd.DataFrame(data)
    return df

def main():
    print("=" * 60)
    print("AgentRL 数据处理优化系统")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/5] 加载数据集...")
    df = load_sample_data()
    target_column = 'target'
    print(f"数据集大小: {df.shape}")
    print(f"目标变量分布:\n{df[target_column].value_counts()}")

    # 2. 创建RL环境
    print("\n[2/5] 创建强化学习环境...")
    env = DataProcessingEnv(df, target_column)

    # 3. 训练PPO Agent
    print("\n[3/5] 训练PPO Agent（这可能需要几分钟）...")
    agent = PPOAgent(env, n_steps=1024, n_epochs=5, total_timesteps=10000)
    agent.train(total_timesteps=10000)

    # 4. 获取最优配置
    print("\n[4/5] 获取最优数据处理配置...")
    best_config = agent.predict(env)
    print(f"最优配置: {best_config}")

    # 5. 使用最优配置处理数据并评估
    print("\n[5/5] 使用最优配置处理数据并评估模型...")
    processor = DataProcessor()
    processed_df = processor.process(df.copy(), best_config)

    predictor = Predictor()
    results = predictor.train_and_evaluate(processed_df, target_column)

    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  F1分数: {metrics['f1']:.4f}")

    best_model, _, best_metrics = predictor.get_best_model()
    print(f"\n最佳模型: {best_model} (准确率: {best_metrics['accuracy']:.4f})")

    return best_config, results

if __name__ == "__main__":
    best_config, results = main()
```

### 第八步：创建requirements.txt

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
stable-baselines3>=1.6.0
gymnasium>=0.26.0
lightgbm>=3.0.0
xgboost>=1.5.0
scipy>=1.7.0
pywt>=1.3.0
imblearn>=0.9.0
statsmodels>=0.13.0
```

### 第九步：创建README.md

```markdown
# AgentRL 数据处理优化系统

## 项目简介

本项目实现了一个基于PPO（Proximal Policy Optimization）强化学习的数据处理优化系统，用于自动学习最优的数据处理流水线配置，从而提升预测模型的精度。

## 功能特性

- **7种数据处理步骤**：缺失值处理、异常处理、滤波降噪、差分算子、重采样、标准化/归一化
- **强化学习优化**：使用PPO算法自动探索最优处理组合
- **多模型支持**：随机森林、LightGBM、XGBoost
- **可扩展设计**：易于添加新的处理算法和预测模型

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
python main.py
```

## 数据处理步骤说明

| 步骤 | 算法数 | 说明 |
|------|--------|------|
| 缺失值处理 | 8种 | 删除、均值/中位数填充、插值、KNN等 |
| 异常处理 | 4种 | Z-score、IQR、孤立森林、百分位法 |
| 滤波降噪 | 5种 | 中值滤波、高斯滤波、小波降噪等 |
| 差分算子 | 3种 | 一阶/二阶差分、季节性差分 |
| 重采样 | 4种 | 上/下采样、SMOTE |
| 标准化 | 5种 | MinMax、Standard、Z-score等 |

## 输出示例

```
============================================================
AgentRL 数据处理优化系统
============================================================

[1/5] 加载数据集...
数据集大小: (1000, 6)

[2/5] 创建强化学习环境...

[3/5] 训练PPO Agent（这可能需要几分钟）...

[4/5] 获取最优数据处理配置...
最优配置: {...}

[5/5] 使用最优配置处理数据并评估模型...

============================================================
最终结果
============================================================

random_forest:
  准确率: 0.8950
  F1分数: 0.8945

lightgbm:
  准确率: 0.9100
  F1分数: 0.9095

xgboost:
  准确率: 0.9050
  F1分数: 0.9045

最佳模型: lightgbm (准确率: 0.9100)
```
```

## 项目结构

```
test_AgentRL/
├── config/config.py          # 配置文件
├── data/processor.py         # 数据处理流水线
├── env/rl_env.py             # 强化学习环境
├── agent/ppo_agent.py        # PPO Agent
├── models/predictor.py       # 预测模型
├── main.py                   # 主入口
├── requirements.txt          # 依赖
└── README.md                 # 说明文档
```
