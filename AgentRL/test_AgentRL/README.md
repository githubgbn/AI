# AgentRL 数据处理优化系统

## 项目简介

本项目实现了一个基于PPO（Proximal Policy Optimization）强化学习的数据处理优化系统，用于自动学习最优的数据处理流水线配置，从而提升预测模型的精度。

## 核心功能

- **7种数据处理步骤**：缺失值处理、异常处理、滤波降噪、差分算子、重采样、标准化/归一化
- **强化学习优化**：使用PPO算法自动探索最优处理组合
- **多模型支持**：随机森林、LightGBM、XGBoost
- **可扩展设计**：易于添加新的处理算法和预测模型

## 算法支持

### 缺失值处理 (8种)
| 算法 | 说明 |
|------|------|
| delete | 删除缺失行 |
| mean | 均值填充 |
| median | 中位数填充 |
| forward_fill | 前向填充 |
| backward_fill | 后向填充 |
| interpolate | 线性插值 |
| knn | KNN填充 |
| multivariate_impute | 多重插补 |

### 异常处理 (4种)
| 算法 | 说明 |
|------|------|
| zscore | Z-score方法 |
| iqr | IQR四分位法 |
| isolation_forest | 孤立森林 |
| percentile | 百分位法 |

### 滤波降噪 (5种)
| 算法 | 说明 |
|------|------|
| median_filter | 中值滤波 |
| mean_filter | 均值滤波 |
| gaussian_filter | 高斯滤波 |
| kalman_filter | 卡尔曼滤波 |
| wavelet_denoise | 小波降噪 |

### 差分算子 (3种)
| 算法 | 说明 |
|------|------|
| first_order | 一阶差分 |
| second_order | 二阶差分 |
| seasonal_diff | 季节性差分 |

### 重采样 (4种)
| 算法 | 说明 |
|------|------|
| upsample | 上采样 |
| downsample | 下采样 |
| interpolate_resample | 插值重采样 |
| smote | SMOTE过采样 |

### 标准化/归一化 (5种)
| 算法 | 说明 |
|------|------|
| minmax | MinMax归一化 |
| standard | Standard标准化 |
| zscore | Z-score标准化 |
| l2 | L2归一化 |
| robust | RobustScaler |

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
cd test_AgentRL

# 处理CSV文件（自动检测目标列）
python main.py data.csv

# 指定目标列
python main.py data.csv --target label

# 指定训练步数
python main.py data.csv --timesteps 10000

# 使用示例数据
python main.py
```

## 输出示例

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

## 项目结构

```
test_AgentRL/
├── config/
│   └── config.py          # 配置文件
├── data/
│   └── processor.py       # 数据处理流水线
├── env/
│   └── rl_env.py          # 强化学习环境
├── agent/
│   └── ppo_agent.py       # PPO Agent
├── models/
│   └── predictor.py       # 预测模型
├── main.py               # 主入口
├── requirements.txt      # 依赖
└── README.md             # 说明文档
```

## 工作原理

1. **环境初始化**：加载数据集，创建强化学习环境
2. **Baseline计算**：使用默认配置处理数据，计算基础精度
3. **PPO训练**：Agent与环境交互，学习最优数据处理策略
4. **配置输出**：获取最优处理配置
5. **模型评估**：使用最优配置处理数据，评估所有模型精度

## 输出示例

```
============================================================
                    AgentRL 数据处理优化系统
============================================================

【步骤 1/6】加载数据集...
  数据集大小: (1000, 9)

【步骤 2/6】创建强化学习环境...

【步骤 3/6】计算baseline精度...

【步骤 4/6】训练PPO Agent...

【步骤 5/6】获取最优数据处理配置...
  最优配置: {...}

【步骤 6/6】使用最优配置处理数据并评估所有模型...

============================================================
                         最终结果汇总
============================================================

【Baseline精度】
  random_forest: 0.7650
  lightgbm: 0.7800
  xgboost: 0.7750

【最优处理配置】
  missing_method: knn
  anomaly_method: iqr
  ...

【优化后精度】
  random_forest: 0.8450 (+0.0800)
  lightgbm: 0.8600 (+0.0800)
  xgboost: 0.8550 (+0.0800)

【最佳模型】: lightgbm
  准确率: 0.8600
  F1分数: 0.8595

【平均精度提升】: +0.0800
============================================================
```

## 配置说明

在 `config/config.py` 中可以修改：

- `RL_CONFIG`: 强化学习参数（训练步数、学习率等）
- `MODEL_CONFIG`: 模型训练参数（测试集比例、树数量等）

## 扩展开发

### 添加新的数据处理算法

在 `data/processor.py` 中的对应方法添加新算法即可：

```python
def process_xxx(self, df, method='new_method'):
    if method == 'new_method':
        # 实现新算法
        pass
```

### 添加新的预测模型

在 `models/predictor.py` 中添加新模型训练方法：

```python
def train_new_model(self, X_train, X_test, y_train, y_test):
    # 实现新模型训练
    pass
```

## 依赖说明

- **stable-baselines3**: PPO算法实现
- **gymnasium**: 强化学习环境接口
- **scikit-learn**: 基础机器学习工具
- **lightgbm**: LightGBM模型
- **xgboost**: XGBoost模型
- **scipy/pywt**: 信号处理和滤波
- **imblearn**: SMOTE过采样

## License

MIT
