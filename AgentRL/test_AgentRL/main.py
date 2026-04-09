"""
AgentRL 数据处理优化系统
主入口文件

本系统使用PPO强化学习自动优化数据处理流水线，以提升预测模型精度
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from config.config import RL_CONFIG, MODEL_CONFIG
from data.processor import DataProcessor
from env.rl_env import DataProcessingEnv
from agent.ppo_agent import PPOAgent
from models.predictor import Predictor


def load_sample_data(n_samples=1000, n_features=5, missing_rate=0.1, anomaly_rate=0.05, random_state=42):
    """生成示例数据集

    Args:
        n_samples: 样本数量
        n_features: 特征数量
        missing_rate: 缺失值比例
        anomaly_rate: 异常值比例
        random_state: 随机种子

    Returns:
        包含缺失值和异常值的示例数据框
    """
    np.random.seed(random_state)

    # 生成基础数据
    data = {}
    for i in range(n_features):
        data[f'feature{i+1}'] = np.random.randn(n_samples)

    # 添加非线性关系
    data['target'] = (
        data['feature1'] * 0.3 +
        data['feature2'] * 0.25 +
        data['feature3'] * 0.2 +
        data['feature4'] * 0.15 +
        data['feature5'] * 0.1 +
        np.random.randn(n_samples) * 0.1
    ) > 0

    df = pd.DataFrame(data)

    # 添加缺失值
    for col in [f'feature{i+1}' for i in range(n_features)]:
        mask = np.random.random(n_samples) < missing_rate
        df.loc[mask, col] = np.nan

    # 添加异常值
    for col in [f'feature{i+1}' for i in range(n_features)]:
        mask = np.random.random(n_samples) < anomaly_rate
        df.loc[mask, col] = df.loc[mask, col] * 5

    return df


def load_uci_data():
    """尝试加载UCI数据集（如果可用）"""
    try:
        from sklearn.datasets import load_breast_cancer, load_iris, load_wine
        import random

        datasets = [load_breast_cancer, load_iris, load_wine]
        dataset = random.choice(datasets)()

        df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        df['target'] = dataset.target

        # 添加一些缺失值
        for col in df.columns[:5]:
            mask = np.random.random(len(df)) < 0.05
            df.loc[mask, col] = np.nan

        return df
    except:
        return None


def print_header():
    """打印系统标题"""
    print("\n" + "=" * 70)
    print("                    AgentRL 数据处理优化系统                     ")
    print("=" * 70)
    print("基于PPO强化学习的数据处理流水线优化")
    print("=" * 70 + "\n")


def print_config_summary(rl_config, model_config):
    """打印配置摘要"""
    print("【强化学习配置】")
    print(f"  训练步数: {rl_config['total_timesteps']}")
    print(f"  折扣因子: {rl_config['gamma']}")
    print(f"  学习率: {rl_config['learning_rate']}")
    print(f"  批量大小: {rl_config['batch_size']}")
    print(f"  策略网络: MLP(64, 64)")
    print()


def main():
    """主函数"""
    print_header()
    print_config_summary(RL_CONFIG, MODEL_CONFIG)

    # 1. 加载数据
    print("【步骤 1/6】加载数据集...")
    target_column = 'target'

    # 尝试加载UCI数据，如果失败则使用示例数据
    df = load_uci_data()
    if df is None:
        df = load_sample_data(n_samples=1000, n_features=8, missing_rate=0.1, anomaly_rate=0.05)

    print(f"  数据集大小: {df.shape}")
    target_dist = df[target_column].value_counts().to_string().replace('\n', '\n  ')
    print(f"  目标变量分布:\n{target_dist}")
    print(f"  缺失值数量: {df.isnull().sum().sum()}")
    print()

    # 2. 创建RL环境
    print("【步骤 2/6】创建强化学习环境...")
    env = DataProcessingEnv(df, target_column, max_steps=6)
    print(f"  动作空间: {env.action_space}")
    print(f"  观察空间: {env.observation_space}")
    print()

    # 3. 计算baseline精度
    print("【步骤 3/6】计算baseline精度（默认处理配置）...")
    processor = DataProcessor()
    default_config = processor.get_default_config()
    processed_df = processor.process(df.copy(), default_config, target_column)

    predictor = Predictor(verbose=False)
    baseline_results = predictor.train_and_evaluate(processed_df, target_column)

    if baseline_results:
        baseline_rf = baseline_results.get('random_forest', {}).get('accuracy', 0)
        print(f"  Baseline随机森林精度: {baseline_rf:.4f}")
    else:
        baseline_rf = 0.5
        print("  Baseline计算失败，使用默认精度0.5")
    print()

    # 4. 训练PPO Agent
    print("【步骤 4/6】训练PPO Agent...")
    print(f"  这可能需要几分钟，请耐心等待...\n")

    agent = PPOAgent(
        env,
        n_steps=RL_CONFIG['n_steps'],
        batch_size=RL_CONFIG['batch_size'],
        n_epochs=RL_CONFIG['n_epochs'],
        gamma=RL_CONFIG['gamma'],
        learning_rate=RL_CONFIG['learning_rate'],
        verbose=0
    )

    agent.train(total_timesteps=RL_CONFIG['total_timesteps'])
    print()

    # 5. 获取最优配置
    print("【步骤 5/6】获取最优数据处理配置...")
    best_config = agent.predict(env)
    print(f"  最优配置: {best_config}")
    print()

    # 6. 使用最优配置评估所有模型
    print("【步骤 6/6】使用最优配置处理数据并评估所有模型...")
    best_processor = DataProcessor()
    best_processed_df = best_processor.process(df.copy(), best_config, target_column)

    if len(best_processed_df) < 10:
        print("  警告: 处理后数据量太少，使用默认配置")
        best_processed_df = processor.process(df.copy(), default_config, target_column)
        best_config = default_config

    final_predictor = Predictor(verbose=False)
    final_results = final_predictor.train_and_evaluate(best_processed_df, target_column)
    print()

    # 输出最终结果
    print("=" * 70)
    print("                         最终结果汇总")
    print("=" * 70)

    if baseline_results:
        print("\n【Baseline精度】")
        for model_name, metrics in baseline_results.items():
            print(f"  {model_name}: {metrics['accuracy']:.4f}")

    print(f"\n【最优处理配置】")
    for step, method in best_config.items():
        print(f"  {step}: {method}")

    print("\n【优化后精度】")
    for model_name, metrics in final_results.items():
        improvement = metrics['accuracy'] - baseline_results.get(model_name, {}).get('accuracy', 0)
        sign = '+' if improvement >= 0 else ''
        print(f"  {model_name}: {metrics['accuracy']:.4f} ({sign}{improvement:.4f})")

    # 最佳模型
    best_model, _, best_metrics = final_predictor.get_best_model()
    print(f"\n【最佳模型】: {best_model}")
    print(f"  准确率: {best_metrics['accuracy']:.4f}")
    print(f"  F1分数: {best_metrics['f1']:.4f}")
    print()

    # 精度提升总结
    total_improvement = 0
    for model_name in final_results:
        baseline_acc = baseline_results.get(model_name, {}).get('accuracy', 0)
        final_acc = final_results[model_name]['accuracy']
        total_improvement += (final_acc - baseline_acc)

    avg_improvement = total_improvement / len(final_results) if final_results else 0
    print(f"【平均精度提升】: {'+' if avg_improvement >= 0 else ''}{avg_improvement:.4f}")
    print("=" * 70)
    print()

    return best_config, final_results, baseline_results


if __name__ == "__main__":
    try:
        best_config, final_results, baseline_results = main()
        print("\n程序执行完成！")
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()
