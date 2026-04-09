"""
AgentRL 数据处理优化系统
主入口文件

本系统使用PPO强化学习自动优化数据处理流水线，以提升预测模型精度
用法: python main.py [csv_file_path]
"""

import sys
import os
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from config.config import RL_CONFIG
from data.processor import DataProcessor
from env.rl_env import DataProcessingEnv
from agent.ppo_agent import PPOAgent
from models.predictor import Predictor


def load_csv_data(csv_path):
    """加载CSV文件

    Args:
        csv_path: CSV文件路径

    Returns:
        DataFrame
    """
    df = pd.read_csv(csv_path)
    return df


def auto_detect_target(df):
    """自动检测目标列

    优先使用 'target' 列，否则使用最后一列
    """
    if 'target' in df.columns:
        return 'target'
    elif 'label' in df.columns:
        return 'label'
    elif 'y' in df.columns:
        return 'y'
    else:
        return df.columns[-1]


def print_header():
    """打印系统标题"""
    print("\n" + "=" * 70)
    print("                    AgentRL 数据处理优化系统                     ")
    print("=" * 70)
    print("基于PPO强化学习的数据处理流水线优化")
    print("=" * 70 + "\n")


def print_separator():
    """打印分隔线"""
    print("-" * 70)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AgentRL 数据处理优化系统')
    parser.add_argument('csv_file', nargs='?', default=None,
                        help='CSV文件路径（可选，不提供则使用示例数据）')
    parser.add_argument('--target', '-t', default=None,
                        help='目标列列名（可选，自动检测）')
    parser.add_argument('--timesteps', default=RL_CONFIG['total_timesteps'], type=int,
                        help=f'训练步数（默认: {RL_CONFIG["total_timesteps"]}）')
    args = parser.parse_args()

    print_header()

    # 1. 加载数据
    print("【步骤 1/5】加载数据...")

    if args.csv_file:
        if not os.path.exists(args.csv_file):
            print(f"  错误: 文件不存在 - {args.csv_file}")
            sys.exit(1)
        df = load_csv_data(args.csv_file)
        print(f"  数据来源: {args.csv_file}")
    else:
        # 生成示例数据
        np.random.seed(42)
        n_samples = 1000
        n_features = 8
        data = {}
        for i in range(n_features):
            data[f'feature{i+1}'] = np.random.randn(n_samples)
        data['target'] = (
            data['feature1'] * 0.3 +
            data['feature2'] * 0.25 +
            data['feature3'] * 0.2 +
            data['feature4'] * 0.15 +
            data['feature5'] * 0.1 +
            np.random.randn(n_samples) * 0.1
        ) > 0
        df = pd.DataFrame(data)

        # 添加缺失值和异常值
        for col in [f'feature{i+1}' for i in range(n_features)]:
            mask = np.random.random(n_samples) < 0.1
            df.loc[mask, col] = np.nan
        for col in [f'feature{i+1}' for i in range(n_features)]:
            mask = np.random.random(n_samples) < 0.05
            df.loc[mask, col] = df.loc[mask, col] * 5

        print(f"  数据来源: 自动生成的示例数据")

    target_column = args.target if args.target else auto_detect_target(df)

    print(f"  数据集大小: {df.shape}")
    print(f"  目标列: {target_column}")

    if target_column not in df.columns:
        print(f"  错误: 目标列 '{target_column}' 不存在")
        sys.exit(1)

    target_dist = df[target_column].value_counts()
    print(f"  目标变量分布: {dict(target_dist)}")
    print(f"  缺失值数量: {df.isnull().sum().sum()}")
    print()

    # 2. 创建RL环境
    print("【步骤 2/5】创建强化学习环境...")
    env = DataProcessingEnv(df, target_column, max_steps=6)
    print()

    # 3. 计算baseline精度（原始数据）
    print("【步骤 3/5】计算原始数据精度...")
    predictor_raw = Predictor(verbose=False)

    # 对原始数据进行简单处理（只删除缺失严重的行）
    df_clean = df.dropna(thresh=int(df.shape[1] * 0.5))
    if len(df_clean) < 10:
        df_clean = df.fillna(df.median())

    baseline_results = predictor_raw.train_and_evaluate(df_clean, target_column)

    if baseline_results:
        print("  【原始数据精度】")
        for model_name, metrics in baseline_results.items():
            print(f"    {model_name}: {metrics['accuracy']:.4f}")
    print()

    # 4. 训练PPO Agent
    print(f"【步骤 4/5】训练PPO Agent (timesteps={args.timesteps})...")
    print("  这可能需要几分钟，请耐心等待...\n")

    agent = PPOAgent(
        env,
        n_steps=RL_CONFIG['n_steps'],
        batch_size=RL_CONFIG['batch_size'],
        n_epochs=RL_CONFIG['n_epochs'],
        gamma=RL_CONFIG['gamma'],
        learning_rate=RL_CONFIG['learning_rate'],
        verbose=0
    )

    agent.train(total_timesteps=args.timesteps)
    print()

    # 5. 获取最优配置并评估
    print("【步骤 5/5】使用最优配置处理数据...")
    best_config = agent.predict(env)
    print()

    # 使用最优配置处理数据
    processor = DataProcessor()
    best_processed_df = processor.process(df.copy(), best_config, target_column)

    if len(best_processed_df) < 10:
        print("  警告: 处理后数据量太少")
        best_processed_df = df.fillna(df.median())

    final_predictor = Predictor(verbose=False)
    final_results = final_predictor.train_and_evaluate(best_processed_df, target_column)

    # 输出最终结果
    print("\n" + "=" * 70)
    print("                         结果汇总")
    print("=" * 70)

    print("\n【数据处理流水线配置】")
    step_names = {
        'missing_method': '缺失值处理',
        'anomaly_method': '异常处理',
        'filter_method': '滤波降噪',
        'diff_method': '差分算子',
        'resample_method': '重采样',
        'normalize_method': '标准化'
    }
    for step, method in best_config.items():
        if step in step_names:
            print(f"  {step_names[step]}: {method}")

    print("\n【精度对比】")
    print(f"  {'模型':<20} {'处理前':<12} {'处理后':<12} {'提升':<12}")
    print("  " + "-" * 56)

    total_before = 0
    total_after = 0
    for model_name in final_results:
        before = baseline_results.get(model_name, {}).get('accuracy', 0)
        after = final_results[model_name]['accuracy']
        improvement = after - before
        sign = '+' if improvement >= 0 else ''
        total_before += before
        total_after += after
        print(f"  {model_name:<20} {before:<12.4f} {after:<12.4f} {sign}{improvement:<11.4f}")

    avg_before = total_before / len(final_results) if final_results else 0
    avg_after = total_after / len(final_results) if final_results else 0
    avg_improvement = avg_after - avg_before
    sign = '+' if avg_improvement >= 0 else ''

    print("  " + "-" * 56)
    print(f"  {'平均':<20} {avg_before:<12.4f} {avg_after:<12.4f} {sign}{avg_improvement:<11.4f}")

    # 最佳模型
    best_model, _, best_metrics = final_predictor.get_best_model()
    print("\n【最佳模型】")
    print(f"  {best_model}: 准确率 {best_metrics['accuracy']:.4f}, F1 {best_metrics['f1']:.4f}")

    print("=" * 70)
    print()

    return best_config, final_results, baseline_results


if __name__ == "__main__":
    try:
        best_config, final_results, baseline_results = main()
        print("执行完成！")
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()