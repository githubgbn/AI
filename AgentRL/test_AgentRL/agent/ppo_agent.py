"""
PPO强化学习Agent
使用Stable-Baselines3实现Proximal Policy Optimization算法
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import torch


class MetricsCallback(BaseCallback):
    """自定义回调，用于记录训练过程中的指标"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self):
        # 记录每步奖励
        if len(self.locals.get('rewards', [])) > 0:
            pass
        return True

    def _on_rollout_end(self):
        # 每个rollout结束时记录
        if self.verbose > 0:
            print(f"Rollout ended, mean reward: {np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0}")
        return True


class PPOAgent:
    """PPO强化学习Agent

    用于学习最优数据处理配置
    """

    def __init__(self, env, policy='MlpPolicy', n_steps=2048, batch_size=64,
                 n_epochs=10, gamma=0.99, learning_rate=3e-4, verbose=1):
        """初始化PPO Agent

        Args:
            env: Gymnasium环境
            policy: 策略网络类型 ('MlpPolicy' 或 'CnnPolicy')
            n_steps: 每次更新前收集的步数
            batch_size: 批量大小
            n_epochs: 每次更新的epoch数
            gamma: 折扣因子
            learning_rate: 学习率
            verbose: 日志详细程度
        """
        self.env = env
        self.verbose = verbose

        # 创建向量化环境
        self.vec_env = make_vec_env(lambda: env, n_envs=1)

        # 策略网络配置：两层MLP，每层64个神经元
        policy_kwargs = {
            'net_arch': [dict(pi=[64, 64], vf=[64, 64])],
            'activation_fn': torch.nn.ReLU
        }

        # 创建PPO模型
        self.model = PPO(
            policy=policy,
            env=self.vec_env,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=42,
            device='auto'
        )

    def train(self, total_timesteps=50000, callback=None):
        """训练Agent

        Args:
            total_timesteps: 总训练步数
            callback: 回调函数

        Returns:
            训练后的模型
        """
        if callback is None:
            callback = MetricsCallback(verbose=self.model.verbose)

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )

        return self.model

    def predict(self, env, deterministic=True):
        """使用训练好的模型预测最优动作序列

        Args:
            env: Gymnasium环境
            deterministic: 是否使用确定性策略

        Returns:
            最优配置
        """
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _states = self.model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

        if self.verbose > 0:
            print(f"Total reward: {total_reward:.4f}")
            print(f"Best config: {env.get_best_config()}")

        return env.get_best_config()

    def save(self, path):
        """保存模型"""
        self.model.save(path)

    def load(self, path):
        """加载模型"""
        self.model = PPO.load(path, env=self.vec_env)
