"""
测试数据
"""
import pandas as pd
import numpy as np

# 创建测试数据
def create_test_data():
    """创建测试数据集"""
    np.random.seed(42)

    n = 100

    df = pd.DataFrame({
        'id': range(1, n + 1),
        'name': [f'User_{i}' for i in range(n)],
        'age': np.random.randint(20, 60, n).astype(float),
        'salary': np.random.randint(3000, 30000, n).astype(float),
        'department': np.random.choice(['Sales', 'IT', 'HR', 'Finance', 'Marketing'], n),
        'score': np.random.randn(n) * 20 + 80
    })

    # 添加缺失值
    df.loc[0:4, 'age'] = np.nan
    df.loc[10:14, 'salary'] = np.nan
    df.loc[20:21, 'department'] = np.nan

    # 添加异常值
    df.loc[0, 'salary'] = 100000  # 异常高收入
    df.loc[1, 'salary'] = -5000    # 异常负收入
    df.loc[2, 'age'] = 150         # 异常年龄

    return df


def save_test_data(df, path='test_data.csv'):
    """保存测试数据"""
    df.to_csv(path, index=False)
    print(f"测试数据已保存至: {path}")
    return path


if __name__ == '__main__':
    df = create_test_data()
    save_test_data(df, '/mnt/workspace/gbn/project/zpy/test_data.csv')
    print(f"\n数据概览:\n{df.head(10)}")
    print(f"\n数据形状: {df.shape}")
    print(f"\n缺失值统计:\n{df.isnull().sum()}")
