"""
测试invoke_data_processor_method skill
"""
import sys
import os
import pandas as pd
import numpy as np

# 添加路径
script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')
sys.path.insert(0, script_dir)

from data_processor import DataProcessor

# 创建测试数据
def create_test_data():
    """创建包含缺失值和噪声的测试数据"""
    np.random.seed(42)
    n = 100

    # 创建基础数据
    data = {
        'date': pd.date_range('2024-01-01', periods=n, freq='D'),
        'value1': np.linspace(10, 100, n) + np.random.normal(0, 5, n),
        'value2': np.exp(np.linspace(0, 2, n)) + np.random.normal(0, 10, n),
        'value3': np.sin(np.linspace(0, 4*np.pi, n)) * 50 + 50 + np.random.normal(0, 3, n),
    }

    df = pd.DataFrame(data)

    # 添加一些缺失值
    missing_indices = np.random.choice(n, 10, replace=False)
    df.loc[missing_indices, 'value1'] = np.nan

    # 添加一些异常值
    outlier_indices = np.random.choice(n, 5, replace=False)
    df.loc[outlier_indices, 'value2'] = df.loc[outlier_indices, 'value2'] * 3

    return df

def test_missing_value():
    """测试缺失值处理"""
    print("=" * 50)
    print("测试缺失值处理")
    print("=" * 50)

    processor = DataProcessor()
    df = create_test_data()

    print(f"原始缺失值:\n{df.isnull().sum()}")

    # 测试前向填充
    df1 = processor.missing_value.fill_forward(df.copy())
    print(f"\n前向填充后缺失值:\n{df1.isnull().sum()}")

    # 测试均值填充
    df2 = processor.missing_value.fill_mean(df.copy())
    print(f"\n均值填充后缺失值:\n{df2.isnull().sum()}")

    # 测试线性插值
    df3 = processor.missing_value.interpolate_linear(df.copy())
    print(f"\n线性插值后缺失值:\n{df3.isnull().sum()}")

    print("\n缺失值处理测试通过!")

def test_filter_denoiser():
    """测试滤波降噪"""
    print("\n" + "=" * 50)
    print("测试滤波降噪")
    print("=" * 50)

    processor = DataProcessor()
    df = create_test_data()

    # 测试移动平均
    df1 = processor.filter_denoiser.moving_average(df.copy(), window=5)
    print(f"移动平均后:\n{df1[['value1', 'value2', 'value3']].describe()}")

    # 测试中值滤波
    df2 = processor.filter_denoiser.median_filter(df.copy(), kernel_size=3)
    print(f"\n中值滤波后:\n{df2[['value1', 'value2', 'value3']].describe()}")

    print("\n滤波降噪测试通过!")

def test_outlier_handler():
    """测试异常处理"""
    print("\n" + "=" * 50)
    print("测试异常处理")
    print("=" * 50)

    processor = DataProcessor()
    df = create_test_data()

    # 测试3-Sigma方法
    df1 = processor.outlier_handler.three_sigma(df.copy(), n_sigma=3.0, method='clip')
    print(f"3-Sigma处理后:\n{df1[['value1', 'value2', 'value3']].describe()}")

    # 测试四分位距法
    df2 = processor.outlier_handler.quartile_method(df.copy(), k=1.5, method='clip')
    print(f"\n四分位距法处理后:\n{df2[['value1', 'value2', 'value3']].describe()}")

    print("\n异常处理测试通过!")

def test_normalizer():
    """测试标准化/归一化"""
    print("\n" + "=" * 50)
    print("测试标准化/归一化")
    print("=" * 50)

    processor = DataProcessor()
    df = create_test_data()

    # 测试Z-Score标准化
    df1 = processor.normalizer.zscore_standardize(df.copy())
    print(f"Z-Score标准化后:\n{df1[['value1', 'value2', 'value3']].describe()}")

    # 测试Min-Max归一化
    df2 = processor.normalizer.minmax_normalize(df.copy())
    print(f"\nMin-Max归一化后:\n{df2[['value1', 'value2', 'value3']].describe()}")

    print("\n标准化/归一化测试通过!")

def test_full_pipeline():
    """测试完整处理流程"""
    print("\n" + "=" * 50)
    print("测试完整处理流程")
    print("=" * 50)

    processor = DataProcessor()
    df = create_test_data()

    # 完整流程：缺失值填充 -> 滤波降噪 -> 异常处理 -> 标准化
    df = processor.missing_value.interpolate_linear(df)
    df = processor.filter_denoiser.moving_average(df, window=5)
    df = processor.outlier_handler.quartile_method(df, method='clip')
    df = processor.normalizer.zscore_standardize(df)

    print(f"处理后数据:\n{df[['value1', 'value2', 'value3']].describe()}")

    # 保存结果
    output_path = '/mnt/workspace/gbn/project/test_invoke_data_method/test_output.csv'
    processor.save_to_csv(df, output_path)
    print(f"\n结果已保存到: {output_path}")

    print("\n完整流程测试通过!")

if __name__ == '__main__':
    test_missing_value()
    test_filter_denoiser()
    test_outlier_handler()
    test_normalizer()
    test_full_pipeline()

    print("\n" + "=" * 50)
    print("所有测试通过!")
    print("=" * 50)
