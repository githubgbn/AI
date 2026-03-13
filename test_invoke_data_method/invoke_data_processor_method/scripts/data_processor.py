"""
数据处理模块 - invoke_data_processor_method
包含7大类数据处理算子：缺失值处理、滤波降噪、数据替换、异常处理、差分算子、重采样、标准化/归一化
统一入口文件
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List

# 导入7个独立的算子模块
try:
    from .missing_value import MissingValueHandler
    from .filter_denoiser import FilterDenoiser
    from .data_transformer import DataTransformer
    from .outlier_handler import OutlierHandler
    from .differential_operator import DifferentialOperator
    from .resampler import Resampler
    from .normalizer import Normalizer
except ImportError:
    # 当作为脚本直接运行时使用绝对导入
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    import sys
    sys.path.insert(0, script_dir)
    from missing_value import MissingValueHandler
    from filter_denoiser import FilterDenoiser
    from data_transformer import DataTransformer
    from outlier_handler import OutlierHandler
    from differential_operator import DifferentialOperator
    from resampler import Resampler
    from normalizer import Normalizer


class DataProcessor:
    """
    数据处理器主类
    整合所有数据处理算子，提供统一的接口
    """

    def __init__(self):
        self.missing_value = MissingValueHandler()
        self.filter_denoiser = FilterDenoiser()
        self.data_transformer = DataTransformer()
        self.outlier_handler = OutlierHandler()
        self.differential_operator = DifferentialOperator()
        self.resampler = Resampler()
        self.normalizer = Normalizer()

    @staticmethod
    def load_csv(file_path: str) -> pd.DataFrame:
        """
        加载CSV文件

        Args:
            file_path: CSV文件路径

        Returns:
            数据框
        """
        return pd.read_csv(file_path)

    @staticmethod
    def save_to_csv(df: pd.DataFrame, file_path: str, index: bool = False) -> None:
        """
        保存数据框到CSV文件

        Args:
            df: 数据框
            file_path: 输出文件路径
            index: 是否保存索引
        """
        df.to_csv(file_path, index=index)

    @staticmethod
    def get_numeric_columns(df: pd.DataFrame) -> List[str]:
        """
        获取数值列

        Args:
            df: 数据框

        Returns:
            数值列列表
        """
        return df.select_dtypes(include=[np.number]).columns.tolist()

    @staticmethod
    def get_missing_info(df: pd.DataFrame) -> pd.DataFrame:
        """
        获取缺失值信息

        Args:
            df: 数据框

        Returns:
            缺失值统计
        """
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        return pd.DataFrame({'缺失数量': missing, '缺失比例(%)': missing_pct})


# 便捷函数
def load_csv(file_path: str) -> pd.DataFrame:
    """加载CSV文件"""
    return DataProcessor.load_csv(file_path)


def save_to_csv(df: pd.DataFrame, file_path: str, index: bool = False) -> None:
    """保存为CSV文件"""
    DataProcessor.save_to_csv(df, file_path, index)
