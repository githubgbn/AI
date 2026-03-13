"""
标准化/归一化算子 - Normalizer
包含：Z-Score、均值中心化、稳健标准化、Min-Max、对数、Box-Cox等
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List, Tuple


class Normalizer:
    """标准化/归一化算子"""

    @staticmethod
    def zscore_standardize(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        Z-Score标准化 - 将数据转换为均值为0，标准差为1

        使用场景：
        ①数据近似正态分布
        ②需要消除量纲影响

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列

        Returns:
            标准化后的数据框
        """
        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col in result.columns:
                mean = result[col].mean()
                std = result[col].std()
                if std != 0:
                    result[col] = (result[col] - mean) / std
        return result

    @staticmethod
    def mean_center(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        均值中心化 - 将数据转换为均值为0

        使用场景：
        ①只需消除偏移量
        ②特征尺度差异不大

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列

        Returns:
            中心化后的数据框
        """
        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col in result.columns:
                result[col] = result[col] - result[col].mean()
        return result

    @staticmethod
    def robust_standardize(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        稳健标准化 - 使用中位数和IQR进行标准化

        使用场景：
        ①数据含有明显异常值
        ②数据分布偏态明显
        ③对极端值敏感的场景

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列

        Returns:
            标准化后的数据框
        """
        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col in result.columns:
                median = result[col].median()
                q1 = result[col].quantile(0.25)
                q3 = result[col].quantile(0.75)
                iqr = q3 - q1
                if iqr != 0:
                    result[col] = (result[col] - median) / iqr
        return result

    @staticmethod
    def minmax_normalize(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None,
                         feature_range: Tuple[float, float] = (0, 1)) -> pd.DataFrame:
        """
        Min-Max归一化 - 将数据缩放到指定区间

        使用场景：
        ①数据分布范围已知

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列
            feature_range: 目标区间

        Returns:
            归一化后的数据框
        """
        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        min_val, max_val = feature_range

        for col in cols:
            if col in result.columns:
                col_min = result[col].min()
                col_max = result[col].max()
                if col_max != col_min:
                    result[col] = (result[col] - col_min) / (col_max - col_min) * (max_val - min_val) + min_val
        return result

    @staticmethod
    def range_normalize(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None,
                        feature_range: Tuple[float, float] = (-1, 1)) -> pd.DataFrame:
        """
        指定区间缩放 - 将数据缩放到指定区间

        使用场景：
        ①需要统一输出区间

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列
            feature_range: 目标区间

        Returns:
            归一化后的数据框
        """
        return Normalizer.minmax_normalize(df, columns, feature_range)

    @staticmethod
    def maxabs_normalize(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        最大绝对值归一化 - 将数据缩放到[-1, 1]区间

        使用场景：
        ①数据已以0为中心
        ②存在正负对称分布
        ③稀疏数据处理
        ④保持零值不变的场景

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列

        Returns:
            归一化后的数据框
        """
        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col in result.columns:
                max_abs = result[col].abs().max()
                if max_abs != 0:
                    result[col] = result[col] / max_abs
        return result

    @staticmethod
    def log_normalize(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None,
                      base: str = 'natural') -> pd.DataFrame:
        """
        对数变换归一化 - 使用对数变换归一化数据

        使用场景：
        ①数据右偏严重
        ②存在长尾分布

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列
            base: 对数底

        Returns:
            归一化后的数据框
        """
        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col in result.columns:
                values = result[col].values
                values = np.where(values > 0, values, np.nan)
                if base == 'natural':
                    result[col] = np.log1p(values)
                else:
                    result[col] = np.log1p(values) / np.log(base)
                result[col] = result[col].fillna(0)
        return result

    @staticmethod
    def boxcox_transform(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None,
                         lmbda: Optional[float] = None) -> pd.DataFrame:
        """
        Box-Cox变换 - 使用Box-Cox变换归一化数据

        使用场景：
        ①数据非正态分布
        ②需要满足正态性假设
        ③用于线性回归建模
        ④数据为正数

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列
            lmbda: 变换参数，None表示自动选择

        Returns:
            变换后的数据框
        """
        from scipy.stats import boxcox

        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col in result.columns:
                values = result[col].values + 1
                values = np.where(values > 0, values, np.nan)
                values = values[~np.isnan(values)]
                if len(values) > 0 and np.all(values > 0):
                    transformed, _ = boxcox(values, lmbda)
                    result[col] = transformed
        return result
