"""
重采样算子 - Resampler
包含：上采样（线性、多项式、样条）、下采样（均值、求和）
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from typing import Optional, Union, List


class Resampler:
    """重采样算子"""

    @staticmethod
    def upsample_linear(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None,
                        factor: int = 2) -> pd.DataFrame:
        """
        线性插值上采样 - 使用线性插值增加样本

        使用场景：
        ①数据变化平缓
        ②局部近似线性变化
        ③对计算效率要求高

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列
            factor: 上采样倍数

        Returns:
            重采样后的数据框
        """
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()

        non_numeric_cols = [c for c in df.columns if c not in cols]

        n_orig = len(df)
        n_new = n_orig * factor

        new_index = np.linspace(0, n_orig - 1, n_new)
        old_index = np.arange(n_orig)

        new_data = {}
        for col in cols:
            if col in df.columns:
                interp_func = interp1d(old_index, df[col].values, kind='linear')
                new_data[col] = interp_func(new_index)

        for col in non_numeric_cols:
            new_data[col] = np.repeat(df[col].values, factor)

        result = pd.DataFrame(new_data)
        return result

    @staticmethod
    def upsample_polynomial(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None,
                            factor: int = 2, degree: int = 3) -> pd.DataFrame:
        """
        多项式插值上采样 - 使用多项式插值增加样本

        使用场景：
        ①存在明显非线性趋势
        ②整体变化较平滑
        ③样本规模较小

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列
            factor: 上采样倍数
            degree: 多项式阶数

        Returns:
            重采样后的数据框
        """
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()

        non_numeric_cols = [c for c in df.columns if c not in cols]

        n_orig = len(df)
        n_new = n_orig * factor

        new_index = np.linspace(0, n_orig - 1, n_new)
        old_index = np.arange(n_orig)

        new_data = {}
        for col in cols:
            if col in df.columns:
                interp_func = interp1d(old_index, df[col].values, kind='polynomial', degree=degree)
                new_data[col] = interp_func(new_index)

        for col in non_numeric_cols:
            new_data[col] = np.repeat(df[col].values, factor)

        result = pd.DataFrame(new_data)
        return result

    @staticmethod
    def upsample_spline(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None,
                        factor: int = 2, order: int = 3) -> pd.DataFrame:
        """
        样条插值上采样 - 使用样条插值增加样本

        使用场景：
        ①要求曲线连续性高
        ②要求平滑性强
        ③中等规模数据建模

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列
            factor: 上采样倍数
            order: 样条阶数

        Returns:
            重采样后的数据框
        """
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()

        non_numeric_cols = [c for c in df.columns if c not in cols]

        n_orig = len(df)
        n_new = n_orig * factor

        new_index = np.linspace(0, n_orig - 1, n_new)
        old_index = np.arange(n_orig)

        new_data = {}
        for col in cols:
            if col in df.columns:
                interp_func = interp1d(old_index, df[col].values, kind='cubic')
                new_data[col] = interp_func(new_index)

        for col in non_numeric_cols:
            new_data[col] = np.repeat(df[col].values, factor)

        result = pd.DataFrame(new_data)
        return result

    @staticmethod
    def downsample_mean(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None,
                        factor: int = 2) -> pd.DataFrame:
        """
        均值聚合下采样 - 使用均值减少样本

        使用场景：
        ①关注整体趋势
        ②需要降低噪声影响
        ③连续型数值数据

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列
            factor: 下采样倍数

        Returns:
            重采样后的数据框
        """
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()

        non_numeric_cols = [c for c in df.columns if c not in cols]

        n_new = len(df) // factor

        new_data = {}
        for col in cols:
            if col in df.columns:
                new_data[col] = df[col].values[:n_new * factor].reshape(n_new, factor).mean(axis=1)

        for col in non_numeric_cols:
            new_data[col] = df[col].values[:n_new * factor].reshape(n_new, factor).max(axis=1)

        result = pd.DataFrame(new_data)
        return result

    @staticmethod
    def downsample_sum(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None,
                       factor: int = 2) -> pd.DataFrame:
        """
        求和聚合下采样 - 使用求和减少样本

        使用场景：
        ①关注累计总量
        ②统计区间贡献

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列
            factor: 下采样倍数

        Returns:
            重采样后的数据框
        """
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()

        non_numeric_cols = [c for c in df.columns if c not in cols]

        n_new = len(df) // factor

        new_data = {}
        for col in cols:
            if col in df.columns:
                new_data[col] = df[col].values[:n_new * factor].reshape(n_new, factor).sum(axis=1)

        for col in non_numeric_cols:
            new_data[col] = df[col].values[:n_new * factor].reshape(n_new, factor).max(axis=1)

        result = pd.DataFrame(new_data)
        return result
