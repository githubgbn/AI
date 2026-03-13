"""
缺失值处理算子 - MissingValueHandler
包含：删除法、数据填充、插值法
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List


class MissingValueHandler:
    """缺失值处理算子"""

    @staticmethod
    def delete_row(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        行删除 - 删除包含缺失值的行

        使用场景：
        ①随机缺失且比例极低
        ②预测任务不需要严格等间隔
        ③标签数据缺失

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有列

        Returns:
            删除缺失值后的数据框
        """
        result = df.copy()
        if columns:
            result = result.dropna(subset=columns if isinstance(columns, list) else [columns])
        else:
            result = result.dropna()
        return result

    @staticmethod
    def delete_column(df: pd.DataFrame, threshold: float = 0.4) -> pd.DataFrame:
        """
        列删除 - 删除缺失率过高的列

        使用场景：
        ①缺失率过高（>40%）
        ②特征无业务意义
        ③未来数据不可获取

        Args:
            df: 输入数据框
            threshold: 缺失率阈值，默认0.4

        Returns:
            删除高缺失率列后的数据框
        """
        result = df.copy()
        missing_ratio = result.isnull().sum() / len(result)
        cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
        result = result.drop(columns=cols_to_drop)
        return result

    @staticmethod
    def fill_forward(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        前项填充 - 使用前一个有效值填充缺失值

        使用场景：
        ①时间序列数据中，缺失值前后变化缓慢，用前值填充合理
        ②传感器数据短暂中断，用前一时刻值替代
        ③数据具有顺序依赖性，如股票价格，缺失时用上一交易日值

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有列

        Returns:
            填充后的数据框
        """
        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col in result.columns:
                result[col] = result[col].ffill()
        return result

    @staticmethod
    def fill_backward(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        后向填充 - 使用后一个有效值填充缺失值

        使用场景：
        ①时间序列中，缺失值后紧接着有效值，且趋势反向合理（如数据回填）
        ②某些实验数据中，后续测量值更准确，用后值填充
        ③在数据预处理中，用于前向填充的补充，如双向填充

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有列

        Returns:
            填充后的数据框
        """
        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col in result.columns:
                result[col] = result[col].bfill()
        return result

    @staticmethod
    def fill_mean(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        均值填充 - 使用均值填充缺失值

        使用场景：
        ①数据分布近似正态，缺失率较低，且变量重要性不高
        ②快速填补连续变量缺失，作为基线方法
        ③在分组均值填充中，按类别分组后填充

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列

        Returns:
            填充后的数据框
        """
        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col in result.columns:
                result[col] = result[col].fillna(result[col].mean())
        return result

    @staticmethod
    def fill_median(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        中位数填充 - 使用中位数填充缺失值

        使用场景：
        ①数据存在异常值或偏态分布，如收入、房价
        ②需要稳健估计，避免极端值影响
        ③有序分类变量（如等级）的缺失填充

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列

        Returns:
            填充后的数据框
        """
        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col in result.columns:
                result[col] = result[col].fillna(result[col].median())
        return result

    @staticmethod
    def fill_constant(df: pd.DataFrame, value: float = 0, columns: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        固定值填充 - 使用固定值填充缺失值

        使用场景：
        ①缺失值有特定含义，如用-1表示"无响应"
        ②业务规则要求，如年龄缺失统一填0
        ③在特征工程中，为缺失单独编码

        Args:
            df: 输入数据框
            value: 固定填充值，默认0
            columns: 指定列，None表示所有数值列

        Returns:
            填充后的数据框
        """
        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col in result.columns:
                result[col] = result[col].fillna(value)
        return result

    @staticmethod
    def fill_mode(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        众数填充 - 使用众数填充缺失值

        使用场景：
        ①分类变量缺失，如性别、职业
        ②离散数据，如整数计数，用最常见值
        ③多选题中，缺失用最常选选项

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有列

        Returns:
            填充后的数据框
        """
        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.columns.tolist()

        for col in cols:
            if col in result.columns:
                mode_val = result[col].mode()
                if len(mode_val) > 0:
                    result[col] = result[col].fillna(mode_val[0])
        return result

    @staticmethod
    def interpolate_linear(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        线性插值 - 使用线性插值填充缺失值

        使用场景：
        ①时间序列中连续缺失，假设线性变化，如气温逐时插值
        ②空间数据中两点间插值，如地理高程
        ③图像缩放中的像素估计

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列

        Returns:
            插值后的数据框
        """
        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col in result.columns:
                result[col] = result[col].interpolate(method='linear')
        return result

    @staticmethod
    def interpolate_spline(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None, order: int = 3) -> pd.DataFrame:
        """
        样条插值 - 使用样条插值填充缺失值

        使用场景：
        ①数据平滑且需要高阶连续性，如曲线拟合
        ②缺失区间较长，线性插值不够精确
        ③图像处理中的超分辨率重建

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列
            order: 样条阶数，默认3

        Returns:
            插值后的数据框
        """
        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col in result.columns:
                result[col] = result[col].interpolate(method='spline', order=order)
        return result
