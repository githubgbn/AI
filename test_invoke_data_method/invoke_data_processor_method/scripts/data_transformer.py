"""
数据替换算子 - DataTransformer
包含：对数变换、差分变换、标准化、类别编码
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List


class DataTransformer:
    """数据替换算子"""

    @staticmethod
    def log_transform(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None,
                      base: str = 'natural') -> pd.DataFrame:
        """
        对数变换 - 对数据进行对数变换

        使用场景：
        ①数据呈指数增长，如人口、GDP
        ②缓解异方差性，使方差稳定
        ③将偏态分布转换为近似正态，便于建模

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列
            base: 对数底，'natural'或数字

        Returns:
            变换后的数据框
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
                    result[col] = np.log(values)
                else:
                    result[col] = np.log(values) / np.log(base)
                result[col] = result[col].fillna(0)
        return result

    @staticmethod
    def diff_transform(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None,
                       periods: int = 1) -> pd.DataFrame:
        """
        差分变换 - 计算相邻值的差分

        使用场景：
        ①使非平稳时间序列平稳，消除趋势
        ②去除季节性，如季节差分
        ③ARIMA模型预处理

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列
            periods: 差分阶数

        Returns:
            变换后的数据框
        """
        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col in result.columns:
                result[col] = result[col].diff(periods)
        return result

    @staticmethod
    def standardize(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        标准化 - Z-Score标准化

        使用场景：
        ①不同量纲特征输入模型，如SVM、神经网络
        ②主成分分析前需标准化
        ③距离度量算法，如KNN、K-means

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
    def category_encode(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None,
                        method: str = 'label') -> pd.DataFrame:
        """
        类别编码替换 - 将分类变量转换为数值

        使用场景：
        ①将文本类别转换为数值，如标签编码（有序）
        ②避免引入顺序关系，用独热编码（无序）
        ③高基数类别用目标编码或嵌入

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有对象类型列
            method: 编码方法，'label'或'onehot'

        Returns:
            编码后的数据框
        """
        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.select_dtypes(include=['object']).columns.tolist()

        if method == 'label':
            for col in cols:
                if col in result.columns:
                    categories = result[col].unique()
                    mapping = {cat: i for i, cat in enumerate(categories)}
                    result[col] = result[col].map(mapping)
        elif method == 'onehot':
            for col in cols:
                if col in result.columns:
                    dummies = pd.get_dummies(result[col], prefix=col)
                    result = pd.concat([result, dummies], axis=1)
                    result = result.drop(columns=[col])
        return result
