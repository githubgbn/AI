"""
异常处理算子 - OutlierHandler
包含：3-Sigma、四分位距法、聚类检测、移动标准差法、Z-score
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List


class OutlierHandler:
    """异常处理算子"""

    @staticmethod
    def three_sigma(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None,
                    n_sigma: float = 3.0, method: str = 'clip') -> pd.DataFrame:
        """
        3-Sigma方法 - 基于3倍标准差检测和处理异常值

        使用场景：
        ①数据近似正态分布，检测异常值
        ②质量控制中，超出控制限的点
        ③简单快速异常检测

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列
            n_sigma: Sigma倍数，默认3
            method: 处理方法，'clip'截断或'nan'替换

        Returns:
            处理后的数据框
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
                lower = mean - n_sigma * std
                upper = mean + n_sigma * std

                if method == 'clip':
                    result[col] = result[col].clip(lower, upper)
                elif method == 'nan':
                    result[col] = result[col].where((result[col] >= lower) & (result[col] <= upper), np.nan)
        return result

    @staticmethod
    def quartile_method(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None,
                        method: str = 'clip', k: float = 1.5) -> pd.DataFrame:
        """
        四分位距法 - 基于IQR检测和处理异常值

        使用场景：
        ①数据非正态，有偏态或异常值
        ②箱线图自动识别异常点
        ③稳健异常检测，不受极端值影响

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列
            method: 处理方法，'clip'截断或'nan'替换
            k: IQR倍数，默认1.5

        Returns:
            处理后的数据框
        """
        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col in result.columns:
                q1 = result[col].quantile(0.25)
                q3 = result[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - k * iqr
                upper = q3 + k * iqr

                if method == 'clip':
                    result[col] = result[col].clip(lower, upper)
                elif method == 'nan':
                    result[col] = result[col].where((result[col] >= lower) & (result[col] <= upper), np.nan)
        return result

    @staticmethod
    def clustering_detection(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None,
                             n_clusters: int = 3, threshold: float = 2.0) -> pd.DataFrame:
        """
        聚类检测 - 使用K-Means聚类检测异常值

        使用场景：
        ①发现数据自然分组，如客户分群
        ②异常检测，远离簇中心的点
        ③图像分割，像素聚类

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列
            n_clusters: 聚类数
            threshold: 距离阈值

        Returns:
            处理后的数据框
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        data = result[cols].values
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_scaled)

        distances = np.min(kmeans.transform(data_scaled), axis=1)
        outlier_mask = distances > threshold

        for i, col in enumerate(cols):
            result.loc[outlier_mask, col] = np.nan

        return result

    @staticmethod
    def moving_std(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None,
                   window: int = 10, threshold: float = 3.0) -> pd.DataFrame:
        """
        移动标准差法 - 基于局部标准差检测异常值

        使用场景：
        ①数据存在趋势或季节性波动
        ②局部突变检测
        ③需要区分局部异常与全局异常

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列
            window: 窗口大小
            threshold: 阈值

        Returns:
            处理后的数据框
        """
        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col in result.columns:
                rolling_mean = result[col].rolling(window=window, min_periods=1).mean()
                rolling_std = result[col].rolling(window=window, min_periods=1).std()
                z_scores = np.abs((result[col] - rolling_mean) / rolling_std)
                result[col] = result[col].where(z_scores < threshold, np.nan)
        return result

    @staticmethod
    def zscore(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None,
               threshold: float = 3.0, method: str = 'clip') -> pd.DataFrame:
        """
        Z-score方法 - 基于Z分数检测异常值

        使用场景：
        ①数据量足够大，异常比例低
        ②数据平稳且无趋势、无周期性
        ③数据服从近似正态分布

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列
            threshold: Z分数阈值
            method: 处理方法

        Returns:
            处理后的数据框
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
                    z_scores = np.abs((result[col] - mean) / std)
                    if method == 'clip':
                        result[col] = result[col].where(z_scores < threshold, np.nan)
                    elif method == 'clip_value':
                        upper = mean + threshold * std
                        lower = mean - threshold * std
                        result[col] = result[col].clip(lower, upper)
        return result
