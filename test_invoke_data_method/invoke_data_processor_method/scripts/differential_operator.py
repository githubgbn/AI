"""
差分算子 - DifferentialOperator
包含：一阶差分、滞后差分、高阶差分、季节性差分、对数差分、百分比差分
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List


class DifferentialOperator:
    """差分算子"""

    @staticmethod
    def first_order_diff(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None,
                         periods: int = 1) -> pd.DataFrame:
        """
        普通一阶差分 - 计算一阶差分

        使用场景：
        ①消除时间序列中的线性趋势，使非平稳序列平稳
        ②处理随机游走类型数据，如股票价格
        ③计算逐期变化量，用于分析增量或变化率

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列
            periods: 差分阶数

        Returns:
            差分后的数据框
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
    def lag_diff(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None,
                 k: int = 1) -> pd.DataFrame:
        """
        滞后k阶差分 - 计算滞后差分

        使用场景：
        ①消除特定周期的季节性，如季度数据做4阶差分
        ②分析相隔k期的变化，如比较今年与去年同月
        ③处理自回归模型中的滞后效应，如ARIMA模型的季节性部分

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列
            k: 滞后阶数

        Returns:
            差分后的数据框
        """
        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col in result.columns:
                result[col] = result[col].diff(k)
        return result

    @staticmethod
    def second_order_diff(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        二阶差分 - 计算二阶差分

        使用场景：
        ①消除二次型趋势，如加速度数据
        ②处理曲率变化，使序列平稳
        ③用于ARIMA模型中需要二阶差分的场景（d=2）

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列

        Returns:
            差分后的数据框
        """
        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col in result.columns:
                result[col] = result[col].diff(2)
        return result

    @staticmethod
    def n_order_diff(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None,
                     n: int = 2) -> pd.DataFrame:
        """
        n阶差分 - 计算n阶差分

        使用场景：
        ①消除高阶多项式趋势
        ②在ARIMA模型中指定任意阶差分
        ③一般化处理复杂趋势，如多次差分直至平稳

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列
            n: 差分阶数

        Returns:
            差分后的数据框
        """
        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col in result.columns:
                result[col] = result[col].diff(n)
        return result

    @staticmethod
    def seasonal_diff(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None,
                      period: int = 12) -> pd.DataFrame:
        """
        季节性差分 - 计算季节性差分

        使用场景：
        ①消除季节性周期，如月度数据做12步差分
        ②处理周期性波动，如每周数据做7步差分
        ③用于季节性时间序列的平稳化，如SARIMA模型

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列
            period: 季节周期

        Returns:
            差分后的数据框
        """
        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col in result.columns:
                result[col] = result[col].diff(period)
        return result

    @staticmethod
    def log_diff(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        对数一阶差分 - 计算对数差分

        使用场景：
        ①计算连续复利收益率，如金融资产收益率
        ②处理指数增长趋势，同时稳定方差
        ③将乘法模型（如乘法季节分解）转化为加法模型

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列

        Returns:
            差分后的数据框
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
                log_values = np.log(values)
                result[col] = pd.Series(log_values).diff().fillna(0)
        return result

    @staticmethod
    def percent_diff(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None,
                     periods: int = 1) -> pd.DataFrame:
        """
        相对变化率 - 计算百分比变化

        使用场景：
        ①计算百分比变化，如销售额增长率
        ②比较不同基数的变量变化，避免绝对数值影响
        ③用于经济指标中的同比或环比增长率分析

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列
            periods: 周期数

        Returns:
            差分后的数据框
        """
        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col in result.columns:
                result[col] = result[col].pct_change(periods).fillna(0)
        return result
