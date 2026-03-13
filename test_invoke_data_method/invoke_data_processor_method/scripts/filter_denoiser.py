"""
滤波降噪算子 - FilterDenoiser
包含：中值滤波、移动平均、低通滤波、傅里叶变换
"""

import pandas as pd
import numpy as np
from scipy import signal, fft
from typing import Optional, Union, List


class FilterDenoiser:
    """滤波降噪算子"""

    @staticmethod
    def median_filter(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None,
                      kernel_size: int = 3) -> pd.DataFrame:
        """
        中值滤波 - 使用中值滤波去除噪声

        使用场景：
        ①去除信号或图像中的椒盐噪声
        ②保持边缘的同时平滑数据，如雷达信号处理
        ③处理非高斯噪声，如脉冲噪声

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列
            kernel_size: 核大小，必须为奇数

        Returns:
            滤波后的数据框
        """
        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        if kernel_size % 2 == 0:
            kernel_size += 1

        for col in cols:
            if col in result.columns:
                result[col] = signal.medfilt(result[col].values, kernel_size)
        return result

    @staticmethod
    def moving_average(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None,
                       window: int = 5) -> pd.DataFrame:
        """
        移动平均 - 使用移动平均平滑数据

        使用场景：
        ①平滑时间序列，消除短期波动，揭示长期趋势
        ②计算技术指标，如股票均线
        ③季节性调整，如月度销售数据

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列
            window: 窗口大小

        Returns:
            平滑后的数据框
        """
        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col in result.columns:
                result[col] = result[col].rolling(window=window, min_periods=1).mean()
        return result

    @staticmethod
    def exponential_moving_average(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None,
                                    span: int = 5) -> pd.DataFrame:
        """
        指数加权移动平均 - 赋予近期数据更高权重

        使用场景：
        ①需要赋予近期数据更高权重，如实时监控
        ②金融风险度量，如波动率估计
        ③时间序列预测中的平滑参数

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列
            span: 跨度参数

        Returns:
            平滑后的数据框
        """
        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col in result.columns:
                result[col] = result[col].ewm(span=span, min_periods=1).mean()
        return result

    @staticmethod
    def lowpass_filter(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None,
                       cutoff: float = 0.1) -> pd.DataFrame:
        """
        低通滤波 - 滤除高频噪声，保留低频信号

        使用场景：
        ①滤除高频噪声，保留低频信号，如音频去噪
        ②生物医学信号处理，如心电图滤波
        ③图像模糊化，去除细节

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列
            cutoff: 截止频率（0-1之间）

        Returns:
            滤波后的数据框
        """
        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        nyquist = 0.5
        normalized_cutoff = cutoff / nyquist
        b, a = signal.butter(8, normalized_cutoff, btype='low')

        for col in cols:
            if col in result.columns:
                result[col] = signal.filtfilt(b, a, result[col].values)
        return result

    @staticmethod
    def fourier_transform(df: pd.DataFrame, columns: Optional[Union[str, List[str]]] = None,
                          low_freq_ratio: float = 0.1) -> pd.DataFrame:
        """
        傅里叶变换 - 频域滤波处理

        使用场景：
        ①信号频谱分析，识别频率成分
        ②图像频域滤波，如高通、低通
        ③特征提取，如音频特征

        Args:
            df: 输入数据框
            columns: 指定列，None表示所有数值列
            low_freq_ratio: 保留低频比例

        Returns:
            滤波后的数据框
        """
        result = df.copy()
        if columns:
            cols = columns if isinstance(columns, list) else [columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col in result.columns:
                values = result[col].values
                fft_vals = fft.fft(values)
                n = len(values)
                cutoff = int(n * low_freq_ratio)
                fft_vals[cutoff:-cutoff] = 0
                result[col] = np.real(fft.ifft(fft_vals))
        return result
