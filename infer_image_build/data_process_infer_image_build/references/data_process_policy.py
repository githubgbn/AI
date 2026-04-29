#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_process_policy 数据处理模块
提供 deal_data 方法：缺失值填充、滤波降噪、异常值处理
"""
import pandas as pd
import numpy as np


def deal_data(df):
    """
    对输入 DataFrame 进行数据清洗处理。

    处理步骤：
    1. 缺失值填充（数值列用中位数，分类列用众数）
    2. 滤波降噪（使用移动平均平滑数值列）
    3. 异常值处理（使用 IQR 方法检测并截断异常值）

    Args:
        df: pandas DataFrame，原始数据

    Returns:
        pandas DataFrame，清洗后的数据
    """
    if df is None or df.empty:
        raise ValueError("输入数据为空")

    cleaned = df.copy()

    # === 1. 缺失值填充 ===
    for col in cleaned.columns:
        if cleaned[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(cleaned[col]):
                # 数值列用中位数填充
                median_val = cleaned[col].median()
                cleaned[col] = cleaned[col].fillna(median_val)
            else:
                # 非数值列用众数填充
                mode_val = cleaned[col].mode()
                if len(mode_val) > 0:
                    cleaned[col] = cleaned[col].fillna(mode_val[0])

    # === 2. 滤波降噪（移动平均平滑） ===
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col == 'status':  # 跳过目标列
            continue
        # 3点移动平均平滑
        cleaned[col] = cleaned[col].rolling(window=3, min_periods=1, center=True).mean()

    # === 3. 异常值处理（IQR 方法截断） ===
    for col in numeric_cols:
        if col == 'status':  # 跳过目标列
            continue
        Q1 = cleaned[col].quantile(0.25)
        Q3 = cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 截断异常值到边界
        cleaned[col] = cleaned[col].clip(lower=lower_bound, upper=upper_bound)

    return cleaned
