---
name: analyze
description: 数据分析skill。用于加载CSV/Excel数据文件、评估数据质量、生成数据分析报告。当用户需要分析数据质量、检查缺失值/异常值、生成数据概览报告时使用此skill。
---

# analyze 数据分析 Skill

本skill负责加载数据文件、评估数据质量、生成数据分析报告。

## 脚本

具体实现见: `scripts/analyze.py`

## 功能概述

## 功能概述

1. **加载数据文件** - 支持CSV和Excel格式
2. **评估数据质量** - 检查缺失值、数据类型、异常值等
3. **生成分析报告** - 输出数据质量评分和统计信息

## 使用方法

### 1. 导入模块

```python
import sys
import pandas as pd
import numpy as np

# 数据处理模块路径
PROCESSOR_PATH = '/mnt/workspace/gbn/project/test_invoke_data_method/invoke_data_processor_method/scripts'
sys.path.append(PROCESSOR_PATH)
```

### 2. 加载数据

```python
def load_data(file_path: str) -> pd.DataFrame:
    """加载CSV或Excel文件"""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")
    return df
```

### 3. 评估数据质量

```python
def assess_data_quality(df: pd.DataFrame, target_columns: list = None) -> dict:
    """
    评估数据质量

    返回评估结果包含:
    - 数据形状 (行数、列数)
    - 缺失值统计 (每列缺失数量和比例)
    - 数据类型分布
    - 数值列统计 (均值、中位数、标准差、最小值、最大值)
    - 异常值检测 (使用IQR方法)
    - 数据质量评分 (0-100)
    """
    result = {
        'shape': df.shape,
        'columns': list(df.columns),
        'missing_values': {},
        'dtypes': {},
        'numeric_stats': {},
        'outliers': {},
        'quality_score': 0
    }

    # 缺失值统计
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_ratio = missing_count / len(df) if len(df) > 0 else 0
        result['missing_values'][col] = {
            'count': int(missing_count),
            'ratio': float(missing_ratio)
        }

    # 数据类型
    result['dtypes'] = df.dtypes.apply(str).to_dict()

    # 数值列统计
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        result['numeric_stats'][col] = {
            'mean': float(df[col].mean()),
            'median': float(df[col].median()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'q25': float(df[col].quantile(0.25)),
            'q75': float(df[col].quantile(0.75))
        }

    # 异常值检测 (IQR方法)
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        result['outliers'][col] = {
            'count': int(outlier_count),
            'ratio': float(outlier_count / len(df)) if len(df) > 0 else 0,
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound)
        }

    # 计算数据质量评分
    score = 100
    # 缺失值扣分
    avg_missing = np.mean([v['ratio'] for v in result['missing_values'].values()])
    score -= avg_missing * 50
    # 异常值扣分
    if result['outliers']:
        avg_outlier = np.mean([v['ratio'] for v in result['outliers'].values()])
        score -= avg_outlier * 30
    result['quality_score'] = max(0, round(score, 2))

    return result
```

### 4. 生成分析报告

```python
def generate_analysis_report(df: pd.DataFrame, assess_result: dict) -> str:
    """生成数据分析报告"""
    report = []
    report.append("# 数据分析报告\n")
    report.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 数据概览
    report.append("## 1. 数据概览\n")
    report.append(f"- **数据形状**: {assess_result['shape'][0]} 行 × {assess_result['shape'][1]} 列\n")
    report.append(f"- **列名**: {', '.join(assess_result['columns'])}\n")
    report.append(f"- **数据质量评分**: {assess_result['quality_score']}/100\n")

    # 缺失值分析
    report.append("\n## 2. 缺失值分析\n")
    for col, info in assess_result['missing_values'].items():
        if info['ratio'] > 0:
            report.append(f"- **{col}**: {info['count']} 个缺失值 ({info['ratio']*100:.1f}%)\n")

    # 数值列统计
    if assess_result['numeric_stats']:
        report.append("\n## 3. 数值列统计\n")
        report.append("| 列名 | 均值 | 中位数 | 标准差 | 最小值 | 最大值 |\n")
        report.append("|------|------|--------|--------|--------|--------|\n")
        for col, stats in assess_result['numeric_stats'].items():
            report.append(f"| {col} | {stats['mean']:.2f} | {stats['median']:.2f} | {stats['std']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} |\n")

    # 异常值分析
    if assess_result['outliers']:
        report.append("\n## 4. 异常值分析\n")
        for col, info in assess_result['outliers'].items():
            if info['count'] > 0:
                report.append(f"- **{col}**: {info['count']} 个异常值 ({info['ratio']*100:.1f}%)\n")

    # 数据类型
    report.append("\n## 5. 数据类型\n")
    for col, dtype in assess_result['dtypes'].items():
        report.append(f"- **{col}**: {dtype}\n")

    return ''.join(report)
```

## 完整工作流

```python
import pandas as pd
import sys

PROCESSOR_PATH = '/mnt/workspace/gbn/project/test_invoke_data_method/invoke_data_processor_method/scripts'
sys.path.append(PROCESSOR_PATH)

def run_analyze(file_path: str, target_columns: list = None):
    """
    执行数据分析流程

    Args:
        file_path: 数据文件路径 (CSV/Excel)
        target_columns: 目标列名列表 (可选)

    Returns:
        dict: 包含 df, assess_result, analysis_report
    """
    # 1. 加载数据
    print(f"正在加载数据: {file_path}")
    df = load_data(file_path)

    # 如果指定了目标列，只保留这些列
    if target_columns:
        available_cols = [c for c in target_columns if c in df.columns]
        if available_cols:
            df = df[available_cols]
            print(f"已筛选目标列: {available_cols}")

    # 2. 评估数据质量
    print("正在评估数据质量...")
    assess_result = assess_data_quality(df)

    # 3. 生成分析报告
    print("正在生成分析报告...")
    analysis_report = generate_analysis_report(df, assess_result)

    print(f"\n数据质量评分: {assess_result['quality_score']}/100")

    return {
        'df': df,
        'assess_result': assess_result,
        'analysis_report': analysis_report
    }
```

## 输出格式

### df
原始 pandas DataFrame 对象

### assess_result
```python
{
    'shape': (行数, 列数),
    'columns': [列名列表],
    'missing_values': {
        '列名': {'count': 数量, 'ratio': 比例}
    },
    'dtypes': {'列名': '数据类型'},
    'numeric_stats': {
        '列名': {'mean': 均值, 'median': 中位数, 'std': 标准差, ...}
    },
    'outliers': {
        '列名': {'count': 数量, 'ratio': 比例, ...}
    },
    'quality_score': 0-100的评分
}
```

### analysis_report
Markdown 格式的分析报告字符串

## 依赖

- pandas
- numpy
- openpyxl (用于读取Excel文件)
