---
name: process
description: 数据处理skill。用于执行数据处理、生成评价报告、生成对比图表。当用户需要根据推荐策略执行数据清洗、处理后生成对比报告时使用此skill。
---

# process 数据处理 Skill

本skill负责执行数据处理、生成评价报告、生成对比图表。

## 脚本

具体实现见: `scripts/process.py`

## 功能概述

1. **执行数据处理** - 根据调用链表执行invoke_data_processor_method的各算子
2. **生成评价报告** - 对比处理前后的数据质量
3. **生成对比图表** - 可视化展示处理效果

## 使用方法

### 1. 导入数据处理模块

```python
import sys
import pandas as pd
import numpy as np

# 数据处理模块路径
PROCESSOR_PATH = '/mnt/workspace/gbn/project/test_invoke_data_method/invoke_data_processor_method/scripts'
sys.path.append(PROCESSOR_PATH)

from data_processor import DataProcessor
from missing_value import MissingValueHandler
from filter_denoiser import FilterDenoiser
from data_transformer import DataTransformer
from outlier_handler import OutlierHandler
from differential_operator import DifferentialOperator
from resampler import Resampler
from normalizer import Normalizer
```

### 2. 执行数据处理

```python
def execute_processing(df: pd.DataFrame, plan: list) -> pd.DataFrame:
    """
    根据调用链表执行数据处理

    Args:
        df: 原始数据DataFrame
        plan: 调用链表

    Returns:
        处理后的DataFrame
    """
    # 创建处理器实例
    processor = DataProcessor()

    # 初始化各算子
    missing_handler = MissingValueHandler()
    filter_handler = FilterDenoiser()
    transformer = DataTransformer()
    outlier_handler = OutlierHandler()
    diff_operator = DifferentialOperator()
    resampler = Resampler()
    normalizer = Normalizer()

    # 算子映射
    processors = {
        'missing_value': missing_handler,
        'filter_denoiser': filter_handler,
        'data_transformer': transformer,
        'outlier_handler': outlier_handler,
        'differential_operator': diff_operator,
        'resampler': resampler,
        'normalizer': normalizer
    }

    df_result = df.copy()
    processing_log = []

    for step in plan:
        processor_name = step['processor']
        method_name = step['method']
        columns = step.get('columns', [])
        params = step.get('params', {})

        try:
            proc = processors[processor_name]
            method = getattr(proc, method_name)

            # 根据方法签名调用
            if columns and method_name not in ['delete_column', 'category_encode']:
                if columns:
                    df_result = method(df_result, columns=columns, **params)
                else:
                    df_result = method(df_result, **params)
            else:
                df_result = method(df_result, **params)

            processing_log.append({
                'step': step['step'],
                'processor': processor_name,
                'method': method_name,
                'columns': columns,
                'status': 'success'
            })
            print(f"步骤 {step['step']}: {processor_name}.{method_name} 执行成功")

        except Exception as e:
            processing_log.append({
                'step': step['step'],
                'processor': processor_name,
                'method': method_name,
                'columns': columns,
                'status': 'failed',
                'error': str(e)
            })
            print(f"步骤 {step['step']}: {processor_name}.{method_name} 执行失败: {e}")

    return df_result, processing_log
```

### 3. 评估处理效果

```python
def evaluate_processing(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
    """
    评估处理效果

    对比处理前后的数据质量指标
    """
    evaluation = {
        'shape_change': {
            'before': df_before.shape,
            'after': df_after.shape
        },
        'missing_change': {},
        'outlier_change': {},
        'numeric_change': {}
    }

    # 缺失值变化
    for col in df_before.columns:
        before_missing = df_before[col].isnull().sum()
        after_missing = df_after[col].isnull().sum()
        evaluation['missing_change'][col] = {
            'before': int(before_missing),
            'after': int(after_missing),
            'change': int(after_missing - before_missing)
        }

    # 数值列变化
    numeric_cols = df_before.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        evaluation['numeric_change'][col] = {
            'before_mean': float(df_before[col].mean()),
            'after_mean': float(df_after[col].mean()),
            'before_std': float(df_before[col].std()),
            'after_std': float(df_after[col].std())
        }

    return evaluation
```

### 4. 生成评价报告

```python
def generate_evaluation_report(df_before: pd.DataFrame, df_after: pd.DataFrame,
                               evaluation: dict, processing_log: list) -> str:
    """生成处理评价报告"""
    report = []
    report.append("# 数据处理评价报告\n")
    report.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 处理概览
    report.append("## 1. 处理概览\n")
    report.append(f"- **原始数据**: {evaluation['shape_change']['before'][0]} 行 × {evaluation['shape_change']['before'][1]} 列\n")
    report.append(f"- **处理后数据**: {evaluation['shape_change']['after'][0]} 行 × {evaluation['shape_change']['after'][1]} 列\n")
    report.append(f"- **执行步骤数**: {len(processing_log)}\n")

    # 处理步骤详情
    report.append("\n## 2. 处理步骤详情\n")
    report.append("| 步骤 | 处理器 | 方法 | 状态 |\n")
    report.append("|------|--------|------|------|\n")
    for log in processing_log:
        status = "✓ 成功" if log['status'] == 'success' else f"✗ 失败: {log.get('error', '')}"
        report.append(f"| {log['step']} | {log['processor']} | {log['method']} | {status} |\n")

    # 缺失值变化
    report.append("\n## 3. 缺失值变化\n")
    missing_changes = [v for v in evaluation['missing_change'].values() if v['change'] != 0]
    if missing_changes:
        report.append("| 列名 | 处理前 | 处理后 | 变化 |\n")
        report.append("|------|--------|--------|------|\n")
        for col, info in evaluation['missing_change'].items():
            if info['change'] != 0:
                change_str = f"{info['change']:+d}"
                report.append(f"| {col} | {info['before']} | {info['after']} | {change_str} |\n")
    else:
        report.append("无变化\n")

    # 数值列变化
    if evaluation['numeric_change']:
        report.append("\n## 4. 数值列统计变化\n")
        report.append("| 列名 | 处理前均值 | 处理后均值 | 处理前标准差 | 处理后标准差 |\n")
        report.append("|------|-----------|-----------|-------------|-------------|\n")
        for col, stats in evaluation['numeric_change'].items():
            report.append(f"| {col} | {stats['before_mean']:.2f} | {stats['after_mean']:.2f} | {stats['before_std']:.2f} | {stats['after_std']:.2f} |\n")

    # 质量评分
    quality_score_before = calculate_quality_score(df_before)
    quality_score_after = calculate_quality_score(df_after)

    report.append("\n## 5. 数据质量评分\n")
    report.append(f"- **处理前**: {quality_score_before}/100\n")
    report.append(f"- **处理后**: {quality_score_after}/100\n")
    report.append(f"- **提升**: {quality_score_after - quality_score_before:+d} 分\n")

    return ''.join(report)


def calculate_quality_score(df: pd.DataFrame) -> float:
    """计算数据质量评分"""
    score = 100

    # 缺失值扣分
    for col in df.columns:
        missing_ratio = df[col].isnull().sum() / len(df) if len(df) > 0 else 0
        score -= missing_ratio * 30

    # 数值列异常值扣分
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        outlier_ratio = outlier_count / len(df) if len(df) > 0 else 0
        score -= outlier_ratio * 20

    return max(0, round(score, 2))
```

### 5. 生成对比图表

```python
import matplotlib.pyplot as plt

def generate_comparison_charts(df_before: pd.DataFrame, df_after: pd.DataFrame,
                               output_dir: str = '.') -> dict:
    """
    生成对比图表

    Args:
        df_before: 处理前数据
        df_after: 处理后数据
        output_dir: 输出目录

    Returns:
        图表文件路径字典
    """
    chart_paths = {}

    # 获取数值列
    numeric_cols = df_before.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        # 1. 缺失值对比图
        fig, ax = plt.subplots(figsize=(10, 6))
        cols = list(df_before.columns)
        before_missing = [df_before[col].isnull().sum() for col in cols]
        after_missing = [df_after[col].isnull().sum() for col in cols]

        x = np.arange(len(cols))
        width = 0.35

        ax.bar(x - width/2, before_missing, width, label='处理前', color='coral')
        ax.bar(x + width/2, after_missing, width, label='处理后', color='steelblue')

        ax.set_xlabel('列名')
        ax.set_ylabel('缺失值数量')
        ax.set_title('缺失值对比')
        ax.set_xticks(x)
        ax.set_xticklabels(cols, rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()

        missing_chart_path = f'{output_dir}/missing_comparison.png'
        plt.savefig(missing_chart_path)
        plt.close()
        chart_paths['missing_comparison'] = missing_chart_path

        # 2. 数值列分布对比图 (取前4个数值列)
        sample_cols = numeric_cols[:4]
        n_cols = len(sample_cols)
        fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 5))

        if n_cols == 1:
            axes = [axes]

        for i, col in enumerate(sample_cols):
            axes[i].hist(df_before[col].dropna(), bins=30, alpha=0.5, label='处理前', color='coral')
            axes[i].hist(df_after[col].dropna(), bins=30, alpha=0.5, label='处理后', color='steelblue')
            axes[i].set_title(f'{col} 分布对比')
            axes[i].legend()

        plt.tight_layout()

        dist_chart_path = f'{output_dir}/distribution_comparison.png'
        plt.savefig(dist_chart_path)
        plt.close()
        chart_paths['distribution_comparison'] = dist_chart_path

        # 3. 质量评分雷达图
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

        metrics = ['完整度', '准确性', '一致性', '可用性']
        before_scores = calculate_radar_scores(df_before)
        after_scores = calculate_radar_scores(df_after)

        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        before_scores += before_scores[:1]
        after_scores += after_scores[:1]

        ax.plot(angles, before_scores, 'o-', linewidth=2, label='处理前', color='coral')
        ax.fill(angles, before_scores, alpha=0.25, color='coral')
        ax.plot(angles, after_scores, 'o-', linewidth=2, label='处理后', color='steelblue')
        ax.fill(angles, after_scores, alpha=0.25, color='steelblue')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_title('数据质量评分对比', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        radar_chart_path = f'{output_dir}/quality_radar.png'
        plt.savefig(radar_chart_path)
        plt.close()
        chart_paths['quality_radar'] = radar_chart_path

    return chart_paths


def calculate_radar_scores(df: pd.DataFrame) -> list:
    """计算雷达图评分"""
    # 完整度 (1 - 缺失率)
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 0

    # 准确性 (基于异常值比例)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        outlier_ratios = []
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
            outlier_ratios.append(outlier_count / len(df))
        accuracy = 1 - np.mean(outlier_ratios) if outlier_ratios else 1
    else:
        accuracy = 1

    # 一致性 (数据类型一致性)
    consistency = 1.0  # 简化处理

    # 可用性 (基于非空比例)
    usability = completeness

    return [completeness, accuracy, consistency, usability]
```

## 完整工作流

```python
def run_process(df: pd.DataFrame, plan: list, output_path: str = 'output.csv'):
    """
    执行数据处理流程

    Args:
        df: 原始数据DataFrame
        plan: 调用链表 (来自recommend)
        output_path: 输出文件路径

    Returns:
        dict: 包含 df_result, evaluation_report, chart
    """
    # 1. 执行数据处理
    print("正在执行数据处理...")
    df_result, processing_log = execute_processing(df, plan)

    # 2. 评估处理效果
    print("正在评估处理效果...")
    evaluation = evaluate_processing(df, df_result)

    # 3. 生成评价报告
    print("正在生成评价报告...")
    evaluation_report = generate_evaluation_report(df, df_result, evaluation, processing_log)

    # 4. 生成对比图表
    print("正在生成对比图表...")
    chart_paths = generate_comparison_charts(df, df_result)

    # 5. 保存处理后的数据
    df_result.to_csv(output_path, index=False)
    print(f"\n处理完成，数据已保存至: {output_path}")

    return {
        'df_result': df_result,
        'evaluation_report': evaluation_report,
        'chart': chart_paths
    }
```

## 输出格式

### df_result
处理后的 pandas DataFrame 对象

### evaluation_report
Markdown 格式的评价报告字符串，包含：
- 处理概览
- 处理步骤详情
- 缺失值变化
- 数值列统计变化
- 数据质量评分

### chart
图表文件路径字典：
```python
{
    'missing_comparison': 'missing_comparison.png',
    'distribution_comparison': 'distribution_comparison.png',
    'quality_radar': 'quality_radar.png'
}
```

## 依赖

- pandas
- numpy
- matplotlib
- invoke_data_processor_method 模块
