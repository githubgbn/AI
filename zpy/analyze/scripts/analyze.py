"""
analyze 数据分析模块
负责加载数据、评估数据质量、生成分析报告
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime


# 数据处理模块路径
PROCESSOR_PATH = '/mnt/workspace/gbn/project/test_invoke_data_method/invoke_data_processor_method/scripts'


def load_data(file_path: str) -> pd.DataFrame:
    """
    加载CSV或Excel文件

    Args:
        file_path: 数据文件路径

    Returns:
        pandas DataFrame

    Raises:
        ValueError: 不支持的文件格式
        FileNotFoundError: 文件不存在
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")

    return df


def assess_data_quality(df: pd.DataFrame, target_columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    评估数据质量

    Args:
        df: pandas DataFrame
        target_columns: 目标列名列表 (可选)

    Returns:
        评估结果字典
    """
    # 如果指定了目标列，只评估这些列
    if target_columns:
        available_cols = [c for c in target_columns if c in df.columns]
        if available_cols:
            df = df[available_cols]

    result = {
        'shape': df.shape,
        'columns': list(df.columns),
        'missing_values': {},
        'dtypes': {},
        'numeric_stats': {},
        'outliers': {},
        'quality_score': 0,
        'assess_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
            'q75': float(df[col].quantile(0.75)),
            'missing_count': int(df[col].isnull().sum())
        }

    # 异常值检测 (IQR方法)
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = outlier_mask.sum()
        result['outliers'][col] = {
            'count': int(outlier_count),
            'ratio': float(outlier_count / len(df)) if len(df) > 0 else 0,
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound)
        }

    # 类别列分析
    categorical_cols = df.select_dtypes(include=['object']).columns
    result['categorical_stats'] = {}
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        result['categorical_stats'][col] = {
            'unique_count': int(df[col].nunique()),
            'top_values': value_counts.head(5).to_dict(),
            'missing_count': int(df[col].isnull().sum())
        }

    # 计算数据质量评分
    score = 100

    # 缺失值扣分
    if result['missing_values']:
        avg_missing = np.mean([v['ratio'] for v in result['missing_values'].values()])
        score -= avg_missing * 50

    # 异常值扣分
    if result['outliers']:
        outlier_ratios = [v['ratio'] for v in result['outliers'].values()]
        if outlier_ratios:
            avg_outlier = np.mean(outlier_ratios)
            score -= avg_outlier * 30

    result['quality_score'] = max(0, round(score, 2))

    return result


def generate_analysis_report(df: pd.DataFrame, assess_result: Dict[str, Any]) -> str:
    """生成数据分析报告"""
    report = []
    report.append("# 数据分析报告\n")
    report.append(f"**生成时间**: {assess_result.get('assess_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}\n")

    # 数据概览
    report.append("## 1. 数据概览\n")
    report.append(f"- **数据形状**: {assess_result['shape'][0]} 行 × {assess_result['shape'][1]} 列\n")
    report.append(f"- **列名**: {', '.join(assess_result['columns'])}\n")
    report.append(f"- **数据质量评分**: {assess_result['quality_score']}/100\n")

    # 缺失值分析
    report.append("\n## 2. 缺失值分析\n")
    has_missing = False
    for col, info in assess_result['missing_values'].items():
        if info['ratio'] > 0:
            has_missing = True
            report.append(f"- **{col}**: {info['count']} 个缺失值 ({info['ratio']*100:.1f}%)\n")

    if not has_missing:
        report.append("✓ 所有列无缺失值\n")

    # 数值列统计
    if assess_result.get('numeric_stats'):
        report.append("\n## 3. 数值列统计\n")
        report.append("| 列名 | 均值 | 中位数 | 标准差 | 最小值 | 最大值 |\n")
        report.append("|------|------|--------|--------|--------|--------|\n")
        for col, stats in assess_result['numeric_stats'].items():
            report.append(f"| {col} | {stats['mean']:.2f} | {stats['median']:.2f} | {stats['std']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} |\n")

    # 异常值分析
    if assess_result.get('outliers'):
        report.append("\n## 4. 异常值分析\n")
        has_outliers = False
        for col, info in assess_result['outliers'].items():
            if info['count'] > 0:
                has_outliers = True
                report.append(f"- **{col}**: {info['count']} 个异常值 ({info['ratio']*100:.1f}%)\n")
                report.append(f"  - 下界: {info['lower_bound']:.2f}, 上界: {info['upper_bound']:.2f}\n")

        if not has_outliers:
            report.append("✓ 未检测到异常值\n")

    # 类别列分析
    if assess_result.get('categorical_stats'):
        report.append("\n## 5. 类别列分析\n")
        for col, stats in assess_result['categorical_stats'].items():
            report.append(f"- **{col}**: {stats['unique_count']} 个唯一值\n")
            if stats['top_values']:
                top_str = ', '.join([f"'{k}': {v}" for k, v in list(stats['top_values'].items())[:3]])
                report.append(f"  - 常见值: {top_str}\n")

    # 数据类型
    report.append("\n## 6. 数据类型\n")
    for col, dtype in assess_result['dtypes'].items():
        report.append(f"- **{col}**: {dtype}\n")

    # 质量评分说明
    report.append("\n## 7. 质量评分说明\n")
    score = assess_result['quality_score']
    if score >= 90:
        report.append("数据质量优秀，适合直接使用。\n")
    elif score >= 70:
        report.append("数据质量良好，存在少量问题需要处理。\n")
    elif score >= 50:
        report.append("数据质量一般，建议进行清洗处理。\n")
    else:
        report.append("数据质量较差，需要进行重点清洗处理。\n")

    return ''.join(report)


def run_analyze(file_path: str, target_columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    执行数据分析流程

    Args:
        file_path: 数据文件路径 (CSV/Excel)
        target_columns: 目标列名列表 (可选)

    Returns:
        包含 df, assess_result, analysis_report 的字典
    """
    # 1. 加载数据
    print(f"[analyze] 正在加载数据: {file_path}")
    df = load_data(file_path)
    print(f"[analyze] 数据加载完成: {df.shape[0]} 行 × {df.shape[1]} 列")

    # 2. 评估数据质量
    print("[analyze] 正在评估数据质量...")
    assess_result = assess_data_quality(df, target_columns)
    print(f"[analyze] 数据质量评分: {assess_result['quality_score']}/100")

    # 3. 生成分析报告
    print("[analyze] 正在生成分析报告...")
    analysis_report = generate_analysis_report(df, assess_result)

    return {
        'df': df,
        'assess_result': assess_result,
        'analysis_report': analysis_report
    }


if __name__ == '__main__':
    # 测试代码
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        result = run_analyze(file_path)
        print("\n" + result['analysis_report'])
    else:
        print("Usage: python analyze.py <file_path>")
