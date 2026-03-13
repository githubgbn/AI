"""
process 数据处理模块
负责执行数据处理、生成评价报告、生成对比图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
from typing import Optional, List, Dict, Any
from datetime import datetime
import os

# 数据处理模块路径
PROCESSOR_PATH = '/mnt/workspace/gbn/project/test_invoke_data_method/invoke_data_processor_method/scripts'


def execute_processing(df: pd.DataFrame, plan: List[Dict[str, Any]]) -> tuple:
    """
    根据调用链表执行数据处理

    Args:
        df: 原始数据DataFrame
        plan: 调用链表

    Returns:
        (处理后的DataFrame, 处理日志)
    """
    # 添加路径
    import sys
    if PROCESSOR_PATH not in sys.path:
        sys.path.append(PROCESSOR_PATH)

    try:
        from data_processor import DataProcessor
        from missing_value import MissingValueHandler
        from filter_denoiser import FilterDenoiser
        from data_transformer import DataTransformer
        from outlier_handler import OutlierHandler
        from differential_operator import DifferentialOperator
        from resampler import Resampler
        from normalizer import Normalizer
    except ImportError as e:
        print(f"[process] 警告: 无法导入数据处理模块 - {e}")
        print("[process] 将使用内置处理方法")

        # 使用内置简化方法
        return execute_processing_builtin(df, plan)

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
            if processor_name not in processors:
                raise ValueError(f"未知处理器: {processor_name}")

            proc = processors[processor_name]
            method = getattr(proc, method_name, None)

            if method is None:
                raise ValueError(f"{processor_name} 没有方法: {method_name}")

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
                'params': params,
                'status': 'success'
            })
            print(f"[process] 步骤 {step['step']}: {processor_name}.{method_name} 执行成功")

        except Exception as e:
            processing_log.append({
                'step': step['step'],
                'processor': processor_name,
                'method': method_name,
                'columns': columns,
                'params': params,
                'status': 'failed',
                'error': str(e)
            })
            print(f"[process] 步骤 {step['step']}: {processor_name}.{method_name} 执行失败: {e}")

    return df_result, processing_log


def execute_processing_builtin(df: pd.DataFrame, plan: List[Dict[str, Any]]) -> tuple:
    """
    使用内置简化方法执行处理（当invoke_data_processor_method不可用时）
    """
    df_result = df.copy()
    processing_log = []

    for step in plan:
        processor_name = step['processor']
        method_name = step['method']
        columns = step.get('columns', [])
        params = step.get('params', {})

        try:
            if processor_name == 'missing_value':
                df_result = handle_missing_builtin(df_result, method_name, columns, params)
            elif processor_name == 'normalizer':
                df_result = handle_normalize_builtin(df_result, method_name, columns, params)
            elif processor_name == 'outlier_handler':
                df_result = handle_outlier_builtin(df_result, method_name, columns, params)
            else:
                # 其他处理器暂不支持
                raise ValueError(f"内置方法不支持: {processor_name}")

            processing_log.append({
                'step': step['step'],
                'processor': processor_name,
                'method': method_name,
                'columns': columns,
                'params': params,
                'status': 'success'
            })
            print(f"[process] 步骤 {step['step']}: {processor_name}.{method_name} 执行成功")

        except Exception as e:
            processing_log.append({
                'step': step['step'],
                'processor': processor_name,
                'method': method_name,
                'columns': columns,
                'params': params,
                'status': 'failed',
                'error': str(e)
            })
            print(f"[process] 步骤 {step['step']}: {processor_name}.{method_name} 执行失败: {e}")

    return df_result, processing_log


def handle_missing_builtin(df: pd.DataFrame, method: str, columns: List[str], params: Dict) -> pd.DataFrame:
    """内置缺失值处理"""
    df = df.copy()

    if method == 'fill_mean':
        for col in columns:
            if col in df.columns and df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].mean(), inplace=True)
    elif method == 'fill_median':
        for col in columns:
            if col in df.columns and df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].median(), inplace=True)
    elif method == 'fill_mode':
        for col in columns:
            if col in df.columns:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else np.nan, inplace=True)
    elif method == 'interpolate_linear':
        for col in columns:
            if col in df.columns:
                df[col] = df[col].interpolate(method='linear')
    elif method == 'delete_column':
        threshold = params.get('threshold', 0.5)
        for col in df.columns:
            if df[col].isnull().sum() / len(df) > threshold:
                df.drop(columns=[col], inplace=True)
    elif method == 'delete_row':
        df.dropna(inplace=True)

    return df


def handle_normalize_builtin(df: pd.DataFrame, method: str, columns: List[str], params: Dict) -> pd.DataFrame:
    """内置标准化处理"""
    df = df.copy()

    if method == 'zscore_standardize':
        for col in columns:
            if col in df.columns and df[col].dtype in [np.float64, np.int64]:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[col] = (df[col] - mean) / std
    elif method == 'minmax_normalize':
        for col in columns:
            if col in df.columns and df[col].dtype in [np.float64, np.int64]:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)

    return df


def handle_outlier_builtin(df: pd.DataFrame, method: str, columns: List[str], params: Dict) -> pd.DataFrame:
    """内置异常值处理"""
    df = df.copy()

    if method == 'quartile_method':
        k = params.get('k', 1.5)
        for col in columns:
            if col in df.columns and df[col].dtype in [np.float64, np.int64]:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - k * IQR
                upper = Q3 + k * IQR
                df[col] = df[col].clip(lower, upper)
    elif method == 'three_sigma':
        n_sigma = params.get('n_sigma', 3.0)
        for col in columns:
            if col in df.columns and df[col].dtype in [np.float64, np.int64]:
                mean = df[col].mean()
                std = df[col].std()
                lower = mean - n_sigma * std
                upper = mean + n_sigma * std
                df[col] = df[col].clip(lower, upper)

    return df


def evaluate_processing(df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict[str, Any]:
    """评估处理效果"""
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
    all_cols = set(df_before.columns) | set(df_after.columns)
    for col in all_cols:
        before_missing = df_before[col].isnull().sum() if col in df_before.columns else 0
        after_missing = df_after[col].isnull().sum() if col in df_after.columns else 0
        evaluation['missing_change'][col] = {
            'before': int(before_missing),
            'after': int(after_missing),
            'change': int(after_missing - before_missing)
        }

    # 数值列变化
    numeric_cols = df_before.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in df_after.columns and df_after[col].dtype in [np.float64, np.int64]:
            evaluation['numeric_change'][col] = {
                'before_mean': float(df_before[col].mean()),
                'after_mean': float(df_after[col].mean()),
                'before_std': float(df_before[col].std()),
                'after_std': float(df_after[col].std())
            }

    return evaluation


def calculate_quality_score(df: pd.DataFrame) -> float:
    """计算数据质量评分"""
    if df.empty:
        return 0

    score = 100

    # 缺失值扣分
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    score -= (missing_cells / total_cells) * 30 if total_cells > 0 else 0

    # 数值列异常值扣分
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].notna().sum() > 0:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
            outlier_ratio = outlier_count / len(df) if len(df) > 0 else 0
            score -= outlier_ratio * 20

    return max(0, round(score, 2))


def generate_evaluation_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    evaluation: Dict[str, Any],
    processing_log: List[Dict[str, Any]]
) -> str:
    """生成处理评价报告"""
    report = []
    report.append("# 数据处理评价报告\n")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

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
        status = "✓ 成功" if log['status'] == 'success' else f"✗ 失败"
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
    report.append(f"- **提升**: {quality_score_after - quality_score_before:+.1f} 分\n")

    # 处理结果总结
    report.append("\n## 6. 处理结果总结\n")
    if quality_score_after > quality_score_before:
        report.append("✓ 数据质量有所提升\n")
    elif quality_score_after == quality_score_before:
        report.append("○ 数据质量保持不变\n")
    else:
        report.append("✗ 数据质量有所下降，请检查处理步骤\n")

    return ''.join(report)


def generate_comparison_charts(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    output_dir: str = '.'
) -> Dict[str, str]:
    """生成对比图表"""
    chart_paths = {}

    # 获取数值列
    numeric_cols = df_before.select_dtypes(include=[np.number]).columns.tolist()

    # 1. 缺失值对比图
    fig, ax = plt.subplots(figsize=(10, 6))
    cols = list(df_before.columns)
    before_missing = [df_before[col].isnull().sum() for col in cols]
    after_missing = [df_after[col].isnull().sum() for col in cols]

    x = np.arange(len(cols))
    width = 0.35

    ax.bar(x - width/2, before_missing, width, label='Before', color='coral')
    ax.bar(x + width/2, after_missing, width, label='After', color='steelblue')

    ax.set_xlabel('Column')
    ax.set_ylabel('Missing Count')
    ax.set_title('Missing Value Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(cols, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()

    missing_chart_path = os.path.join(output_dir, 'missing_comparison.png')
    plt.savefig(missing_chart_path)
    plt.close()
    chart_paths['missing_comparison'] = missing_chart_path

    # 2. 数值列分布对比图
    if numeric_cols:
        sample_cols = numeric_cols[:min(4, len(numeric_cols))]
        n_cols = len(sample_cols)
        fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 5))

        if n_cols == 1:
            axes = [axes]

        for i, col in enumerate(sample_cols):
            before_data = df_before[col].dropna()
            after_data = df_after[col].dropna()
            if len(before_data) > 0 and len(after_data) > 0:
                axes[i].hist(before_data, bins=30, alpha=0.5, label='Before', color='coral')
                axes[i].hist(after_data, bins=30, alpha=0.5, label='After', color='steelblue')
                axes[i].set_title(f'{col}')
                axes[i].legend()

        plt.tight_layout()

        dist_chart_path = os.path.join(output_dir, 'distribution_comparison.png')
        plt.savefig(dist_chart_path)
        plt.close()
        chart_paths['distribution_comparison'] = dist_chart_path

    # 3. 质量评分对比柱状图
    quality_score_before = calculate_quality_score(df_before)
    quality_score_after = calculate_quality_score(df_after)

    fig, ax = plt.subplots(figsize=(6, 5))
    scores = [quality_score_before, quality_score_after]
    labels = ['Before', 'After']
    colors = ['coral', 'steelblue']

    bars = ax.bar(labels, scores, color=colors)
    ax.set_ylabel('Quality Score')
    ax.set_title('Quality Score Comparison')
    ax.set_ylim(0, 100)

    # 添加数值标签
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.1f}',
                ha='center', va='bottom')

    plt.tight_layout()

    score_chart_path = os.path.join(output_dir, 'quality_score_comparison.png')
    plt.savefig(score_chart_path)
    plt.close()
    chart_paths['quality_score_comparison'] = score_chart_path

    return chart_paths


def run_process(
    df: pd.DataFrame,
    plan: List[Dict[str, Any]],
    output_path: str = 'output.csv'
) -> Dict[str, Any]:
    """
    执行数据处理流程

    Args:
        df: 原始数据DataFrame
        plan: 调用链表 (来自recommend)
        output_path: 输出文件路径

    Returns:
        包含 df_result, evaluation_report, chart 的字典
    """
    # 1. 执行数据处理
    print("[process] 正在执行数据处理...")
    df_result, processing_log = execute_processing(df, plan)

    # 2. 评估处理效果
    print("[process] 正在评估处理效果...")
    evaluation = evaluate_processing(df, df_result)

    # 3. 生成评价报告
    print("[process] 正在生成评价报告...")
    evaluation_report = generate_evaluation_report(df, df_result, evaluation, processing_log)

    # 4. 生成对比图表
    print("[process] 正在生成对比图表...")
    output_dir = os.path.dirname(output_path) or '.'
    chart_paths = generate_comparison_charts(df, df_result, output_dir)

    # 5. 保存处理后的数据
    df_result.to_csv(output_path, index=False)
    print(f"[process] 处理完成，数据已保存至: {output_path}")

    return {
        'df_result': df_result,
        'evaluation_report': evaluation_report,
        'chart': chart_paths,
        'processing_log': processing_log,
        'evaluation': evaluation
    }


if __name__ == '__main__':
    # 测试代码
    # 创建测试数据
    df_test = pd.DataFrame({
        'id': range(1, 101),
        'name': [f'User_{i}' for i in range(100)],
        'age': [25, 30, 35, None, 45, 50, 55, 60, 28, 32] * 10,
        'salary': [5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 5500, 6500] * 10
    })

    # 测试计划
    test_plan = [
        {
            'step': 1,
            'processor': 'missing_value',
            'method': 'fill_mean',
            'columns': ['age'],
            'params': {}
        },
        {
            'step': 2,
            'processor': 'normalizer',
            'method': 'zscore_standardize',
            'columns': ['salary'],
            'params': {}
        }
    ]

    result = run_process(df_test, test_plan, '/tmp/test_output.csv')
    print("\n" + result['evaluation_report'])
    print("\n图表已生成:")
    for name, path in result['chart'].items():
        print(f"  {name}: {path}")
