"""
recommend 策略推荐模块
负责分析数据特征、推荐处理策略、生成调用链表
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime


# invoke_data_processor_method 的7大类方法映射
PROCESSOR_METHODS = {
    # 缺失值处理
    'missing_value': {
        'fill_forward': '前向填充',
        'fill_backward': '后向填充',
        'fill_mean': '均值填充',
        'fill_median': '中位数填充',
        'fill_constant': '固定值填充',
        'fill_mode': '众数填充',
        'interpolate_linear': '线性插值',
        'interpolate_spline': '样条插值',
        'delete_row': '行删除',
        'delete_column': '列删除'
    },
    # 滤波降噪
    'filter_denoiser': {
        'median_filter': '中值滤波',
        'moving_average': '移动平均',
        'exponential_moving_average': '指数加权移动平均',
        'lowpass_filter': '低通滤波',
        'fourier_transform': '傅里叶变换'
    },
    # 数据替换
    'data_transformer': {
        'log_transform': '对数变换',
        'diff_transform': '差分变换',
        'standardize': '标准化',
        'category_encode': '类别编码'
    },
    # 异常处理
    'outlier_handler': {
        'three_sigma': '3-Sigma方法',
        'quartile_method': '四分位距法',
        'clustering_detection': '聚类检测',
        'moving_std': '移动标准差法',
        'zscore': 'Z-score方法'
    },
    # 差分算子
    'differential_operator': {
        'first_order_diff': '一阶差分',
        'lag_diff': '滞后k阶差分',
        'second_order_diff': '二阶差分',
        'seasonal_diff': '季节性差分',
        'log_diff': '对数一阶差分',
        'percent_diff': '百分比变化'
    },
    # 重采样
    'resampler': {
        'upsample_linear': '线性插值上采样',
        'upsample_polynomial': '多项式插值上采样',
        'upsample_spline': '样条插值上采样',
        'downsample_mean': '均值聚合下采样',
        'downsample_sum': '求和聚合下采样'
    },
    # 标准化/归一化
    'normalizer': {
        'zscore_standardize': 'Z-Score标准化',
        'mean_center': '均值中心化',
        'robust_standardize': '稳健标准化',
        'minmax_normalize': 'Min-Max归一化',
        'range_normalize': '指定区间缩放',
        'maxabs_normalize': '最大绝对值归一化',
        'log_normalize': '对数归一化',
        'boxcox_transform': 'Box-Cox变换'
    }
}


def analyze_features(assess_result: Dict[str, Any]) -> Dict[str, Any]:
    """分析数据特征，识别需要处理的问题"""
    features = {
        'has_missing': False,
        'has_outliers': False,
        'needs_normalization': False,
        'needs_filtering': False,
        'needs_encoding': False,
        'needs_differencing': False,
        'needs_resampling': False,
        'problem_columns': {
            'missing': [],
            'outliers': [],
            'numeric': [],
            'categorical': []
        }
    }

    # 检查缺失值
    for col, info in assess_result.get('missing_values', {}).items():
        if info['ratio'] > 0:
            features['has_missing'] = True
            features['problem_columns']['missing'].append({
                'column': col,
                'missing_ratio': info['ratio'],
                'missing_count': info['count']
            })

    # 检查异常值
    for col, info in assess_result.get('outliers', {}).items():
        if info['count'] > 0:
            features['has_outliers'] = True
            features['problem_columns']['outliers'].append({
                'column': col,
                'outlier_ratio': info['ratio'],
                'outlier_count': info['count']
            })

    # 检查数值列
    if assess_result.get('numeric_stats'):
        features['problem_columns']['numeric'] = list(assess_result['numeric_stats'].keys())
        features['needs_normalization'] = True

    # 检查类别列
    dtypes = assess_result.get('dtypes', {})
    for col, dtype in dtypes.items():
        if dtype == 'object':
            features['problem_columns']['categorical'].append(col)
            features['needs_encoding'] = True

    return features


def generate_strategy(
    assess_result: Dict[str, Any],
    features: Dict[str, Any],
    user_custom_rules: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    生成推荐策略

    Args:
        assess_result: 数据评估结果
        features: 数据特征分析结果
        user_custom_rules: 用户自定义规则列表

    Returns:
        strategy_table: 推荐策略表
    """
    strategies = []

    # 1. 缺失值处理策略
    if features['has_missing']:
        for col_info in features['problem_columns']['missing']:
            col = col_info['column']
            ratio = col_info['missing_ratio']

            if ratio > 0.5:
                # 缺失率超过50%，建议删除列
                strategies.append({
                    'step': len(strategies) + 1,
                    'category': 'missing_value',
                    'method': 'delete_column',
                    'column': col,
                    'reason': f'缺失率 {ratio*100:.1f}% 过高，建议删除列',
                    'priority': 1,
                    'params': {'threshold': 0.5}
                })
            elif ratio > 0.3:
                # 缺失率较高，建议样条插值
                strategies.append({
                    'step': len(strategies) + 1,
                    'category': 'missing_value',
                    'method': 'interpolate_spline',
                    'column': col,
                    'reason': f'缺失率 {ratio*100:.1f}%，建议样条插值',
                    'priority': 2,
                    'params': {'order': 3}
                })
            elif ratio > 0.1:
                # 缺失率中等，建议线性插值
                strategies.append({
                    'step': len(strategies) + 1,
                    'category': 'missing_value',
                    'method': 'interpolate_linear',
                    'column': col,
                    'reason': f'缺失率 {ratio*100:.1f}%，建议线性插值',
                    'priority': 2,
                    'params': {}
                })
            else:
                # 缺失率低，可以填充
                is_numeric = col in features['problem_columns']['numeric']
                strategies.append({
                    'step': len(strategies) + 1,
                    'category': 'missing_value',
                    'method': 'fill_mean' if is_numeric else 'fill_mode',
                    'column': col,
                    'reason': f'缺失率 {ratio*100:.1f}%，建议填充',
                    'priority': 3,
                    'params': {}
                })

    # 2. 异常值处理策略
    if features['has_outliers']:
        for col_info in features['problem_columns']['outliers']:
            col = col_info['column']
            ratio = col_info['outlier_ratio']

            if ratio < 0.01:
                strategies.append({
                    'step': len(strategies) + 1,
                    'category': 'outlier_handler',
                    'method': 'quartile_method',
                    'column': col,
                    'reason': f'异常值比例 {ratio*100:.2f}%，建议使用四分位距法(严格)',
                    'priority': 2,
                    'params': {'k': 1.5, 'method': 'clip'}
                })
            elif ratio < 0.05:
                strategies.append({
                    'step': len(strategies) + 1,
                    'category': 'outlier_handler',
                    'method': 'quartile_method',
                    'column': col,
                    'reason': f'异常值比例 {ratio*100:.1f}%，建议使用四分位距法',
                    'priority': 2,
                    'params': {'k': 1.5, 'method': 'clip'}
                })
            else:
                strategies.append({
                    'step': len(strategies) + 1,
                    'category': 'outlier_handler',
                    'method': 'three_sigma',
                    'column': col,
                    'reason': f'异常值比例 {ratio*100:.1f}%，建议使用3-Sigma方法',
                    'priority': 2,
                    'params': {'n_sigma': 3.0, 'method': 'clip'}
                })

    # 3. 标准化策略 (仅对数值列)
    if features['needs_normalization'] and features['problem_columns']['numeric']:
        # 根据数据分布特征选择标准化方法
        numeric_cols = features['problem_columns']['numeric']
        has_extreme_values = False

        for col in numeric_cols:
            stats = assess_result.get('numeric_stats', {}).get(col, {})
            if stats:
                cv = stats['std'] / abs(stats['mean']) if stats['mean'] != 0 else 0
                if cv > 1:  # 变异系数大于1，表示数据分散
                    has_extreme_values = True
                    break

        if has_extreme_values:
            strategies.append({
                'step': len(strategies) + 1,
                'category': 'normalizer',
                'method': 'robust_standardize',
                'column': numeric_cols,
                'reason': '数值列分布分散，建议使用稳健标准化',
                'priority': 4,
                'params': {}
            })
        else:
            strategies.append({
                'step': len(strategies) + 1,
                'category': 'normalizer',
                'method': 'zscore_standardize',
                'column': numeric_cols,
                'reason': '数值列需要标准化处理',
                'priority': 4,
                'params': {}
            })

    # 4. 类别编码策略
    if features['needs_encoding'] and features['problem_columns']['categorical']:
        cat_cols = features['problem_columns']['categorical']

        # 根据唯一值数量决定编码方式
        for col in cat_cols:
            cat_stats = assess_result.get('categorical_stats', {}).get(col, {})
            unique_count = cat_stats.get('unique_count', 0) if cat_stats else 0

            if unique_count <= 10:
                # 唯一值较少，使用独热编码
                strategies.append({
                    'step': len(strategies) + 1,
                    'category': 'data_transformer',
                    'method': 'category_encode',
                    'column': col,
                    'reason': f'唯一值数量 {unique_count} <= 10，建议使用独热编码',
                    'priority': 5,
                    'params': {'method': 'onehot'}
                })
            else:
                # 唯一值较多，使用标签编码
                strategies.append({
                    'step': len(strategies) + 1,
                    'category': 'data_transformer',
                    'method': 'category_encode',
                    'column': col,
                    'reason': f'唯一值数量 {unique_count} > 10，建议使用标签编码',
                    'priority': 5,
                    'params': {'method': 'label'}
                })

    # 5. 用户自定义规则
    if user_custom_rules:
        for rule in user_custom_rules:
            strategies.append({
                'step': len(strategies) + 1,
                'category': rule.get('category', 'custom'),
                'method': rule.get('method', 'custom'),
                'column': rule.get('column'),
                'reason': rule.get('reason', '用户自定义'),
                'priority': rule.get('priority', 10),
                'params': rule.get('params', {})
            })

    # 按优先级排序，重新编号
    strategies.sort(key=lambda x: x['priority'])
    for i, s in enumerate(strategies, 1):
        s['step'] = i

    return {'strategies': strategies}


def generate_plan(strategy_table: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    根据策略表生成调用链表

    Returns:
        调用链表列表
    """
    plan = []

    for strategy in strategy_table['strategies']:
        column = strategy.get('column')

        plan.append({
            'step': strategy['step'],
            'processor': strategy['category'],
            'method': strategy['method'],
            'columns': [column] if isinstance(column, str) else (column if column else []),
            'params': strategy.get('params', {}),
            'reason': strategy.get('reason', '')
        })

    return plan


def format_strategy_table(strategy_table: Dict[str, Any]) -> str:
    """格式化策略表为Markdown"""
    lines = ["# 推荐策略表\n"]
    lines.append("| 步骤 | 类别 | 方法 | 列名 | 原因 | 优先级 |")
    lines.append("|------|------|------|------|------|--------|")

    for s in strategy_table['strategies']:
        col = s.get('column', '-')
        if isinstance(col, list):
            col = ', '.join(map(str, col))
        lines.append(f"| {s['step']} | {s['category']} | {s['method']} | {col} | {s['reason']} | {s['priority']} |")

    lines.append(f"\n共 {len(strategy_table['strategies'])} 个策略，用户可修改后确认执行。")

    return '\n'.join(lines)


def run_recommend(
    assess_result: Dict[str, Any],
    analysis_report: Optional[str] = None,
    business_scene: Optional[str] = None,
    user_custom_rules: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    执行策略推荐流程

    Args:
        assess_result: 数据评估结果 (来自analyze)
        analysis_report: 分析报告 (可选)
        business_scene: 业务场景描述 (可选)
        user_custom_rules: 用户自定义规则列表 (可选)

    Returns:
        包含 strategy_table, plan 的字典
    """
    # 1. 分析数据特征
    print("[recommend] 正在分析数据特征...")
    features = analyze_features(assess_result)
    print(f"[recommend] 发现问题: 缺失值={features['has_missing']}, 异常值={features['has_outliers']}")

    # 2. 生成推荐策略
    print("[recommend] 正在生成推荐策略...")
    strategy_table = generate_strategy(assess_result, features, user_custom_rules)
    print(f"[recommend] 共推荐 {len(strategy_table['strategies'])} 个处理策略")

    # 3. 生成调用链表
    print("[recommend] 正在生成调用链表...")
    plan = generate_plan(strategy_table)

    # 4. 格式化策略表
    strategy_table['formatted'] = format_strategy_table(strategy_table)

    return {
        'strategy_table': strategy_table,
        'plan': plan,
        'features': features
    }


if __name__ == '__main__':
    # 测试代码
    # 模拟评估结果
    test_assess_result = {
        'shape': (100, 5),
        'columns': ['id', 'name', 'age', 'salary', 'dept'],
        'missing_values': {
            'id': {'count': 0, 'ratio': 0},
            'name': {'count': 0, 'ratio': 0},
            'age': {'count': 5, 'ratio': 0.05},
            'salary': {'count': 10, 'ratio': 0.1},
            'dept': {'count': 2, 'ratio': 0.02}
        },
        'dtypes': {'id': 'int64', 'name': 'object', 'age': 'float64', 'salary': 'float64', 'dept': 'object'},
        'numeric_stats': {
            'age': {'mean': 35.0, 'median': 34.0, 'std': 10.0, 'min': 22.0, 'max': 60.0, 'q25': 28.0, 'q75': 42.0},
            'salary': {'mean': 15000.0, 'median': 12000.0, 'std': 8000.0, 'min': 3000.0, 'max': 50000.0, 'q25': 8000.0, 'q75': 20000.0}
        },
        'outliers': {
            'age': {'count': 2, 'ratio': 0.02, 'lower_bound': 7.0, 'upper_bound': 63.0},
            'salary': {'count': 5, 'ratio': 0.05, 'lower_bound': -10000.0, 'upper_bound': 38000.0}
        },
        'quality_score': 75.0
    }

    result = run_recommend(test_assess_result)
    print("\n" + result['strategy_table']['formatted'])
    print("\n调用链表:")
    for p in result['plan']:
        print(f"  {p['step']}. {p['processor']}.{p['method']}")
