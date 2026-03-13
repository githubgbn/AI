---
name: recommend
description: 策略推荐skill。用于分析数据特征、推荐数据处理策略、生成调用链表。当用户需要基于数据分析结果推荐数据清洗策略时使用此skill。
---

# recommend 策略推荐 Skill

本skill负责分析数据特征、推荐处理策略、生成调用链表。

## 脚本

具体实现见: `scripts/recommend.py`

## 功能概述

1. **分析数据特征** - 基于评估结果识别问题
2. **推荐处理策略** - 根据invoke_data_processor_method的7大类方法推荐策略
3. **生成调用链表** - 创建可执行的处理计划

## 使用方法

### 1. 策略推荐引擎

```python
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
```

### 2. 分析数据特征并推荐策略

```python
def analyze_features(assess_result: dict) -> dict:
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
    for col, info in assess_result['missing_values'].items():
        if info['ratio'] > 0:
            features['has_missing'] = True
            features['problem_columns']['missing'].append({
                'column': col,
                'missing_ratio': info['ratio']
            })

    # 检查异常值
    for col, info in assess_result['outliers'].items():
        if info['count'] > 0:
            features['has_outliers'] = True
            features['problem_columns']['outliers'].append({
                'column': col,
                'outlier_ratio': info['ratio']
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
```

### 3. 生成策略推荐

```python
def generate_strategy(assess_result: dict, features: dict,
                      user_custom_rules: list = None) -> dict:
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
            elif ratio > 0.1:
                # 缺失率中等，建议插值
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
                strategies.append({
                    'step': len(strategies) + 1,
                    'category': 'missing_value',
                    'method': 'fill_mean' if col in features['problem_columns']['numeric'] else 'fill_mode',
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

            if ratio < 0.05:
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

    # 3. 标准化策略
    if features['needs_normalization']:
        strategies.append({
            'step': len(strategies) + 1,
            'category': 'normalizer',
            'method': 'zscore_standardize',
            'column': features['problem_columns']['numeric'],
            'reason': '数值列需要标准化处理',
            'priority': 4,
            'params': {}
        })

    # 4. 类别编码策略
    if features['needs_encoding']:
        strategies.append({
            'step': len(strategies) + 1,
            'category': 'data_transformer',
            'method': 'category_encode',
            'column': features['problem_columns']['categorical'],
            'reason': '类别列需要编码处理',
            'priority': 5,
            'params': {'method': 'onehot'}
        })

    # 5. 用户自定义规则
    if user_custom_rules:
        for rule in user_custom_rules:
            strategies.append({
                'step': len(strategies) + 1,
                'category': rule['category'],
                'method': rule['method'],
                'column': rule.get('column'),
                'reason': rule.get('reason', '用户自定义'),
                'priority': rule.get('priority', 10),
                'params': rule.get('params', {})
            })

    # 按优先级排序
    strategies.sort(key=lambda x: x['priority'])

    return {'strategies': strategies}
```

### 4. 生成调用链表

```python
def generate_plan(strategy_table: dict) -> list:
    """
    根据策略表生成调用链表

    返回格式:
    [
        {
            'step': 1,
            'processor': 'missing_value',
            'method': 'fill_mean',
            'columns': ['column_name'],
            'params': {}
        },
        ...
    ]
    """
    plan = []

    for strategy in strategy_table['strategies']:
        plan.append({
            'step': strategy['step'],
            'processor': strategy['category'],
            'method': strategy['method'],
            'columns': [strategy['column']] if strategy.get('column') and isinstance(strategy['column'], str) else strategy.get('column', []),
            'params': strategy.get('params', {})
        })

    return plan
```

## 完整工作流

```python
def run_recommend(assess_result: dict, analysis_report: str = None,
                 business_scene: str = None, user_custom_rules: list = None):
    """
    执行策略推荐流程

    Args:
        assess_result: 数据评估结果 (来自analyze)
        analysis_report: 分析报告 (可选)
        business_scene: 业务场景描述 (可选)
        user_custom_rules: 用户自定义规则列表 (可选)

    Returns:
        dict: 包含 strategy_table, plan
    """
    # 1. 分析数据特征
    print("正在分析数据特征...")
    features = analyze_features(assess_result)

    # 2. 生成推荐策略
    print("正在生成推荐策略...")
    strategy_table = generate_strategy(assess_result, features, user_custom_rules)

    # 3. 生成调用链表
    print("正在生成调用链表...")
    plan = generate_plan(strategy_table)

    print(f"\n共推荐 {len(strategy_table['strategies'])} 个处理策略")

    return {
        'strategy_table': strategy_table,
        'plan': plan
    }
```

## 输出格式

### strategy_table
```python
{
    'strategies': [
        {
            'step': 1,
            'category': 'missing_value',
            'method': 'fill_mean',
            'column': 'age',
            'reason': '缺失率 5.2%，建议填充',
            'priority': 3,
            'params': {}
        },
        ...
    ]
}
```

### plan
```python
[
    {
        'step': 1,
        'processor': 'missing_value',
        'method': 'fill_mean',
        'columns': ['age'],
        'params': {}
    },
    ...
]
```

## 策略表展示模板

```markdown
# 推荐策略表

| 步骤 | 类别 | 方法 | 列名 | 原因 | 优先级 |
|------|------|------|------|------|--------|
| 1 | missing_value | fill_mean | age | 缺失率 5.2% | 3 |
| 2 | outlier_handler | quartile_method | price | 异常值比例 3.2% | 2 |
| 3 | normalizer | zscore_standardize | [所有数值列] | 需要标准化 | 4 |

共 3 个策略，用户可修改后确认执行。
```

## 依赖

- pandas
- numpy
