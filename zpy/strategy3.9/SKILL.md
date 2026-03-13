---
name: strategy3.9
description: 主协调skill。用于数据处理全流程协调，包括用户交互、业务语义理解、流程编排、结果展示。调用analyze(数据分析)、recommend(策略推荐)、process(数据处理)三个子skill完成数据处理任务。
---

# strategy3.9 主协调 Skill

本skill是整个数据处理流程的主协调器，负责用户交互、业务语义理解、流程编排和结果展示。

## 功能概述

1. **用户交互** - 接收用户输入（文件路径、需求描述）
2. **业务语义理解** - 解析用户需求，确定处理目标
3. **流程编排** - 协调analyze、recommend、process三个子skill
4. **结果展示** - 输出分析报告、推荐策略表、评价报告、对比图表

## 架构

```
strategy3.9 (主协调)
      │
      ├─→ analyze (数据分析)
      │       └─→ df, assess_result, analysis_report
      │
      ├─→ recommend (策略推荐)
      │       └─→ strategy_table, plan
      │
      └─→ process (数据处理)
              └─→ df_result, evaluation_report, chart
```

## 使用方法

### 完整工作流

```python
import pandas as pd
import sys

# 添加子skill路径
SKILL_PATHS = {
    'analyze': '/mnt/workspace/gbn/project/zpy/analyze',
    'recommend': '/mnt/workspace/gbn/project/zpy/recommend',
    'process': '/mnt/workspace/gbn/project/zpy/process'
}

PROCESSOR_PATH = '/mnt/workspace/gbn/project/test_invoke_data_method/invoke_data_processor_method/scripts'
sys.path.append(PROCESSOR_PATH)


def run_full_workflow(file_path: str, target_columns: list = None,
                      business_scene: str = None, user_custom_rules: list = None,
                      output_dir: str = '.'):
    """
    执行完整的数据处理流程

    Args:
        file_path: 数据文件路径 (CSV/Excel)
        target_columns: 目标列名列表 (可选)
        business_scene: 业务场景描述 (可选)
        user_custom_rules: 用户自定义规则列表 (可选)
        output_dir: 输出目录

    Returns:
        dict: 包含所有结果
    """
    results = {}

    # ==================== 步骤1: 用户交互 ====================
    print("=" * 60)
    print("欢迎使用数据处理主协调系统")
    print("=" * 60)
    print(f"\n输入文件: {file_path}")
    if target_columns:
        print(f"目标列: {target_columns}")
    if business_scene:
        print(f"业务场景: {business_scene}")

    # ==================== 步骤2: 数据分析 (analyze) ====================
    print("\n" + "=" * 60)
    print("步骤 1/3: 数据分析")
    print("=" * 60)

    from analyze.SKILL import run_analyze

    analyze_result = run_analyze(file_path, target_columns)
    df = analyze_result['df']
    assess_result = analyze_result['assess_result']
    analysis_report = analyze_result['analysis_report']

    results['analyze'] = analyze_result

    # 展示分析报告
    print("\n" + analysis_report)

    # ==================== 步骤3: 策略推荐 (recommend) ====================
    print("\n" + "=" * 60)
    print("步骤 2/3: 策略推荐")
    print("=" * 60)

    from recommend.SKILL import run_recommend

    recommend_result = run_recommend(
        assess_result=assess_result,
        analysis_report=analysis_report,
        business_scene=business_scene,
        user_custom_rules=user_custom_rules
    )

    strategy_table = recommend_result['strategy_table']
    plan = recommend_result['plan']

    results['recommend'] = recommend_result

    # 展示推荐策略表
    print("\n" + format_strategy_table(strategy_table))

    # ==================== 步骤4: 用户确认策略 ====================
    print("\n" + "=" * 60)
    print("策略确认")
    print("=" * 60)

    # 用户可以修改plan后确认执行
    # 这里直接使用推荐的策略执行
    confirmed_plan = plan

    print(f"已确认执行 {len(confirmed_plan)} 个处理步骤")

    # ==================== 步骤5: 数据处理 (process) ====================
    print("\n" + "=" * 60)
    print("步骤 3/3: 数据处理")
    print("=" * 60)

    from process.SKILL import run_process

    output_path = f"{output_dir}/processed_data.csv"

    process_result = run_process(
        df=df,
        plan=confirmed_plan,
        output_path=output_path
    )

    df_result = process_result['df_result']
    evaluation_report = process_result['evaluation_report']
    chart_paths = process_result['chart']

    results['process'] = process_result

    # 展示评价报告
    print("\n" + evaluation_report)

    # ==================== 步骤6: 结果展示 ====================
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  - 处理后数据: {output_path}")
    for chart_name, chart_path in chart_paths.items():
        print(f"  - {chart_name}: {chart_path}")

    # 汇总所有结果
    results['summary'] = {
        'input_file': file_path,
        'output_file': output_path,
        'original_shape': assess_result['shape'],
        'processed_shape': df_result.shape,
        'quality_score_before': assess_result['quality_score'],
        'quality_score_after': calculate_final_score(df_result),
        'strategies_applied': len(plan),
        'charts': chart_paths
    }

    return results


def format_strategy_table(strategy_table: dict) -> str:
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


def calculate_final_score(df: pd.DataFrame) -> float:
    """计算最终数据质量评分"""
    score = 100
    for col in df.columns:
        missing_ratio = df[col].isnull().sum() / len(df) if len(df) > 0 else 0
        score -= missing_ratio * 30
    return max(0, round(score, 2))
```

## 交互式用户流程

### 1. 接收用户输入

```python
def collect_user_input():
    """收集用户输入"""
    print("\n请提供以下信息:\n")

    # 文件路径
    file_path = input("1. 数据文件路径 (CSV/Excel): ").strip()
    while not file_path:
        print("文件路径不能为空，请重新输入")
        file_path = input("数据文件路径: ").strip()

    # 目标列 (可选)
    cols_input = input("\n2. 目标列 (可选，用逗号分隔，直接回车跳过): ").strip()
    target_columns = [c.strip() for c in cols_input.split(',')] if cols_input else None

    # 业务场景 (可选)
    business_scene = input("\n3. 业务场景描述 (可选，直接回车跳过): ").strip() or None

    # 用户自定义规则 (可选)
    print("\n4. 是否添加自定义处理规则? (y/n): ")
    add_rules = input().strip().lower() == 'y'
    user_custom_rules = []
    if add_rules:
        print("请输入自定义规则 (输入 'done' 完成):")
        while True:
            rule_input = input("规则 (格式: category.method.column): ").strip()
            if rule_input.lower() == 'done':
                break
            if rule_input:
                parts = rule_input.split('.')
                if len(parts) >= 2:
                    user_custom_rules.append({
                        'category': parts[0],
                        'method': parts[1],
                        'column': parts[2] if len(parts) > 2 else None,
                        'reason': '用户自定义'
                    })

    return {
        'file_path': file_path,
        'target_columns': target_columns,
        'business_scene': business_scene,
        'user_custom_rules': user_custom_rules
    }
```

### 2. 策略确认与修改

```python
def confirm_and_modify_strategy(strategy_table: dict):
    """
    展示策略表并允许用户修改

    Args:
        strategy_table: 推荐的策略表

    Returns:
        用户确认后的plan
    """
    print("\n" + "=" * 60)
    print("推荐策略表")
    print("=" * 60)
    print(format_strategy_table(strategy_table))

    print("\n请选择操作:")
    print("  1. 确认执行当前策略")
    print("  2. 删除某个策略")
    print("  3. 添加自定义策略")
    print("  4. 重新生成策略")

    choice = input("\n请输入选项 (1/2/3/4): ").strip()

    if choice == '1':
        return strategy_table['strategies']
    elif choice == '2':
        # 删除策略
        step_to_delete = int(input("请输入要删除的步骤号: "))
        return [s for s in strategy_table['strategies'] if s['step'] != step_to_delete]
    elif choice == '3':
        # 添加自定义策略
        return add_custom_strategy(strategy_table['strategies'])
    else:
        # 重新生成 (返回空以触发重新推荐)
        return None
```

## 子Skill接口

### analyze (数据分析)

**输入**:
- `file_path`: 数据文件路径
- `target_columns`: 目标列列表 (可选)

**输出**:
- `df`: DataFrame对象
- `assess_result`: 数据评估结果
- `analysis_report`: 分析报告 (Markdown)

### recommend (策略推荐)

**输入**:
- `assess_result`: 数据评估结果
- `analysis_report`: 分析报告
- `business_scene`: 业务场景 (可选)
- `user_custom_rules`: 用户自定义规则 (可选)

**输出**:
- `strategy_table`: 推荐策略表
- `plan`: 调用链表

### process (数据处理)

**输入**:
- `df`: 原始数据DataFrame
- `plan`: 调用链表
- `output_path`: 输出文件路径

**输出**:
- `df_result`: 处理后数据
- `evaluation_report`: 评价报告 (Markdown)
- `chart`: 图表路径字典

## 错误处理

```python
def handle_errors(e: Exception, context: str):
    """统一错误处理"""
    error_messages = {
        'file_not_found': f"文件未找到，请检查路径是否正确",
        'invalid_format': f"不支持的文件格式，请使用 CSV 或 Excel",
        'empty_data': f"数据为空，请检查文件内容",
        'processing_error': f"处理失败: {str(e)}"
    }

    print(f"\n错误: {context}")
    print(f"详情: {error_messages.get(type(e).__name__, str(e))}")

    # 建议用户检查
    print("\n请检查:")
    print("  1. 文件路径是否正确")
    print("  2. 文件格式是否支持")
    print("  3. 文件是否为空")

    return None
```

## 结果展示模板

### 最终报告

```markdown
# 数据处理完成报告

## 处理概览

| 项目 | 数值 |
|------|------|
| 原始数据 | {行数} 行 × {列数} 列 |
| 处理后数据 | {行数} 行 × {列数} 列 |
| 应用策略数 | {数量} |
| 处理前质量评分 | {分数}/100 |
| 处理后质量评分 | {分数}/100 |
| 质量提升 | {分数} 分 |

## 输出文件

- 处理后数据: `processed_data.csv`
- 缺失值对比图: `missing_comparison.png`
- 分布对比图: `distribution_comparison.png`
- 质量雷达图: `quality_radar.png`

## 详细报告

- [分析报告](#analysis_report)
- [推荐策略表](#strategy_table)
- [评价报告](#evaluation_report)
```

## 依赖

- pandas
- numpy
- matplotlib
- analyze skill
- recommend skill
- process skill
- invoke_data_processor_method 模块
