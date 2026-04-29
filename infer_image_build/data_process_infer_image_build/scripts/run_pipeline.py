#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理推理流水线执行脚本

用法:
    python run_pipeline.py --input-csv <输入CSV路径> [--model-path <模型路径>] [--output-csv <输出CSV路径>]
"""

import argparse
import sys
import os
import pickle
import glob


def find_skill_references_dir():
    """自动定位 references 目录"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    refs_dir = os.path.join(script_dir, "..", "references")
    refs_dir = os.path.abspath(refs_dir)
    if os.path.isdir(refs_dir):
        return refs_dir
    # 备选：从当前工作目录查找
    cwd_refs = os.path.join(os.getcwd(), "references")
    if os.path.isdir(cwd_refs):
        return cwd_refs
    return None


def read_csv_auto_encoding(csv_path):
    """自动尝试多种编码读取 CSV"""
    import pandas as pd
    encodings = ["utf-8", "gbk", "latin1", "gb18030", "utf-8-sig"]
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            raise RuntimeError(f"读取 CSV 失败（编码 {enc}）: {e}")
    raise RuntimeError(f"无法读取 CSV 文件，已尝试编码: {encodings}")


def load_model(model_path):
    """加载 sklearn .pkl 模型，优先 joblib，备选 pickle"""
    try:
        import joblib
        model = joblib.load(model_path)
        return model, "joblib"
    except Exception:
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            return model, "pickle"
        except Exception as e:
            raise RuntimeError(f"无法加载模型文件 {model_path}: {e}")


def find_model_file(refs_dir, specified_path=None):
    """查找模型文件"""
    if specified_path and os.path.isfile(specified_path):
        return specified_path

    pkl_files = glob.glob(os.path.join(refs_dir, "*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"在 {refs_dir} 下未找到任何 .pkl 模型文件")

    pkl_files.sort()
    return pkl_files[0]


def get_feature_columns(df, model):
    """确定用于推理的特征列"""
    # 优先使用模型记录的 feature_names_in_
    if hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)
        missing = set(feature_cols) - set(df.columns)
        if not missing:
            return feature_cols

    # 排除常见非特征列
    exclude_keywords = ["id", "index", "target", "label", "prediction", "y", "类别", "标签", "预测", "结果"]
    feature_cols = []
    for col in df.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in exclude_keywords):
            continue
        feature_cols.append(col)

    if not feature_cols:
        feature_cols = list(df.columns)

    return feature_cols


def ensure_unique_column(df, base_name="prediction"):
    """确保预测结果列名不冲突"""
    col_name = base_name
    suffix = 1
    while col_name in df.columns:
        col_name = f"{base_name}_{suffix}"
        suffix += 1
    return col_name


def run_pipeline(input_csv, model_path=None, output_csv=None):
    """执行完整的数据处理推理流水线"""
    import pandas as pd

    # 1. 定位 references 目录
    refs_dir = find_skill_references_dir()
    if not refs_dir:
        raise RuntimeError("未找到 references 目录，请确认 skill 目录结构正确")

    # 2. 添加 references 目录到模块路径（使 Python 能找到 data_process_policy.py）
    if os.path.isdir(refs_dir) and refs_dir not in sys.path:
        sys.path.insert(0, refs_dir)

    # 3. 读取 CSV
    print(f"[1/5] 读取 CSV 文件: {input_csv}")
    raw_df = read_csv_auto_encoding(input_csv)
    print(f"      原始数据形状: {raw_df.shape[0]} 行 × {raw_df.shape[1]} 列")

    # 4. 数据清洗
    print(f"[2/5] 调用 data_process_policy.deal_data 进行数据清洗...")
    try:
        from data_process_policy import deal_data
        cleaned_df = deal_data(raw_df)
    except ImportError as e:
        raise RuntimeError(
            f"无法导入 data_process_policy 模块: {e}\n"
            f"请确认 {refs_dir}/data_process_policy.py 文件存在"
        )
    except Exception as e:
        print(f"      原始数据前 5 行:\n{raw_df.head()}")
        print(f"      列名: {list(raw_df.columns)}")
        raise RuntimeError(f"deal_data 执行失败: {e}")

    print(f"      清洗后数据形状: {cleaned_df.shape[0]} 行 × {cleaned_df.shape[1]} 列")

    # 5. 加载模型
    print(f"[3/5] 加载模型...")
    actual_model_path = find_model_file(refs_dir, model_path)
    model, loader = load_model(actual_model_path)
    print(f"      模型路径: {actual_model_path}")
    print(f"      加载方式: {loader}")
    print(f"      模型类型: {type(model).__name__}")

    # 6. 确定特征列并执行预测
    print(f"[4/5] 执行模型推理...")
    feature_cols = get_feature_columns(cleaned_df, model)
    X = cleaned_df[feature_cols]
    print(f"      使用特征列 ({len(feature_cols)} 个): {feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''}")

    # 检查特征维度
    if hasattr(model, "n_features_in_"):
        expected = model.n_features_in_
        actual = X.shape[1]
        if expected != actual:
            raise RuntimeError(
                f"特征维度不匹配: 模型期望 {expected} 个特征，实际数据有 {actual} 个特征"
            )

    try:
        predictions = model.predict(X)
    except Exception as e:
        print(f"      清洗后数据形状: {cleaned_df.shape}")
        print(f"      特征数据形状: {X.shape}")
        print(f"      特征列: {feature_cols}")
        raise RuntimeError(f"模型预测失败: {e}")

    # 7. 追加预测结果
    pred_col = ensure_unique_column(cleaned_df)
    cleaned_df[pred_col] = predictions
    print(f"      预测完成，新增列: '{pred_col}'，共 {len(predictions)} 条预测结果")

    # 8. 输出结果
    print(f"[5/5] 保存结果...")
    if not output_csv:
        base, ext = os.path.splitext(input_csv)
        output_csv = f"{base}_result{ext}"

    cleaned_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"      结果已保存至: {output_csv}")

    # 9. 打印摘要
    print("\n" + "=" * 50)
    print("推理完成摘要")
    print("=" * 50)
    print(f"输入文件:     {input_csv}")
    print(f"原始数据:     {raw_df.shape[0]} 行 × {raw_df.shape[1]} 列")
    print(f"清洗后数据:   {cleaned_df.shape[0]} 行 × {cleaned_df.shape[1]} 列")
    print(f"模型文件:     {actual_model_path}")
    print(f"预测列名:     {pred_col}")
    print(f"输出文件:     {output_csv}")
    print(f"\n预测结果前 5 行:")
    preview_cols = [pred_col]
    if len(cleaned_df.columns) > 1:
        preview_cols = list(cleaned_df.columns[:3]) + [pred_col]
    print(cleaned_df[preview_cols].head().to_string(index=False))
    print("=" * 50)

    return output_csv


def main():
    parser = argparse.ArgumentParser(description="数据处理推理流水线")
    parser.add_argument("--input-csv", required=True, help="输入 CSV 文件路径")
    parser.add_argument("--model-path", default=None, help="模型 .pkl 文件路径（可选，默认在 references 下查找）")
    parser.add_argument("--output-csv", default=None, help="输出 CSV 文件路径（可选，默认在原文件名后加 _result）")
    args = parser.parse_args()

    try:
        run_pipeline(args.input_csv, args.model_path, args.output_csv)
    except Exception as e:
        print(f"\n[错误] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
