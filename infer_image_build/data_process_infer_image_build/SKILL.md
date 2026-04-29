---
name: data_process_infer_image_build
description: 数据处理推理流水线。当用户需要对 CSV 文件进行数据清洗（缺失值填充、滤波降噪、异常处理）后使用 sklearn 训练的 .pkl 模型进行预测推理时使用。触发场景包括：CSV 数据处理推理、加载 pkl 模型预测、data_process_policy 数据清洗、模型推理流水线、数据清洗后预测、批量推理等。只要用户提到 CSV 文件 + 模型预测/推理，或提到 data_process_policy 包，就应该使用此 skill。
---

# 数据处理推理流水线

本 skill 用于完成"CSV 数据读取 → 数据清洗 → 模型推理 → 结果输出"的完整流水线。

## 目录结构说明

```
data_process_infer_image_build/
├── SKILL.md                          # 本文件
├── references/
│   ├── data_process_policy.py        # 数据处理 Python 模块（内含 deal_data 方法）
│   └── <模型名称>.pkl                # sklearn 训练的模型文件
└── scripts/
    └── run_pipeline.py               # 流水线执行脚本
```

- `references/data_process_policy.py`：数据处理模块文件，内含 `deal_data` 函数，执行前需确保该文件已放置于此。
- `references/*.pkl`：模型文件，执行前需确保模型已放置于此。

## 核心工作流程

整个流程分为三个阶段，用户只需提供输入 CSV 文件路径，其余步骤自动完成，**无需中间交互**。

### 阶段一：数据清洗

1. 使用 `pandas.read_csv()` 读取用户指定的 CSV 文件。
2. 将 `references/` 目录添加到 Python 模块搜索路径（`sys.path`）。
3. 导入 `data_process_policy` 模块，调用 `deal_data(raw_df)` 函数。
4. `deal_data` 会完成：缺失值填充、滤波降噪、异常值处理。
5. 得到清洗后的 DataFrame：`cleaned_df`。

### 阶段二：模型推理

1. 在 `references/` 目录下查找 `.pkl` 模型文件（若存在多个，优先使用用户指定的；若用户未指定，使用找到的第一个 `.pkl` 文件）。
2. 使用 `joblib.load()` 加载 sklearn 模型（若 joblib 失败，尝试 `pickle.load()`）。
3. 从 `cleaned_df` 中确定特征列（默认排除明显的非特征列，如 ID、索引、目标列等；若模型有 `feature_names_in_` 属性则优先使用该属性）。
4. 调用 `model.predict(X)` 生成预测结果。
5. 将预测结果作为新列（默认列名 `prediction`）追加到 `cleaned_df`。

### 阶段三：结果输出

1. 将带预测结果的 DataFrame 保存为 CSV 文件。
2. 输出路径规则：
   - 若用户指定了输出路径，使用用户指定路径。
   - 若用户未指定，默认在原 CSV 文件名后追加 `_result.csv`（如 `input.csv` → `input_result.csv`）。
3. 在控制台打印以下信息：
   - 原始数据形状（行数 × 列数）
   - 清洗后数据形状
   - 模型名称及路径
   - 预测结果前 5 行
   - 输出文件保存路径

## 执行方式

优先使用 `scripts/run_pipeline.py` 脚本执行完整流水线。脚本参数：

```bash
python scripts/run_pipeline.py \
  --input-csv <输入CSV路径> \
  --model-path references/<模型名称>.pkl \
  --output-csv <输出CSV路径（可选）>
```

若脚本不存在或需要调整，可直接用 Python 代码按上述三阶段流程执行。

## 错误处理

| 错误场景 | 处理方式 |
|---------|---------|
| CSV 读取失败 | 尝试 `utf-8`、`gbk`、`latin1` 编码依次读取；均失败则报错并提示检查文件路径 |
| data_process_policy 模块未找到 | 检查 `references/data_process_policy.py` 文件是否存在；若不存在提示用户放置 |
| deal_data 调用失败 | 捕获异常，打印原始数据前 5 行和列信息，帮助排查 |
| 模型文件未找到 | 在 `references/` 目录下列出所有 `.pkl` 文件，提示用户确认 |
| 模型加载失败 | 尝试 `joblib.load()` 和 `pickle.load()` 两种方式 |
| 特征维度不匹配 | 对比模型期望的特征数与实际列数，打印差异信息 |
| 预测失败 | 捕获异常，打印清洗后数据的 shape、dtypes 和模型类型 |

## 使用示例

**示例 1：用户指定输入和输出**
> 用 data_process_policy 清洗 /data/input.csv，然后用 references 下的 model.pkl 推理，结果保存到 /data/output.csv

Claude 直接执行完整流水线，无需交互。

**示例 2：用户只给输入 CSV**
> 对 /data/test.csv 进行数据清洗和模型推理

Claude 自动在 `references/` 下查找 data_process_policy 包和 .pkl 模型，执行后输出到 `/data/test_result.csv`。

## 注意事项

- 执行前检查 `references/data_process_policy.py` 和 `references/*.pkl` 是否存在。
- 清洗和推理过程全自动，不询问用户确认。
- 若存在多个 `.pkl` 模型文件且用户未指定，默认使用按字母排序的第一个，同时在输出中提醒用户。
- 预测结果列默认命名为 `prediction`，若该列名已存在则命名为 `prediction_1`、`prediction_2` 以此类推。
