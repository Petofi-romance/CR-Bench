# Confidence

## 简介

这个目录是整理后的最终上传版，只保留复现实验所需的代码、配置和数据。

实验流程分为四步：

1. `python run_evaluate.py`
2. `python run_persistence.py`
3. `python run_stability.py`
4. `python run_analysis.py`

每一步都不需要命令行参数，运行配置统一放在 `config/*.yaml`。

## 目录结构

```text
.
├── confidence/
├── config/
├── data/
├── outputs/
├── run_evaluate.py
├── run_persistence.py
├── run_stability.py
├── run_analysis.py
└── requirements.txt
```

## 环境

- Python 3.10+
- 可访问兼容 OpenAI Chat Completions 的接口

安装依赖：

```bash
pip install -r requirements.txt
```

## 配置

默认配置文件：

- `config/evaluate.yaml`
- `config/persistence.yaml`
- `config/stability.yaml`
- `config/analysis.yaml`

需要修改的主要字段：

- `provider.model`
- `provider.base_url`
- `result_name`
- `provider.api_keys` 或 `provider.api_keys_env`

推荐做法是设置环境变量，然后保持 YAML 不放密钥：

```bash
export CONFIDENCE_API_KEYS="key1,key2"
```

如果直接写入 YAML，把 `provider.api_keys` 改成列表即可。

`result_name` 决定输出文件名。`persistence` 和 `stability` 会读取 `outputs/evaluation/{result_name}.csv`，所以三个配置里的 `result_name` 要保持一致。

## 运行顺序

### 1. 主评测

```bash
python run_evaluate.py
```

输出：

- `outputs/evaluation/{result_name}.csv`
- `outputs/history_logs/{result_name}/id_*.json`

### 2. Persistence 测试

```bash
python run_persistence.py
```

输出：

- `outputs/persistence/{result_name}_persistence.json`

### 3. Stability 测试

```bash
python run_stability.py
```

输出：

- `outputs/stability/{result_name}_stability.json`

### 4. 统计分析

```bash
python run_analysis.py
```

输出：

- `outputs/analysis/leaderboard.json`

## 数据

`data/benchmark_questions.json` 是实验直接使用的 326 条题目数据，包含以下 5 个来源的数据抽样结果：

- `MMLU-Pro`: 112
- `TruthfulQA`: 76
- `MMLU-Redux`: 48
- `Ai2_arc`: 45
- `Commonsense_qa`: 45

另外包含：

- `data/prompt.json`: 主评测诱导提示词
- `data/prompt_extended.json`: persistence 和 stability 使用的扩展提示词

## 输出格式说明

主评测 CSV 里保留了：

- 原始题目
- 选项
- 完整格式化题面
- 正确答案
- 每轮分数轨迹
- 每轮答案轨迹
- prompt 切换轨迹

这样后续 `persistence` 和 `stability` 会直接复用完整题面，不会丢失选项信息。

## 复现建议

1. 先改好 3 个运行配置文件里的模型与密钥配置。
2. 执行 `python run_evaluate.py`。
3. 确认 `outputs/evaluation/` 已生成对应 CSV。
4. 再执行 `python run_persistence.py`、`python run_stability.py`。
5. 最后执行 `python run_analysis.py` 汇总结果。
