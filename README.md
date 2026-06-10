# Confidence

## Overview

This directory is the final upload package. It keeps only the main evaluation pipeline.

Run:

```bash
python run_evaluate.py
```

The script does not require command-line arguments. Runtime settings are stored in `config/evaluate.yaml`.

## Directory Structure

```text
.
├── confidence/
├── config/
├── data/
├── outputs/
├── run_evaluate.py
└── requirements.txt
```

## Environment

- Python 3.10+
- Access to an OpenAI-compatible Chat Completions API endpoint

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config/evaluate.yaml` before running.

Main fields:

- `provider.model`
- `provider.base_url`
- `result_name`
- `provider.api_keys` or `provider.api_keys_env`

Recommended setup:

```bash
export CONFIDENCE_API_KEYS="key1,key2"
```

If you prefer to place keys in YAML, fill `provider.api_keys` with a list.

## Run

```bash
python run_evaluate.py
```

Outputs:

- `outputs/evaluation/{result_name}.csv`
- `outputs/history_logs/{result_name}/id_*.json`

## Data

`data/benchmark_questions.json` contains the 326 benchmark questions used in the evaluation.

Source distribution:

- `MMLU-Pro`: 112
- `TruthfulQA`: 76
- `MMLU-Redux`: 48
- `Ai2_arc`: 45
- `Commonsense_qa`: 45

`data/prompt.json` contains the prompt set used by the evaluation pipeline.

## Output Format

The evaluation CSV keeps:

- original question text
- answer options
- formatted question prompt
- correct answer
- per-round score trace
- per-round answer trace
- prompt-switch trace

## Reproduction

1. Update `config/evaluate.yaml`.
2. Set `CONFIDENCE_API_KEYS` if needed.
3. Run `python run_evaluate.py`.
4. Check `outputs/evaluation/` for the result CSV.
