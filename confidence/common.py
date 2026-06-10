from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
VALID_CHOICES = set("ABCDEFGHIJKL")
OPTION_PATTERN = r"[A-L]"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def ensure_dir(path: str | Path) -> Path:
    target = resolve_path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_json(path: str | Path) -> Any:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> None:
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def model_slug(model_name: str, result_name: str | None = None) -> str:
    source = result_name or model_name.rsplit("/", 1)[-1]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", source).strip("_")
    return slug or "model"


def load_api_keys(provider_config: Dict[str, Any]) -> List[str]:
    keys = [str(key).strip() for key in provider_config.get("api_keys", []) if str(key).strip()]
    env_name = str(provider_config.get("api_keys_env", "")).strip()
    if not keys and env_name:
        raw = os.getenv(env_name, "")
        keys = [item.strip() for item in raw.split(",") if item.strip()]
    if not keys:
        raise ValueError("No API keys found. Fill provider.api_keys or set provider.api_keys_env.")
    return keys


def ordered_prompts(prompt_block: Dict[str, Any]) -> List[str]:
    pairs = [(key, value) for key, value in prompt_block.items() if key.startswith("cri_")]
    pairs.sort(key=lambda item: int(item[0].split("_")[1]))
    return [str(value) for _, value in pairs]


def load_prompt_bundle(path: str | Path) -> tuple[str, List[str], List[str], List[str]]:
    data = load_json(path)
    return (
        str(data[0]["system"]),
        ordered_prompts(data[1]),
        ordered_prompts(data[2]),
        ordered_prompts(data[3]),
    )


def load_questions(path: str | Path) -> List[Dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported question format in {path}")


def build_question_text(question: str, options: Sequence[Any]) -> str:
    labels = [chr(65 + index) for index in range(len(options))]
    body = "\n".join(f"{label}. {option}" for label, option in zip(labels, options))
    return f"{question}\n{body}\nAnswer:"


def format_question(item: Dict[str, Any]) -> tuple[str, str]:
    options = item["options"]
    answer_label = chr(65 + int(item["answer_index"]))
    return build_question_text(str(item["question"]), options), answer_label


def parse_sequence_cell(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(text)
            except Exception:
                continue
            if isinstance(parsed, list):
                return parsed
    return []


def evaluation_fieldnames() -> List[str]:
    return [
        "id",
        "dataset",
        "category",
        "question",
        "options",
        "formatted_question",
        "correct_answer",
        "final_status",
        "score_history",
        "label_history",
        "prompt_flow",
        "rounds_taken",
    ]


def extract_answer(response_text: str | None) -> str:
    if not response_text:
        return "UNKNOWN"
    text = response_text.strip()
    boxed_matches = re.findall(r"\\boxed\s*\{\s*(" + OPTION_PATTERN + r")\s*\}", text)
    if boxed_matches:
        return boxed_matches[-1].upper()
    bold_matches = re.findall(r"\*\*(" + OPTION_PATTERN + r")\*\*", text)
    if bold_matches:
        return bold_matches[-1].upper()
    strong_patterns = [
        r"(?:Answer|Option|Choice)(?:\s*is)?\s*[:：]?\s*(?:Option|Choice)?\s*(" + OPTION_PATTERN + r")\b",
        r"The answer is\s*(?:Option|Choice)?\s*(" + OPTION_PATTERN + r")\b",
        r"Select\s*(" + OPTION_PATTERN + r")\b",
    ]
    for pattern in strong_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches and matches[-1].upper() in VALID_CHOICES:
            return matches[-1].upper()
    end_patterns = [
        r"[\(\[](" + OPTION_PATTERN + r")[\)\]]\s*$",
        r"^(" + OPTION_PATTERN + r")\s*$",
        r"\b(" + OPTION_PATTERN + r")\.\s*$",
    ]
    for pattern in end_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    search_area = text[-200:] if len(text) > 200 else text
    fallback = re.findall(r"(?<![a-zA-Z])(" + OPTION_PATTERN + r")(?![a-zA-Z])", search_area)
    if fallback:
        return fallback[-1].upper()
    return "UNKNOWN"
