from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

try:
    from colorama import Fore, init
except ModuleNotFoundError:
    class _Color:
        def __getattr__(self, _: str) -> str:
            return ""

    Fore = _Color()

    def init(*_: Any, **__: Any) -> None:
        return None

from tqdm import tqdm

from .api_client import ParallelChatClient
from .common import (
    build_question_text,
    ensure_dir,
    extract_answer,
    load_api_keys,
    load_json,
    load_prompt_bundle,
    load_yaml,
    model_slug,
    parse_sequence_cell,
    resolve_path,
    write_json,
)


init(autoreset=True)


@dataclass
class StabilityTask:
    task_info: Dict[str, Any]
    system_prompt: str
    prompts: List[str]
    max_rounds: int
    stability_threshold: int
    unknown_retry_limit: int
    messages: List[Dict[str, str]] = field(init=False)
    round_count: int = 0
    is_finished: bool = False
    stabilized_at: int = -1
    unknown_retries: int = 0
    label_trace: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.task_info["formatted_question"]},
        ]

    @property
    def id(self) -> int:
        return int(self.task_info["id"])

    def update(self, response: str) -> None:
        label = extract_answer(response)
        if label == "UNKNOWN":
            self.unknown_retries += 1
            if self.unknown_retries >= self.unknown_retry_limit:
                self.is_finished = True
            return
        self.unknown_retries = 0
        self.round_count += 1
        self.label_trace.append(label)
        self.messages.append({"role": "assistant", "content": response})
        if len(self.label_trace) >= self.stability_threshold:
            recent = self.label_trace[-self.stability_threshold :]
            if len(set(recent)) == 1:
                self.stabilized_at = self.round_count - self.stability_threshold + 1
                self.is_finished = True
                return
        if self.round_count >= self.max_rounds:
            self.is_finished = True
            return
        prompt_index = self.round_count - 1
        if prompt_index >= len(self.prompts):
            self.is_finished = True
            return
        self.messages.append({"role": "user", "content": self.prompts[prompt_index]})

    def to_result(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "original_id": self.task_info["original_id"],
            "dataset": self.task_info["dataset"],
            "category": self.task_info["category"],
            "stabilized_at": self.stabilized_at,
            "final_answer": self.label_trace[-1] if self.label_trace else "UNKNOWN",
            "label_trace": self.label_trace,
            "messages": self.messages,
        }


def load_unstable_targets(evaluation_csv: Path, copies_per_task: int) -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []
    expanded_id = 0
    with evaluation_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scores = [int(value) for value in parse_sequence_cell(row.get("score_history"))]
            if len(set(scores)) <= 1:
                continue
            options = parse_sequence_cell(row.get("options"))
            formatted_question = row.get("formatted_question") or build_question_text(row.get("question", ""), options)
            original_id = int(row["id"])
            for _ in range(copies_per_task):
                targets.append(
                    {
                        "id": expanded_id,
                        "original_id": original_id,
                        "dataset": row.get("dataset", ""),
                        "category": row.get("category", ""),
                        "formatted_question": formatted_question,
                    }
                )
                expanded_id += 1
    return targets


def load_done_ids(output_file: Path) -> set[int]:
    if not output_file.exists():
        return set()
    data = load_json(output_file)
    return {int(item["id"]) for item in data}


def load_results(output_file: Path) -> List[Dict[str, Any]]:
    if not output_file.exists():
        return []
    data = load_json(output_file)
    return data if isinstance(data, list) else []


def run(config_path: str = "config/stability.yaml") -> None:
    config = load_yaml(config_path)
    provider = config.get("provider", {})
    runtime = config.get("runtime", {})
    paths = config.get("paths", {})
    model_name = str(provider["model"])
    result_name = model_slug(model_name, config.get("result_name"))
    evaluation_csv = resolve_path(paths["evaluation_dir"]) / f"{result_name}.csv"
    output_dir = ensure_dir(paths["output_dir"])
    output_file = output_dir / f"{result_name}_stability.json"
    system_prompt, _, _, _ = load_prompt_bundle(paths["prompts"])
    prompt_data = load_json(paths["prompt_extended"])
    targets = load_unstable_targets(evaluation_csv, int(runtime.get("copies_per_task", 2)))
    done_ids = load_done_ids(output_file)
    results = load_results(output_file)
    pending = [item for item in targets if item["id"] not in done_ids]
    request_kwargs = {}
    if "temperature" in provider:
        request_kwargs["temperature"] = provider["temperature"]
    client = ParallelChatClient(
        api_keys=load_api_keys(provider),
        base_url=provider.get("base_url"),
        model=model_name,
        max_retries=int(runtime.get("max_retries", 30)),
        timeout=int(runtime.get("timeout", 60)),
        **request_kwargs,
    )
    batch_size = int(runtime.get("batch_size", 5))
    max_rounds = int(runtime.get("max_rounds", 16))
    stability_threshold = int(runtime.get("stability_threshold", 3))
    unknown_retry_limit = int(runtime.get("unknown_retry_limit", 3))
    prompts = [str(item) for item in prompt_data["stability_guide"]]
    print(f"{Fore.CYAN}model={model_name} result={result_name} pending={len(pending)}")
    task_iter = iter(pending)
    active_tasks: List[StabilityTask] = []
    progress = tqdm(total=len(targets), initial=len(done_ids), desc="stability", unit="task")
    while len(active_tasks) < batch_size:
        try:
            active_tasks.append(
                StabilityTask(
                    task_info=next(task_iter),
                    system_prompt=system_prompt,
                    prompts=prompts,
                    max_rounds=max_rounds,
                    stability_threshold=stability_threshold,
                    unknown_retry_limit=unknown_retry_limit,
                )
            )
        except StopIteration:
            break
    while active_tasks:
        responses = client.chat_batch([task.messages for task in active_tasks], max_workers=len(active_tasks))
        next_active: List[StabilityTask] = []
        updated = False
        for task, response in zip(active_tasks, responses):
            task.update(response or "")
            if task.is_finished:
                results.append(task.to_result())
                progress.update(1)
                updated = True
                print(
                    f"{Fore.GREEN}id={task.id} original_id={task.task_info['original_id']} "
                    f"stabilized_at={task.stabilized_at}"
                )
            else:
                next_active.append(task)
        if updated:
            write_json(output_file, results)
        active_tasks = next_active
        while len(active_tasks) < batch_size:
            try:
                active_tasks.append(
                    StabilityTask(
                        task_info=next(task_iter),
                        system_prompt=system_prompt,
                        prompts=prompts,
                        max_rounds=max_rounds,
                        stability_threshold=stability_threshold,
                        unknown_retry_limit=unknown_retry_limit,
                    )
                )
            except StopIteration:
                break
    progress.close()
    print(f"{Fore.CYAN}saved to {output_file}")
