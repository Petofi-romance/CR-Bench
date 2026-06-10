from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

try:
    from colorama import Fore, Style, init
except ModuleNotFoundError:
    class _Color:
        def __getattr__(self, _: str) -> str:
            return ""

    Fore = Style = _Color()

    def init(*_: Any, **__: Any) -> None:
        return None

from tqdm import tqdm

from .api_client import ParallelChatClient
from .common import (
    ensure_dir,
    evaluation_fieldnames,
    extract_answer,
    format_question,
    load_api_keys,
    load_prompt_bundle,
    load_questions,
    load_yaml,
    model_slug,
    resolve_path,
    write_json,
)


init(autoreset=True)


@dataclass
class EvaluationTask:
    item: Dict[str, Any]
    system_prompt: str
    error_prompts: List[str]
    correct_prompts: List[str]
    stability_prompts: List[str]
    max_rounds: int
    consistency_threshold: int
    formatted_question: str = field(init=False)
    correct_answer: str = field(init=False)
    messages: List[Dict[str, str]] = field(init=False)
    history_answers: List[str] = field(default_factory=list)
    history_labels: List[str] = field(default_factory=list)
    history_scores: List[int] = field(default_factory=list)
    history_prompts: List[str] = field(default_factory=list)
    round_count: int = 0
    current_mode: str = "initial"
    is_finished: bool = False
    mode_index: Dict[str, int] = field(default_factory=lambda: {"error": 0, "correct": 0, "stability": 0})
    status_message: str = ""

    def __post_init__(self) -> None:
        self.formatted_question, self.correct_answer = format_question(self.item)
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.formatted_question},
        ]

    @property
    def id(self) -> int:
        return int(self.item["id"])

    def update(self, model_response: str) -> None:
        self.round_count += 1
        predicted = extract_answer(model_response)
        score = int(predicted == self.correct_answer)
        self.messages.append({"role": "assistant", "content": model_response})
        self.history_answers.append(model_response)
        self.history_labels.append(predicted)
        self.history_scores.append(score)
        if self._is_consistent():
            self.is_finished = True
            self.status_message = f"stopped after {self.consistency_threshold} identical answers"
            return
        if self.round_count >= self.max_rounds:
            self.is_finished = True
            self.status_message = "max rounds reached"
            return
        next_mode = "stability" if self._should_trigger_stability() else self.current_mode
        next_prompt = self._next_prompt(next_mode, score)
        if next_prompt:
            self.current_mode = next_mode
            self.messages.append({"role": "user", "content": next_prompt})
            return
        self.is_finished = True

    def _is_consistent(self) -> bool:
        if len(self.history_labels) < self.consistency_threshold:
            return False
        recent = self.history_labels[-self.consistency_threshold :]
        return len(set(recent)) == 1

    def _should_trigger_stability(self) -> bool:
        if self.current_mode == "stability" or len(self.history_scores) < 2:
            return False
        if len(self.history_scores) >= 3:
            if (
                self.history_scores[-3] == 1
                and self.history_scores[-2] == 0
                and self.history_labels[-1] != self.history_labels[-2]
            ):
                return True
        return (
            self.history_scores[-1] == 0
            and self.history_scores[-2] == 0
            and self.history_labels[-1] != self.history_labels[-2]
        )

    def _next_prompt(self, next_mode: str, current_score: int) -> str | None:
        if next_mode == "stability":
            return self._take_prompt("stability")
        if self.current_mode == "initial":
            if current_score == 1:
                self.status_message = "initially correct, switching to error prompts"
                return self._switch_and_take("error")
            self.status_message = "initially wrong, switching to correct prompts"
            return self._switch_and_take("correct")
        if self.current_mode == "error":
            if current_score == 1:
                self.status_message = "still correct, continue error prompts"
                return self._take_prompt("error")
            self.status_message = "turned wrong, switching to correct prompts"
            self.mode_index["correct"] = 0
            return self._switch_and_take("correct")
        if current_score == 0:
            self.status_message = "still wrong, continue correct prompts"
            return self._take_prompt("correct")
        self.status_message = "turned correct, switching to error prompts"
        self.mode_index["error"] = 0
        return self._switch_and_take("error")

    def _switch_and_take(self, mode: str) -> str | None:
        self.current_mode = mode
        return self._take_prompt(mode)

    def _take_prompt(self, mode: str) -> str | None:
        pools = {
            "error": self.error_prompts,
            "correct": self.correct_prompts,
            "stability": self.stability_prompts,
        }
        index = self.mode_index[mode]
        pool = pools[mode]
        if index >= len(pool):
            return None
        self.mode_index[mode] += 1
        self.history_prompts.append(f"{mode}_{index + 1}")
        return pool[index]

    def to_row(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "dataset": self.item.get("dataset", ""),
            "category": self.item.get("category", ""),
            "question": self.item["question"],
            "options": json.dumps(self.item.get("options", []), ensure_ascii=False),
            "formatted_question": self.formatted_question,
            "correct_answer": self.correct_answer,
            "final_status": "Finished",
            "score_history": json.dumps(self.history_scores),
            "label_history": json.dumps(self.history_labels),
            "prompt_flow": json.dumps(self.history_prompts),
            "rounds_taken": self.round_count,
        }

    def to_history(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "dataset": self.item.get("dataset", ""),
            "category": self.item.get("category", ""),
            "question": self.item["question"],
            "formatted_question": self.formatted_question,
            "correct_answer": self.correct_answer,
            "rounds_taken": self.round_count,
            "score_history": self.history_scores,
            "label_history": self.history_labels,
            "prompt_flow": self.history_prompts,
            "messages": self.messages,
        }


def load_done_ids(output_csv: Path) -> set[int]:
    if not output_csv.exists():
        return set()
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        done_ids = set()
        for row in reader:
            value = row.get("id")
            if value is None or value == "":
                continue
            done_ids.add(int(value))
        return done_ids


def ensure_output_csv(output_csv: Path) -> None:
    if output_csv.exists() and output_csv.stat().st_size > 0:
        return
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=evaluation_fieldnames())
        writer.writeheader()


def append_rows(output_csv: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with output_csv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=evaluation_fieldnames())
        writer.writerows(rows)


def run(config_path: str = "config/evaluate.yaml") -> None:
    config = load_yaml(config_path)
    provider = config.get("provider", {})
    runtime = config.get("runtime", {})
    paths = config.get("paths", {})
    model_name = str(provider["model"])
    result_name = model_slug(model_name, config.get("result_name"))
    output_dir = ensure_dir(paths["output_dir"])
    history_root = ensure_dir(paths["history_dir"])
    output_csv = output_dir / f"{result_name}.csv"
    history_dir = history_root / result_name
    history_dir.mkdir(parents=True, exist_ok=True)
    questions = load_questions(paths["questions"])
    system_prompt, error_prompts, correct_prompts, stability_prompts = load_prompt_bundle(paths["prompts"])
    done_ids = load_done_ids(output_csv)
    ensure_output_csv(output_csv)
    pending_items = [item for item in questions if int(item["id"]) not in done_ids]
    request_kwargs = {}
    if "temperature" in provider:
        request_kwargs["temperature"] = provider["temperature"]
    client = ParallelChatClient(
        api_keys=load_api_keys(provider),
        base_url=provider.get("base_url"),
        model=model_name,
        max_retries=int(runtime.get("max_retries", 5)),
        timeout=int(runtime.get("timeout", 60)),
        **request_kwargs,
    )
    batch_size = int(runtime.get("batch_size", 8))
    max_rounds = int(runtime.get("max_rounds", 8))
    consistency_threshold = int(runtime.get("consistency_threshold", 6))
    print(f"{Fore.CYAN}model={model_name} result={result_name} pending={len(pending_items)}")
    task_iter = iter(pending_items)
    active_tasks: List[EvaluationTask] = []
    progress = tqdm(total=len(questions), initial=len(done_ids), desc="evaluation", unit="task")
    while len(active_tasks) < batch_size:
        try:
            active_tasks.append(
                EvaluationTask(
                    item=next(task_iter),
                    system_prompt=system_prompt,
                    error_prompts=error_prompts,
                    correct_prompts=correct_prompts,
                    stability_prompts=stability_prompts,
                    max_rounds=max_rounds,
                    consistency_threshold=consistency_threshold,
                )
            )
        except StopIteration:
            break
    while active_tasks:
        responses = client.chat_batch([task.messages for task in active_tasks], max_workers=len(active_tasks))
        completed_rows: List[Dict[str, Any]] = []
        next_active: List[EvaluationTask] = []
        for task, response in zip(active_tasks, responses):
            task.update(response or "")
            if task.is_finished:
                completed_rows.append(task.to_row())
                write_json(history_dir / f"id_{task.id}.json", task.to_history())
                progress.update(1)
                score_trace = ",".join(str(value) for value in task.history_scores)
                print(
                    f"{Fore.GREEN}id={task.id}{Style.RESET_ALL} "
                    f"scores=[{score_trace}] rounds={task.round_count} {task.status_message}"
                )
            else:
                next_active.append(task)
        append_rows(output_csv, completed_rows)
        active_tasks = next_active
        while len(active_tasks) < batch_size:
            try:
                active_tasks.append(
                    EvaluationTask(
                        item=next(task_iter),
                        system_prompt=system_prompt,
                        error_prompts=error_prompts,
                        correct_prompts=correct_prompts,
                        stability_prompts=stability_prompts,
                        max_rounds=max_rounds,
                        consistency_threshold=consistency_threshold,
                    )
                )
            except StopIteration:
                break
    progress.close()
    print(f"{Fore.CYAN}saved to {resolve_path(output_csv)}")
