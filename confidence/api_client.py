from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from openai import OpenAI


class ParallelChatClient:
    def __init__(
        self,
        api_keys: List[str],
        base_url: Optional[str],
        model: str,
        max_retries: int,
        timeout: int,
        **request_kwargs: Any,
    ) -> None:
        if not api_keys:
            raise ValueError("api_keys cannot be empty")
        self.clients = [OpenAI(api_key=key, base_url=base_url) for key in api_keys]
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.request_kwargs = request_kwargs

    def _run_single(self, messages: List[Dict[str, str]], start_index: int) -> Dict[str, Any]:
        key_index = start_index
        errors: List[str] = []
        for attempt in range(self.max_retries + 1):
            client = self.clients[key_index]
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    timeout=self.timeout,
                    **self.request_kwargs,
                )
                return {"success": True, "content": response.choices[0].message.content or ""}
            except Exception as exc:
                errors.append(str(exc))
                if attempt == self.max_retries:
                    return {"success": False, "content": None, "errors": errors}
                key_index = (key_index + 1) % len(self.clients)
                time.sleep(1)
        return {"success": False, "content": None, "errors": errors}

    def chat_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        max_workers: Optional[int] = None,
    ) -> List[Optional[str]]:
        if not messages_list:
            return []
        worker_count = min(max_workers or len(self.clients), len(messages_list))
        results: List[Optional[str]] = [None] * len(messages_list)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {
                executor.submit(self._run_single, messages, index % len(self.clients)): index
                for index, messages in enumerate(messages_list)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    payload = future.result()
                except Exception:
                    results[index] = None
                    continue
                results[index] = payload["content"] if payload["success"] else None
        return results
