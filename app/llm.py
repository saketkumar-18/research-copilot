"""LLM client: OpenAI-compatible chat completions with retry + JSON extraction.

Works with any OpenAI-compatible endpoint (OpenRouter, tokenrouter, vLLM...).
Handles reasoning models that emit `reasoning_content` before `content`.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

import httpx

from .config import settings


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.llm_api_key
        self.model = model or settings.llm_model
        self.timeout = timeout or settings.llm_timeout_s
        self.max_retries = max_retries if max_retries is not None else settings.llm_max_retries
        self.max_tokens = max_tokens or settings.llm_max_tokens

    # ------------------------------------------------------------------ core
    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.4,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        """Single chat completion. Returns the assistant's text content."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_err = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                    )
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    time.sleep(min(2 ** attempt, 20))
                    continue
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                msg = choice["message"]
                content = (msg.get("content") or "").strip()
                finish_reason = choice.get("finish_reason", "")
                if not content:
                    # reasoning model may have burned all tokens on reasoning
                    last_err = "empty content (reasoning consumed budget?)"
                    if attempt < self.max_retries:
                        payload["max_tokens"] = int((max_tokens or self.max_tokens) * 1.5)
                        time.sleep(2)
                        continue
                    raise LLMError(last_err)
                if finish_reason == "length" and attempt < self.max_retries:
                    # output was cut off mid-generation — retry with a
                    # bigger budget rather than returning a truncated report
                    last_err = "truncated output (finish_reason=length)"
                    payload["max_tokens"] = int(payload["max_tokens"] * 1.6)
                    time.sleep(2)
                    continue
                return content
            except httpx.HTTPError as e:
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(min(2 ** attempt, 20))
        raise LLMError(f"LLM call failed after {self.max_retries} attempts: {last_err}")

    # ------------------------------------------------------------------ json
    def chat_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """Chat completion that must return a JSON object. Tolerant parser."""
        raw = self.chat(
            system + "\n\nRespond ONLY with a valid JSON object. No prose, no markdown fences.",
            user,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        return extract_json(raw)


def extract_json(text: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from LLM output."""
    text = text.strip()
    # strip markdown fences
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # find the outermost {...}
    start = text.find("{")
    if start == -1:
        raise LLMError(f"No JSON object in LLM output: {text[:200]}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    break
    raise LLMError(f"Unparseable JSON in LLM output: {text[:200]}")
