"""Application configuration.

All runtime configuration is driven by environment variables so the same
code runs locally and on Render without changes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name, "")
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env(name, "")
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


@dataclass
class Settings:
    # --- LLM ---
    llm_base_url: str = field(
        default_factory=lambda: _env("LLM_BASE_URL", "https://api.tokenrouter.com/v1")
    )
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", ""))
    llm_model: str = field(
        default_factory=lambda: _env("LLM_MODEL", "qwen/qwen3.8-max-free")
    )
    llm_timeout_s: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_S", 300.0))
    llm_max_retries: int = field(default_factory=lambda: _env_int("LLM_MAX_RETRIES", 5))
    # Reasoning models (e.g. qwen3.8-max) spend tokens on reasoning before
    # emitting content; keep the ceiling generous so content is never cut off.
    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 8000))

    # --- Search ---
    search_backend: str = field(default_factory=lambda: _env("SEARCH_BACKEND", "ddgs"))
    search_results_per_query: int = field(
        default_factory=lambda: _env_int("SEARCH_RESULTS_PER_QUERY", 5)
    )
    max_fetch_pages: int = field(default_factory=lambda: _env_int("MAX_FETCH_PAGES", 8))
    fetch_timeout_s: float = field(default_factory=lambda: _env_float("FETCH_TIMEOUT_S", 12.0))
    fetch_max_chars: int = field(default_factory=lambda: _env_int("FETCH_MAX_CHARS", 4000))

    # --- Pipeline ---
    max_revision_rounds: int = field(default_factory=lambda: _env_int("MAX_REVISION_ROUNDS", 2))
    max_report_words_target: int = field(
        default_factory=lambda: _env_int("REPORT_WORDS_TARGET", 1400)
    )

    # --- Evaluation ---
    judge_weight: float = field(default_factory=lambda: _env_float("JUDGE_WEIGHT", 0.5))

    # --- Server ---
    host: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("PORT", 10000))

    def validate_for_llm(self) -> None:
        if not self.llm_api_key:
            raise RuntimeError(
                "LLM_API_KEY is not set. Export it (or set it on Render) before "
                "running the pipeline."
            )


settings = Settings()
