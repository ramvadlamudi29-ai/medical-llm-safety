from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List

# Optionally hydrate from a local .env without forcing the dependency.
try:  # pragma: no cover - optional
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(override=False)
except Exception:  # noqa: BLE001
    pass


def _csv_env(name: str, default: str = "") -> List[str]:
    raw = os.getenv(name, default)
    return [x.strip() for x in raw.split(",") if x.strip()]


def _bool_env(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # ---- API / transport ----
    api_keys: List[str] = field(
        default_factory=lambda: _csv_env("APP_API_KEYS", "test-key")
    )
    max_input_chars: int = int(os.getenv("MAX_INPUT_CHARS", "4000"))
    max_top_k: int = int(os.getenv("MAX_TOP_K", "10"))
    max_body_bytes: int = int(os.getenv("MAX_BODY_BYTES", str(64 * 1024)))
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))

    # ---- LLM ----
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1")
    llm_timeout_s: float = float(os.getenv("LLM_TIMEOUT_S", "20"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
    pipeline_timeout_s: float = float(os.getenv("PIPELINE_TIMEOUT_S", "30"))

    # ---- Cost ----
    enable_budget_tracking: bool = _bool_env("ENABLE_BUDGET_TRACKING", "false")
    cost_per_1k_input: float = float(os.getenv("COST_PER_1K_INPUT", "0.00015"))
    cost_per_1k_output: float = float(os.getenv("COST_PER_1K_OUTPUT", "0.0006"))

    # ---- Versioning ----
    model_version: str = os.getenv("MODEL_VERSION", "v1.0.0")
    prompt_version: str = os.getenv("PROMPT_VERSION", "v1")
    dataset_version: str = os.getenv("DATASET_VERSION", "v1")

    # ---- Reranker ----
    reranker_model: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    enable_reranker: bool = _bool_env("ENABLE_RERANKER", "false")

    # ---- Cache ----
    response_cache_size: int = int(os.getenv("RESPONSE_CACHE_SIZE", "256"))
    embedding_cache_size: int = int(os.getenv("EMBEDDING_CACHE_SIZE", "1024"))

    # ---- Feature flags ----
    enable_cache: bool = _bool_env("ENABLE_CACHE", "true")
    enable_eval: bool = _bool_env("ENABLE_EVAL", "true")
    enable_security_guard: bool = _bool_env("ENABLE_SECURITY_GUARD", "true")
    debug_mode: bool = _bool_env("DEBUG_MODE", "false")

    # ---- Misc ----
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    experiments_dir: str = os.getenv("EXPERIMENTS_DIR", "experiments")


settings = Settings()
