"""LLM abstraction with OpenAI / Ollama / local heuristic fallback.

All providers expose ``generate(query, contexts, intent="search") -> LLMResponse``.
Errors trigger graceful fallback to the local heuristic so the system never crashes.
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.config import settings
from core.logging_setup import get_logger
from core.monitor import metrics
from core.prompts import render_prompt

log = get_logger("llm")


# -----------------------------
# Response model
# -----------------------------
@dataclass
class LLMResponse:
    text: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Cost / token estimation
# -----------------------------
def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token (good enough for budgeting)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        (prompt_tokens / 1000.0) * settings.cost_per_1k_input
        + (completion_tokens / 1000.0) * settings.cost_per_1k_output
    )


# -----------------------------
# Local heuristic (offline-safe)
# -----------------------------
class LocalHeuristicLLM:
    provider = "local_heuristic"
    model = "heuristic"

    def generate(
        self, query: str, contexts: List[str], intent: str = "search"
    ) -> LLMResponse:
        if not contexts:
            return LLMResponse(
                text=f"(no context) {query}",
                provider=self.provider,
                model=self.model,
            )
        if intent == "summarize":
            joined = " ".join(contexts)
            sentences = [
                s.strip() for s in joined.replace("\n", " ").split(".") if s.strip()
            ]
            summary = ". ".join(sentences[:2])
            text = f"Summary: {summary}."
        else:
            text = contexts[0].strip().split("\n")[0]
        return LLMResponse(text=text, provider=self.provider, model=self.model)


# -----------------------------
# Retry helper
# -----------------------------
def _with_retry(fn, retries: int, base_delay: float = 0.25):
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if attempt >= retries:
                break
            time.sleep(base_delay * (2**attempt))
    assert last_exc is not None
    raise last_exc


# -----------------------------
# OpenAI provider (HTTP via httpx, no SDK required)
# -----------------------------
class OpenAILLM:
    provider = "openai"

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model or settings.openai_model
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")
        self.timeout = timeout or settings.llm_timeout_s

    def _call(self, system: str, user: str) -> Dict[str, Any]:
        import httpx  # local import to avoid hard dep at import time

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            return r.json()

    def generate(
        self, query: str, contexts: List[str], intent: str = "search"
    ) -> LLMResponse:
        prompt_name = "summarize" if intent == "summarize" else "search"
        rendered = render_prompt(prompt_name, query, contexts)
        system, user = rendered["system"], rendered["user"]

        def _go() -> Dict[str, Any]:
            return self._call(system, user)

        data = _with_retry(_go, retries=settings.llm_max_retries)
        text = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {}) or {}
        ptok = int(usage.get("prompt_tokens") or estimate_tokens(system + user))
        ctok = int(usage.get("completion_tokens") or estimate_tokens(text))
        cost = estimate_cost(ptok, ctok)
        return LLMResponse(
            text=text,
            provider=self.provider,
            prompt_tokens=ptok,
            completion_tokens=ctok,
            cost_usd=cost,
            model=self.model,
            extra={"raw_usage": usage},
        )


# -----------------------------
# Ollama provider (local LLM)
# -----------------------------
class OllamaLLM:
    provider = "ollama"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout = timeout or settings.llm_timeout_s

    def generate(
        self, query: str, contexts: List[str], intent: str = "search"
    ) -> LLMResponse:
        import httpx

        prompt_name = "summarize" if intent == "summarize" else "search"
        rendered = render_prompt(prompt_name, query, contexts)
        full_prompt = f"{rendered['system']}\n\n{rendered['user']}"

        def _go() -> Dict[str, Any]:
            url = f"{self.base_url}/api/generate"
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(
                    url,
                    json={"model": self.model, "prompt": full_prompt, "stream": False},
                )
                r.raise_for_status()
                return r.json()

        data = _with_retry(_go, retries=settings.llm_max_retries)
        text = (data.get("response") or "").strip()
        ptok = estimate_tokens(full_prompt)
        ctok = estimate_tokens(text)
        return LLMResponse(
            text=text,
            provider=self.provider,
            prompt_tokens=ptok,
            completion_tokens=ctok,
            cost_usd=0.0,  # local
            model=self.model,
        )


# -----------------------------
# Wrapper with automatic fallback
# -----------------------------
class LLMClient:
    """Tries OpenAI -> Ollama -> local heuristic. Always returns an answer."""

    def __init__(
        self,
        openai: OpenAILLM | None = None,
        ollama: OllamaLLM | None = None,
        local: LocalHeuristicLLM | None = None,
    ) -> None:
        self.openai = openai
        self.ollama = ollama
        self.local = local or LocalHeuristicLLM()

    @classmethod
    def from_settings(cls) -> "LLMClient":
        openai = (
            OpenAILLM(api_key=settings.openai_api_key)
            if settings.openai_api_key
            else None
        )
        ollama = OllamaLLM() if settings.ollama_base_url else None
        return cls(openai=openai, ollama=ollama)

    def generate(
        self,
        query: str,
        contexts: List[str],
        intent: str = "search",
        offline_safe: bool = False,
    ) -> LLMResponse:
        if offline_safe:
            return self.local.generate(query, contexts, intent=intent)

        for provider in (self.openai, self.ollama):
            if provider is None:
                continue
            try:
                resp = provider.generate(query, contexts, intent=intent)
                metrics.incr(f"llm.{provider.provider}.ok")
                metrics.add_tokens(
                    f"llm.{provider.provider}.tokens",
                    resp.prompt_tokens + resp.completion_tokens,
                )
                if settings.enable_budget_tracking:
                    metrics.add_cost(f"llm.{provider.provider}.cost", resp.cost_usd)
                return resp
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "llm provider failed, falling back",
                    extra={"provider": provider.provider, "error": str(e)},
                )
                metrics.incr(f"llm.{provider.provider}.error")
                continue

        metrics.incr("llm.local.fallback")
        return self.local.generate(query, contexts, intent=intent)
