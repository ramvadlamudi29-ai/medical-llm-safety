"""Versioned prompt templates."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    system: str
    user: str

    def render(self, **kwargs: object) -> Dict[str, str]:
        return {
            "system": self.system.format(**kwargs),
            "user": self.user.format(**kwargs),
        }


def _join_contexts(contexts: List[str]) -> str:
    if not contexts:
        return "(no retrieved context)"
    return "\n\n".join(f"[{i+1}] {c.strip()}" for i, c in enumerate(contexts))


SEARCH_V1 = PromptTemplate(
    name="search",
    version="v1",
    system=(
        "You are a precise retrieval-augmented assistant. "
        "Answer ONLY using the provided context. "
        "If the answer is not in the context, say you don't know. "
        "Cite sources as [1], [2], ..."
    ),
    user=(
        "Question: {query}\n\n"
        "Context:\n{context}\n\n"
        "Write a concise, factual answer with citations."
    ),
)

SUMMARIZE_V1 = PromptTemplate(
    name="summarize",
    version="v1",
    system=(
        "You are a careful summarizer. Produce a short, faithful summary "
        "of the provided context. Do not invent facts."
    ),
    user=(
        "Summarize the following context for the user query: {query}\n\n"
        "Context:\n{context}\n\n"
        "Return 2-4 sentences."
    ),
)

JSON_EXTRACT_V1 = PromptTemplate(
    name="json_extract",
    version="v1",
    system=(
        "You extract structured information. Respond with ONLY a JSON object "
        "matching the requested schema. No prose, no code fences."
    ),
    user=(
        "Schema (JSON-like): {schema}\n"
        "User query: {query}\n"
        "Context:\n{context}\n"
    ),
)


REGISTRY: Dict[str, PromptTemplate] = {
    f"{p.name}@{p.version}": p for p in [SEARCH_V1, SUMMARIZE_V1, JSON_EXTRACT_V1]
}


def get_prompt(name: str, version: str = "v1") -> PromptTemplate:
    key = f"{name}@{version}"
    if key not in REGISTRY:
        raise KeyError(f"unknown prompt: {key}")
    return REGISTRY[key]


def render_prompt(
    name: str,
    query: str,
    contexts: List[str],
    *,
    version: str = "v1",
    schema: str = "",
) -> Dict[str, str]:
    tmpl = get_prompt(name, version)
    return tmpl.render(query=query, context=_join_contexts(contexts), schema=schema)
