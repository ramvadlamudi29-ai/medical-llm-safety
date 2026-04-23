"""Data ingestion pipeline: load -> clean -> chunk -> index.

Supports plain-text files (one doc per file) and JSONL where each line is
``{"id": "...", "title": "...", "text": "..."}``.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Iterable, List

from core.rag import Document, HybridRAG


_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    return _WS.sub(" ", (text or "")).strip()


def chunk_text(text: str, max_chars: int = 800, overlap: int = 80) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    step = max(1, max_chars - overlap)
    while start < len(text):
        chunks.append(text[start : start + max_chars])
        start += step
    return chunks


def load_text_file(path: Path) -> List[Document]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_text(raw)
    return [
        Document(id=f"{path.stem}-{i}", title=path.stem, text=c)
        for i, c in enumerate(chunks)
    ]


def load_jsonl(path: Path) -> List[Document]:
    docs: List[Document] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = clean_text(str(obj.get("text", "")))
        if not text:
            continue
        title = str(obj.get("title") or obj.get("id") or f"doc-{i}")
        for j, chunk in enumerate(chunk_text(text)):
            docs.append(
                Document(
                    id=f"{obj.get('id', path.stem)}-{i}-{j}",
                    title=title,
                    text=chunk,
                )
            )
    return docs


def load_path(path: str | Path) -> List[Document]:
    p = Path(path)
    if not p.exists():
        return []
    if p.is_file():
        if p.suffix.lower() == ".jsonl":
            return load_jsonl(p)
        return load_text_file(p)
    docs: List[Document] = []
    for child in sorted(p.rglob("*")):
        if child.is_file() and child.suffix.lower() in {".txt", ".md", ".jsonl"}:
            docs.extend(load_path(child))
    return docs


def build_rag(paths: Iterable[str | Path] | None = None) -> HybridRAG:
    """Build a HybridRAG from the given paths, falling back to defaults."""
    docs: List[Document] = []
    for p in paths or ():
        docs.extend(load_path(p))
    if not docs:
        return HybridRAG()
    return HybridRAG(documents=docs)
