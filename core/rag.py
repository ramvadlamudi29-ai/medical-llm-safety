"""Hybrid RAG: BM25 + TF-IDF cosine, with optional cross-encoder reranking."""
from __future__ import annotations
import math
import re
from dataclasses import dataclass
from typing import List, Optional

from core.config import settings
from core.logging_setup import get_logger

log = get_logger("rag")

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


@dataclass
class Document:
    id: str
    title: str
    text: str


@dataclass
class RetrievalResult:
    document: Document
    score: float


DEFAULT_DOCS: List[Document] = [
    Document(
        id="sec-001",
        title="Zero Trust Security Policy",
        text=(
            "Zero trust security policy requires continuous verification of every user and device. "
            "All access requests are authenticated, authorized, and encrypted regardless of network location. "
            "The zero trust model assumes no implicit trust based on network perimeter."
        ),
    ),
    Document(
        id="ops-002",
        title="Incident Response Guide",
        text=(
            "The incident response guide describes how to detect, contain, eradicate, and recover from security incidents. "
            "On-call engineers must triage alerts within fifteen minutes and escalate critical issues to the security team."
        ),
    ),
    Document(
        id="hr-003",
        title="Remote Work Policy",
        text=(
            "Employees may work remotely up to four days a week with manager approval. "
            "All remote workers must use company-managed devices and the corporate VPN."
        ),
    ),
    Document(
        id="eng-004",
        title="Code Review Standards",
        text=(
            "Code reviews must be completed within one business day. "
            "Every pull request requires at least one approving review and passing CI checks before merge."
        ),
    ),
]


class CrossEncoderReranker:
    """Cross-encoder reranker with graceful fallback to a lexical heuristic.

    If ``sentence-transformers`` is installed and the model can be loaded,
    real cross-encoder scores are used. Otherwise we fall back to a fast
    token-overlap score so the rest of the pipeline keeps working offline.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.reranker_model
        self._model = None
        self.available = False
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            self._model = CrossEncoder(self.model_name)
            self.available = True
        except Exception as e:  # noqa: BLE001
            log.info(
                "cross-encoder unavailable, using lexical fallback",
                extra={"error": str(e)},
            )

    def rerank(
        self, query: str, results: List[RetrievalResult], top_k: int | None = None
    ) -> List[RetrievalResult]:
        if not results:
            return results
        if self.available and self._model is not None:
            pairs = [(query, r.document.text) for r in results]
            try:
                scores = self._model.predict(pairs)  # type: ignore[attr-defined]
                ranked = sorted(
                    (
                        RetrievalResult(document=r.document, score=float(s))
                        for r, s in zip(results, scores)
                    ),
                    key=lambda r: r.score,
                    reverse=True,
                )
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "reranker failed, using lexical fallback",
                    extra={"error": str(e)},
                )
                ranked = self._lexical_rerank(query, results)
        else:
            ranked = self._lexical_rerank(query, results)
        return ranked[: top_k or len(ranked)]

    @staticmethod
    def _lexical_rerank(
        query: str, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        q_tokens = set(tokenize(query))
        boosted: List[RetrievalResult] = []
        for r in results:
            d_tokens = set(tokenize(r.document.text))
            overlap = len(q_tokens & d_tokens) / (len(q_tokens) or 1)
            boosted.append(
                RetrievalResult(document=r.document, score=r.score + overlap)
            )
        boosted.sort(key=lambda r: r.score, reverse=True)
        return boosted


class HybridRAG:
    """Tiny offline hybrid retriever combining BM25 and TF-IDF cosine."""

    def __init__(
        self,
        documents: List[Document] | None = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> None:
        self.documents: List[Document] = (
            list(documents) if documents else list(DEFAULT_DOCS)
        )
        self.reranker = reranker
        self._build_index()

    # ---- index ----
    def _build_index(self) -> None:
        self._doc_tokens = [tokenize(d.title + " " + d.text) for d in self.documents]
        self._doc_lens = [len(t) or 1 for t in self._doc_tokens]
        self._avgdl = sum(self._doc_lens) / max(1, len(self._doc_lens))
        self._df: dict[str, int] = {}
        for toks in self._doc_tokens:
            for t in set(toks):
                self._df[t] = self._df.get(t, 0) + 1
        self._N = max(1, len(self.documents))

    # ---- public mutation ----
    def add_documents(self, docs: List[Document]) -> None:
        self.documents.extend(docs)
        self._build_index()

    # ---- scoring ----
    def _bm25(
        self, query_tokens: List[str], idx: int, k1: float = 1.5, b: float = 0.75
    ) -> float:
        score = 0.0
        toks = self._doc_tokens[idx]
        dl = self._doc_lens[idx]
        tf_counts: dict[str, int] = {}
        for t in toks:
            tf_counts[t] = tf_counts.get(t, 0) + 1
        for q in query_tokens:
            if q not in tf_counts:
                continue
            df = self._df.get(q, 0)
            idf = math.log(1 + (self._N - df + 0.5) / (df + 0.5))
            tf = tf_counts[q]
            denom = tf + k1 * (1 - b + b * dl / self._avgdl)
            score += idf * (tf * (k1 + 1)) / denom
        return score

    def _tfidf_cosine(self, query_tokens: List[str], idx: int) -> float:
        toks = self._doc_tokens[idx]
        if not toks or not query_tokens:
            return 0.0
        q_set = set(query_tokens)

        def w(term: str, count: int) -> float:
            df = self._df.get(term, 0)
            idf = math.log((self._N + 1) / (df + 1)) + 1
            return count * idf

        d_counts: dict[str, int] = {}
        for t in toks:
            d_counts[t] = d_counts.get(t, 0) + 1
        q_counts: dict[str, int] = {}
        for t in query_tokens:
            q_counts[t] = q_counts.get(t, 0) + 1

        d_w_all = {t: w(t, c) for t, c in d_counts.items()}
        q_w_all = {t: w(t, c) for t, c in q_counts.items()}
        dot = sum(q_w_all[t] * d_w_all.get(t, 0.0) for t in q_set)
        d_norm = math.sqrt(sum(v * v for v in d_w_all.values())) or 1.0
        q_norm = math.sqrt(sum(v * v for v in q_w_all.values())) or 1.0
        return dot / (d_norm * q_norm)

    def retrieve(
        self, query: str, top_k: int = 3, rerank: bool = False
    ) -> List[RetrievalResult]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scored: List[RetrievalResult] = []
        for i, doc in enumerate(self.documents):
            s = (
                0.7 * self._bm25(q_tokens, i)
                + 0.3 * self._tfidf_cosine(q_tokens, i) * 10
            )
            if s > 0:
                scored.append(RetrievalResult(document=doc, score=float(s)))
        scored.sort(key=lambda r: r.score, reverse=True)

        if rerank and self.reranker is not None:
            # rerank a slightly larger window then trim
            window = scored[: max(top_k * 3, top_k)]
            scored = self.reranker.rerank(query, window, top_k=top_k)

        return scored[: max(0, int(top_k))]


# Backward-compat alias
RAGPipeline = HybridRAG
