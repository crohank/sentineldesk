"""D3 stack in one object: language detection, retrieval, drafting, validation (for D5's pipeline).

    from sentineldesk.d3 import D3
    d3 = D3.load()                 # once at startup; warms EmbeddingGemma and Gemma 4
    lang = d3.detect_language(masked)                    # "en" | "de" | other ISO | "und"
    precedents = d3.retrieve(masked, 5)                  # list[Precedent] (D5 fields)
    draft = d3.draft(masked, lang, precedents)           # Draft (D5 fields) or an abstaining Draft
    d3.validate(draft, lang, precedents)                 # {"valid": bool, "errors": [...]}
    d3.last_retrieval_mode, d3.last_generator_mode       # what actually served the last call

Retrieval: Vertex AI RAG Engine when artifacts/rag_engine/full.json exists and GOOGLE_CLOUD_PROJECT
is set; else EmbeddingGemma query vectors (local Ollama) searched in the Qdrant vector DB
(artifacts/qdrant/full, or SD_QDRANT_URL), then in memory; else sparse TF-IDF. Drafting: Gemini 3.5 Flash on Vertex,
then Gemma 4 (local Ollama), then extractive quotes. Every fallback is automatic and per request.
The `last_*_mode` attributes are per instance; use `analyze_ticket` when calls may run concurrently.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sentineldesk.generation import Draft, Generator, validate as _validate
from sentineldesk.language import detect_language as _detect
from sentineldesk.retrieval import Precedent, Retriever

ROOT = Path(__file__).resolve().parents[1]

# Relevance guard per retrieval mode (D3-19): below this top score, drafting abstains.
MIN_TOP_SCORE = {"qdrant:embeddinggemma": 0.35, "dense:embeddinggemma": 0.35, "sparse": 0.12, "rag_engine": float(os.environ.get("SD_RAG_MIN_SCORE", "0.55"))}


def _text(t: Any) -> tuple[str | None, str]:
    get = (lambda k: t.get(k)) if isinstance(t, dict) else (lambda k: getattr(t, k, None))
    return get("subject"), get("body") or ""


@dataclass
class D3:
    local: Retriever
    rag: Any = None                     # RagEngineRetriever or None
    generator: Generator = field(default_factory=Generator.default)
    last_retrieval_mode: str = "none"
    last_generator_mode: str = "none"

    @classmethod
    def load(cls, root: Path = ROOT, use_rag_engine: bool | None = None) -> "D3":
        local = Retriever.load(root / "artifacts" / "retrieval" / "full",
                               dense=root / "artifacts" / "dense" / "embeddinggemma" / "full",
                               vectordb=root / "artifacts" / "qdrant" / "full",
                               vectordb_url=os.environ.get("SD_QDRANT_URL"))
        local.warm_up()
        rag = None
        manifest = root / "artifacts" / "rag_engine" / "full.json"
        if use_rag_engine is None:
            use_rag_engine = manifest.exists() and bool(os.environ.get("GOOGLE_CLOUD_PROJECT"))
        if use_rag_engine:
            from sentineldesk.rag_engine import RagEngineRetriever
            rag = RagEngineRetriever.from_manifest(manifest, local=local)
        return cls(local=local, rag=rag)

    def detect_language(self, ticket: Any) -> str:
        subject, body = _text(ticket)
        return _detect(f"{subject or ''}\n{body}")[0]

    def retrieve(self, ticket: Any, k: int = 5) -> list[Precedent]:
        precedents, self.last_retrieval_mode = (self.rag or self.local).retrieve_with_mode(ticket, k)
        return precedents

    def draft(self, ticket: Any, language: str, precedents: list[Any]) -> Draft:
        guard = MIN_TOP_SCORE.get(self.last_retrieval_mode)
        res = self.generator.draft_with_mode(ticket, language, precedents, min_top_score=guard)
        self.last_generator_mode = res.mode
        return res.draft

    def validate(self, draft: Draft, language: str, precedents: list[Any]) -> dict:
        return _validate(draft, language, precedents)

    def analyze_ticket(self, ticket: Any, k: int = 5) -> dict:
        """Language, precedents, draft and the modes that served them, without shared state."""
        lang = self.detect_language(ticket)
        precedents, rmode = (self.rag or self.local).retrieve_with_mode(ticket, k)
        res = self.generator.draft_with_mode(ticket, lang, precedents, min_top_score=MIN_TOP_SCORE.get(rmode)) \
            if lang in ("en", "de") else None
        return {"language": lang, "precedents": precedents, "retrieval_mode": rmode,
                "draft": res.draft if res else None, "generator_mode": res.mode if res else "none",
                "generation_attempts": res.attempts if res else []}
