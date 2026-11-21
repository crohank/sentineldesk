"""Precedent retrieval over the historical KB (D3).

    retriever = Retriever.load("artifacts/retrieval/full",
                               dense="artifacts/dense/embeddinggemma/full")
    retriever.warm_up()   # once at startup: loads EmbeddingGemma into Ollama memory
    precedents, mode = retriever.retrieve_with_mode({"subject": None, "body": "..."}, k=5)
    # mode: "dense:embeddinggemma", "sparse" (fallback) or "none" (blank body)

Release mode (D3-12): EmbeddingGemma dense search through local Ollama. If Ollama is
unreachable or slow, the same call falls back to sparse TF-IDF and the returned mode says
so. Use retrieve_with_mode in a server: `last_mode` (set by retrieve) is shared state and
can be overwritten by a concurrent request.
Scores from the two modes are on different scales; thresholds must be chosen per mode.

Search runs on cleaned historical subject + body (D3-03, D3-04). Answers come from
the evidence store by ticket ID after retrieval.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2_normalize

from sentineldesk.textprep import clean_for_search

HOLDOUT_FIRST_ID = "TKT-22857"


@dataclass
class Precedent:
    ticket_id: str
    score: float
    subject: str
    answer_snippet: str
    answer_language: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _field(ticket: Any, name: str) -> str | None:
    if isinstance(ticket, dict):
        return ticket.get(name)
    return getattr(ticket, name, None)


class SparseEncoder:
    """Word 1-2 grams + char_wb 3-5 grams, each block L2-normalized, equal weight."""

    def __init__(self) -> None:
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True,
                                    max_features=300_000)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_df=0.5,
                                    sublinear_tf=True, max_features=300_000)

    def fit_transform(self, texts: list[str]) -> sparse.csr_matrix:
        w = l2_normalize(self.word.fit_transform(texts))
        c = l2_normalize(self.char.fit_transform(texts))
        return (sparse.hstack([w, c]).tocsr() / np.sqrt(2)).astype(np.float32)

    def transform(self, texts: list[str]) -> sparse.csr_matrix:
        w = l2_normalize(self.word.transform(texts))
        c = l2_normalize(self.char.transform(texts))
        return (sparse.hstack([w, c]).tocsr() / np.sqrt(2)).astype(np.float32)


class Retriever:
    def __init__(self, ids: np.ndarray, matrix: sparse.csr_matrix, encoder: SparseEncoder,
                 evidence: pd.DataFrame, manifest: dict) -> None:
        self.ids = ids
        self.matrix = matrix
        self.encoder = encoder
        self.evidence = evidence  # indexed by ticket_id
        self.manifest = manifest
        self.dense = None  # optional DenseIndex over the same IDs
        self.vdb = None    # optional QdrantStore holding the same vectors + payloads (D3-22)
        self.mode = "sparse"
        self.last_mode = "sparse"

    # ---------- build / persist ----------
    @classmethod
    def build(cls, docs: pd.DataFrame, evidence: pd.DataFrame, scope: str) -> "Retriever":
        """docs: ticket_id, subject, body (KB rows only). evidence: indexed by ticket_id."""
        ids = docs["ticket_id"].to_numpy()
        assert (ids < HOLDOUT_FIRST_ID).all(), "holdout ID in retrieval index"
        assert pd.Index(ids).is_unique
        texts = [clean_for_search(s, b) for s, b in zip(docs["subject"], docs["body"])]
        enc = SparseEncoder()
        matrix = enc.fit_transform(texts)
        manifest = {
            "mode": "sparse", "scope": scope, "n_docs": int(len(ids)),
            "first_id": str(ids.min()), "last_id": str(ids.max()),
            "word_vocab": len(enc.word.vocabulary_), "char_vocab": len(enc.char.vocabulary_),
        }
        return cls(ids, matrix, enc, evidence.loc[ids], manifest)

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.encoder, out / "encoder.joblib")
        sparse.save_npz(out / "matrix.npz", self.matrix)
        np.save(out / "ids.npy", self.ids)
        self.evidence.reset_index().to_parquet(out / "evidence.parquet", index=False)
        (out / "manifest.json").write_text(json.dumps(self.manifest, indent=2))

    @classmethod
    def load(cls, path: str | Path, dense: str | Path | None = None,
             vectordb: str | Path | None = None, vectordb_url: str | None = None) -> "Retriever":
        path = Path(path)
        ids = np.load(path / "ids.npy", allow_pickle=True)
        assert (ids < HOLDOUT_FIRST_ID).all(), "holdout ID in retrieval index"
        evidence = pd.read_parquet(path / "evidence.parquet").set_index("ticket_id")
        r = cls(ids, sparse.load_npz(path / "matrix.npz"), joblib.load(path / "encoder.joblib"),
                evidence, json.loads((path / "manifest.json").read_text()))
        if dense is not None and Path(dense).exists():
            from sentineldesk.dense import DenseIndex
            d = DenseIndex.load(dense)
            assert (d.ids == ids).all(), "dense and sparse indexes cover different IDs"
            r.dense = d
            r.mode = f"dense:{d.backend}"
            r.manifest = {**r.manifest, "dense": d.manifest}
            if vectordb_url or (vectordb is not None and Path(vectordb).exists()):
                from sentineldesk.vectordb import QdrantStore
                store = QdrantStore.open(vectordb, url=vectordb_url)
                assert store.count() == len(ids), "vector DB and index cover a different number of tickets"
                r.vdb = store
                r.mode = f"qdrant:{d.backend}"
        return r

    # ---------- query ----------
    def scores(self, query_texts: list[str]) -> np.ndarray:
        q = self.encoder.transform(query_texts)
        return (q @ self.matrix.T).toarray()

    def _hits(self, query_text: str, k: int = 5, dense_timeout: float = 2.0) -> tuple[list[tuple[str, float, dict | None]], str]:
        """Top-k (ticket_id, score, payload or None) and the mode that served them.
        Order: Qdrant (exact) -> in-memory NumPy dense -> sparse TF-IDF; never raises for a backend failure."""
        if self.dense is not None:
            try:
                from sentineldesk.dense import encode
                qvec = encode([query_text], "query", self.dense.backend, timeout=dense_timeout)[0]
            except Exception:  # Ollama down or slow
                qvec = None
            if qvec is not None:
                if self.vdb is not None:
                    try:
                        return self.vdb.search(qvec, k), f"qdrant:{self.dense.backend}"
                    except Exception:  # vector DB unavailable: same vector, in-memory search
                        pass
                s = self.dense.vectors @ qvec
                top = np.argpartition(-s, k)[:k]
                top = top[np.argsort(-s[top])]
                return [(str(self.ids[i]), float(s[i]), None) for i in top], f"dense:{self.dense.backend}"
        s = self.scores([query_text])[0]
        top = np.argpartition(-s, k)[:k]
        top = top[np.argsort(-s[top])]
        return [(str(self.ids[i]), float(s[i]), None) for i in top], "sparse"

    def search(self, query_text: str, k: int = 5,
               dense_timeout: float = 2.0) -> tuple[list[tuple[str, float]], str]:
        """Top-k (ticket_id, score) and the mode that served them."""
        hits, mode = self._hits(query_text, k, dense_timeout)
        return [(t, s) for t, s, _ in hits], mode

    def retrieve_with_mode(self, ticket: Any, k: int = 5) -> tuple[list[Precedent], str]:
        """ticket has `subject` (nullable) and `body`; dict or object. Expects masked text."""
        body = _field(ticket, "body")
        text = clean_for_search(_field(ticket, "subject"), body)
        if not text or not (body or "").strip():  # blank body = needs clarification (gate rule 1)
            return [], "none"
        hits, mode = self._hits(text, k)
        out = []
        for tid, score, payload in hits:
            # answer_snippet carries the exact historical answer (D3-14): answers are short
            # (p99 ~100 words), and drafts quote evidence spans that must match it verbatim.
            # With Qdrant the document comes from the vector DB payload, else from the evidence store.
            doc = payload if payload is not None else self.evidence.loc[tid]
            out.append(Precedent(tid, round(score, 4), str(doc["subject"]), str(doc["answer"]),
                                 str(doc["answer_language"])))
        return out, mode

    def retrieve(self, ticket: Any, k: int = 5) -> list[Precedent]:
        """Same as retrieve_with_mode, but returns only the precedents and stores the mode in last_mode."""
        out, self.last_mode = self.retrieve_with_mode(ticket, k)
        return out

    def warm_up(self) -> str:
        """Load the dense model into Ollama before the first request (cold query ~0.5 s, warm ~0.07 s).

        Returns the mode a query would use right now: "dense:<backend>" or "sparse".
        """
        _, mode = self.search("warm up", 1, dense_timeout=60.0)
        return mode

    def full_answer(self, ticket_id: str) -> str:
        """Full historical answer, for D4 validation and the console's expanded view."""
        return str(self.evidence.loc[ticket_id, "answer"])


def timed_retrieve(retriever: Retriever, ticket: Any, k: int = 5) -> tuple[list[Precedent], float]:
    t0 = time.perf_counter()
    res = retriever.retrieve(ticket, k)
    return res, (time.perf_counter() - t0) * 1000
