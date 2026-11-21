"""Precedent retrieval over the historical KB (D3).

    retriever = Retriever.load("artifacts/retrieval/full")
    precedents = retriever.retrieve({"subject": None, "body": "..."}, k=5)

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

from sentineldesk.textprep import answer_snippet, clean_for_search

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
        self.mode = "sparse"

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
    def load(cls, path: str | Path) -> "Retriever":
        path = Path(path)
        ids = np.load(path / "ids.npy", allow_pickle=True)
        assert (ids < HOLDOUT_FIRST_ID).all(), "holdout ID in retrieval index"
        evidence = pd.read_parquet(path / "evidence.parquet").set_index("ticket_id")
        return cls(ids, sparse.load_npz(path / "matrix.npz"), joblib.load(path / "encoder.joblib"),
                   evidence, json.loads((path / "manifest.json").read_text()))

    # ---------- query ----------
    def scores(self, query_texts: list[str]) -> np.ndarray:
        q = self.encoder.transform(query_texts)
        return (q @ self.matrix.T).toarray()

    def search(self, query_text: str, k: int = 5) -> list[tuple[str, float]]:
        s = self.scores([query_text])[0]
        top = np.argpartition(-s, k)[:k]
        top = top[np.argsort(-s[top])]
        return [(str(self.ids[i]), float(s[i])) for i in top]

    def retrieve(self, ticket: Any, k: int = 5) -> list[Precedent]:
        """ticket has `subject` (nullable) and `body`; dict or object. Expects masked text."""
        text = clean_for_search(_field(ticket, "subject"), _field(ticket, "body"))
        if not text:
            return []
        out = []
        for tid, score in self.search(text, k):
            ev = self.evidence.loc[tid]
            out.append(Precedent(tid, round(score, 4), ev["subject"], answer_snippet(ev["answer"]),
                                 ev["answer_language"]))
        return out

    def full_answer(self, ticket_id: str) -> str:
        """Full historical answer, for D4 validation and the console's expanded view."""
        return str(self.evidence.loc[ticket_id, "answer"])


def timed_retrieve(retriever: Retriever, ticket: Any, k: int = 5) -> tuple[list[Precedent], float]:
    t0 = time.perf_counter()
    res = retriever.retrieve(ticket, k)
    return res, (time.perf_counter() - t0) * 1000
