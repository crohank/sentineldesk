"""Dense retrieval with multilingual-e5-small (optional; needs sentence-transformers).

E5 expects "query: " / "passage: " prefixes and normalized vectors; cosine = dot product.
Exact in-memory search: 22,857 x 384 floats is ~35 MB, no ANN index needed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MODEL_ID = "intfloat/multilingual-e5-small"
_model = None


def get_model(device: str | None = None):
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        _model = SentenceTransformer(MODEL_ID, device=device)
        _model.max_seq_length = 256  # bodies: median 54 words, p99 104
    return _model


def encode(texts: list[str], kind: str, batch_size: int = 32) -> np.ndarray:
    assert kind in ("query", "passage")
    model = get_model()
    return model.encode([f"{kind}: {t}" for t in texts], batch_size=batch_size,
                        normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


class DenseIndex:
    def __init__(self, ids: np.ndarray, vectors: np.ndarray, manifest: dict) -> None:
        self.ids = ids
        self.vectors = vectors
        self.manifest = manifest

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "ids.npy", self.ids)
        np.save(out / "vectors.npy", self.vectors)
        (out / "manifest.json").write_text(json.dumps(self.manifest, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "DenseIndex":
        path = Path(path)
        ids = np.load(path / "ids.npy", allow_pickle=True)
        assert (ids < "TKT-22857").all(), "holdout ID in dense index"
        return cls(ids, np.load(path / "vectors.npy"), json.loads((path / "manifest.json").read_text()))

    def scores(self, query_texts: list[str]) -> np.ndarray:
        return encode(query_texts, "query") @ self.vectors.T


def hybrid_scores(sparse_s: np.ndarray, dense_s: np.ndarray, k: int = 60) -> np.ndarray:
    """Reciprocal rank fusion of two score matrices (rows = queries). Higher is better."""
    def rrf(s: np.ndarray) -> np.ndarray:
        ranks = np.empty_like(s, dtype=np.int64)
        order = np.argsort(-s, axis=1)
        rows = np.arange(s.shape[0])[:, None]
        ranks[rows, order] = np.arange(s.shape[1])[None, :]
        return 1.0 / (k + ranks + 1)
    return rrf(sparse_s) + rrf(dense_s)
