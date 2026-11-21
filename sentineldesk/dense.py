"""Dense retrieval backends (D3-10, D3-12).

Backends:
  embeddinggemma  Google EmbeddingGemma (Gemma family, open weights), served locally by Ollama.
                  Needs `ollama serve` and `ollama pull embeddinggemma`. No Python deps.
  e5              intfloat/multilingual-e5-small via sentence-transformers (requirements-dense.txt).

Vectors are L2-normalized, so cosine = dot product. Exact in-memory search:
22,857 x 768 floats is ~70 MB, no ANN index needed.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import numpy as np

BACKENDS = {
    "embeddinggemma": {"model": "embeddinggemma", "source": "ollama"},
    "e5": {"model": "intfloat/multilingual-e5-small", "source": "sentence-transformers"},
}
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Prompt formats from each model card.
_PREFIX = {
    "embeddinggemma": {"query": "task: search result | query: ", "passage": "title: none | text: "},
    "e5": {"query": "query: ", "passage": "passage: "},
}

_st_model = None


def _e5_model():
    global _st_model
    if _st_model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _st_model = SentenceTransformer(BACKENDS["e5"]["model"], device=device)
        _st_model.max_seq_length = 256
    return _st_model


def _ollama_embed(texts: list[str], timeout: float) -> np.ndarray:
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embed",
        # keep_alive=-1: keep the model loaded, otherwise Ollama unloads it after 5 idle minutes
        # and the next query pays a ~0.5 s reload.
        data=json.dumps({"model": BACKENDS["embeddinggemma"]["model"], "input": texts,
                         "keep_alive": -1}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return np.asarray(json.loads(resp.read())["embeddings"], dtype=np.float32)


def ollama_available(timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/version", timeout=timeout):
            return True
    except OSError:
        return False


def encode(texts: list[str], kind: str, backend: str = "embeddinggemma", batch_size: int = 64,
           timeout: float = 30.0) -> np.ndarray:
    assert kind in ("query", "passage") and backend in BACKENDS
    prefixed = [_PREFIX[backend][kind] + t for t in texts]
    if backend == "e5":
        return _e5_model().encode(prefixed, batch_size=32, normalize_embeddings=True,
                                  convert_to_numpy=True).astype(np.float32)
    vecs = np.vstack([_ollama_embed(prefixed[i:i + batch_size], timeout)
                      for i in range(0, len(prefixed), batch_size)])
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


class DenseIndex:
    def __init__(self, ids: np.ndarray, vectors: np.ndarray, manifest: dict) -> None:
        self.ids = ids
        self.vectors = vectors
        self.manifest = manifest
        self.backend = manifest.get("backend", "e5")

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

    def scores(self, query_texts: list[str], timeout: float = 30.0) -> np.ndarray:
        return encode(query_texts, "query", self.backend, timeout=timeout) @ self.vectors.T


def hybrid_scores(sparse_s: np.ndarray, dense_s: np.ndarray, k: int = 60) -> np.ndarray:
    """Reciprocal rank fusion of two score matrices (rows = queries). Higher is better."""
    def rrf(s: np.ndarray) -> np.ndarray:
        ranks = np.empty_like(s, dtype=np.int64)
        order = np.argsort(-s, axis=1)
        rows = np.arange(s.shape[0])[:, None]
        ranks[rows, order] = np.arange(s.shape[1])[None, :]
        return 1.0 / (k + ranks + 1)
    return rrf(sparse_s) + rrf(dense_s)
