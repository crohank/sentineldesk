"""EmbeddingGemma wrapper: load once, encode with a named prompt, L2-normalize, cache on disk.

Cache layout: artifacts/embeddings/<key>/{vectors.npy, ids.parquet, meta.json}
key = sha256(model_id, revision, prompt text, max_len, dim, sha256(texts in order)).
Because the prompt is part of the key, retrieval (D3) can reuse this module with a retrieval
prompt (e.g. prompt_name="Retrieval-document" / "Retrieval-query") without colliding with the
classification vectors used by triage.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts" / "embeddings"
MODEL_ID = "google/embeddinggemma-300m"


def texts_sha(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


def resolve_revision(model_id: str, revision: str | None) -> str:
    if revision:
        return revision
    from huggingface_hub import HfApi
    return HfApi().model_info(model_id).sha


class Embedder:
    def __init__(self, model_id: str = MODEL_ID, revision: str | None = None, prompt_name: str = "Classification",
                 max_len: int = 512, dim: int | None = None, device: str | None = None, batch_size: int = 32):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.revision = resolve_revision(model_id, revision)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # float32: the model card says EmbeddingGemma does not support float16.
        self.model = SentenceTransformer(model_id, revision=self.revision, device=self.device,
                                         truncate_dim=dim, model_kwargs={"torch_dtype": torch.float32})
        self.model.max_seq_length = max_len
        if prompt_name not in self.model.prompts:
            raise KeyError(f"prompt {prompt_name!r} not in model prompts {sorted(self.model.prompts)}")
        self.prompt_name = prompt_name
        self.prompt = self.model.prompts[prompt_name]
        self.max_len = max_len
        self.dim = dim or self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size

    def config(self) -> dict:
        return {"model_id": self.model_id, "revision": self.revision, "prompt_name": self.prompt_name,
                "prompt": self.prompt, "max_len": self.max_len, "dim": self.dim}

    def encode(self, texts: list[str], batch_size: int | None = None, show_progress: bool = False) -> np.ndarray:
        v = self.model.encode(list(texts), prompt_name=self.prompt_name, batch_size=batch_size or self.batch_size,
                              convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=show_progress)
        v = v.astype(np.float32)
        # normalize after any Matryoshka truncation
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    def cache_key(self, texts: list[str]) -> str:
        cfg = {**self.config(), "texts_sha256": texts_sha(texts)}
        return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:24]

    def cached_encode(self, ids: list[str], texts: list[str], show_progress: bool = True) -> tuple[np.ndarray, dict]:
        """Vectors aligned with `ids`. Returns (vectors, meta); meta['cache_hit'] tells whether we encoded."""
        assert len(ids) == len(texts)
        key = self.cache_key(texts)
        d = CACHE / key
        if (d / "meta.json").exists():
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            stored = pd.read_parquet(d / "ids.parquet")["id"].tolist()
            if stored != list(ids):
                raise ValueError(f"cache {key}: ids differ from request")
            return np.load(d / "vectors.npy"), {**meta, "cache_hit": True}
        t0 = time.perf_counter()
        vec = self.encode(texts, show_progress=show_progress)
        secs = time.perf_counter() - t0
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "vectors.tmp.npy"
        np.save(tmp, vec)
        tmp.replace(d / "vectors.npy")
        pd.DataFrame({"id": list(ids)}).to_parquet(d / "ids.parquet", index=False)
        meta = {**self.config(), "key": key, "n": len(ids), "texts_sha256": texts_sha(texts),
                "device": self.device, "encode_seconds": secs}
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return vec, {**meta, "cache_hit": False}
