"""Qdrant vector database for KB precedents (D3-22).

    store = QdrantStore.open("artifacts/qdrant/full")            # embedded, on disk, works offline
    store = QdrantStore.open(url="http://localhost:6333")        # same code against a Qdrant server / Cloud
    hits = store.search(query_vector, k=5)                       # [(ticket_id, cosine, payload), ...]

Collection `kb_tickets`: one point per KB ticket, the EmbeddingGemma vector (768-d, cosine) built by
scripts/build_dense.py, and a payload with ticket_id, subject, the exact historical answer and its
detected language. Point IDs are the ticket number (TKT-00123 -> 123). Search uses Qdrant's exact
mode, so results equal the brute-force cosine search that was evaluated (scripts/verify_qdrant.py).
Holdout IDs are rejected at build and at query time.
"""
from __future__ import annotations

import atexit
import json
import threading
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

COLLECTION = "kb_tickets"
HOLDOUT_FIRST_ID = "TKT-22857"
_CLIENTS: dict[str, Any] = {}       # embedded Qdrant allows one client per storage folder per process
_LOCK = threading.Lock()


@atexit.register
def _close_all() -> None:
    """Close embedded clients before interpreter teardown (closing during teardown prints tracebacks)."""
    for c in list(_CLIENTS.values()):
        try:
            c.close()
        except Exception:
            pass
    _CLIENTS.clear()


def point_id(ticket_id: str) -> int:
    return int(ticket_id.split("-")[1])


def _client(path: str | Path | None, url: str | None):
    from qdrant_client import QdrantClient
    key = url or str(Path(path).resolve())
    with _LOCK:
        if key not in _CLIENTS:
            with warnings.catch_warnings():
                # Embedded mode warns above 20,000 points; measured 30 ms per search at 22,852 (D3-22).
                warnings.filterwarnings("ignore", message="Local mode is not recommended")
                _CLIENTS[key] = QdrantClient(url=url, timeout=10) if url else QdrantClient(path=key)
        return _CLIENTS[key]


class QdrantStore:
    def __init__(self, client: Any, collection: str = COLLECTION) -> None:
        self.client = client
        self.collection = collection

    @classmethod
    def open(cls, path: str | Path | None = None, url: str | None = None, collection: str = COLLECTION) -> "QdrantStore":
        return cls(_client(path, url), collection)

    @classmethod
    def build(cls, path: str | Path, ids: np.ndarray, vectors: np.ndarray, payloads: list[dict],
              embedding_model: str, batch: int = 512) -> "QdrantStore":
        from qdrant_client import models
        assert len(ids) == len(vectors) == len(payloads)
        assert all(t < HOLDOUT_FIRST_ID for t in ids), "holdout ID in the vector DB"
        store = cls.open(path)
        c = store.client
        if c.collection_exists(store.collection):
            c.delete_collection(store.collection)
        c.create_collection(store.collection, vectors_config=models.VectorParams(
            size=int(vectors.shape[1]), distance=models.Distance.COSINE))
        for i in range(0, len(ids), batch):
            c.upsert(store.collection, points=[
                models.PointStruct(id=point_id(t), vector=vectors[j].tolist(), payload=payloads[j])
                for j, t in enumerate(ids[i:i + batch], start=i)])
        manifest = {"collection": store.collection, "points": store.count(), "vector_size": int(vectors.shape[1]),
                    "distance": "cosine", "embedding_model": embedding_model, "search": "exact",
                    "built": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        (Path(path) / "sentineldesk_manifest.json").write_text(json.dumps(manifest, indent=2))
        return store

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    def search(self, vector: np.ndarray, k: int = 5, exclude: set[str] | None = None) -> list[tuple[str, float, dict]]:
        from qdrant_client import models
        with warnings.catch_warnings():
            # exact=True matters for a Qdrant server (HNSW otherwise); embedded mode is always exact.
            warnings.filterwarnings("ignore", message="Local mode performs exact")
            res = self._query(vector, k + (len(exclude) if exclude else 0), models)
        out = []
        for p in res:
            tid = p.payload["ticket_id"]
            if exclude and tid in exclude:
                continue
            assert tid < HOLDOUT_FIRST_ID, f"holdout ID {tid} returned by the vector DB"
            out.append((tid, float(p.score), p.payload))
        return out[:k]

    def _query(self, vector: np.ndarray, limit: int, models: Any) -> list:
        return self.client.query_points(
            self.collection, query=np.asarray(vector, dtype=np.float32).tolist(),
            limit=limit, with_payload=True, search_params=models.SearchParams(exact=True)).points
