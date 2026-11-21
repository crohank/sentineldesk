"""Qdrant vector DB (D3-22). Needs artifacts/qdrant/full (scripts/build_qdrant.py --scope full)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QDRANT = ROOT / "artifacts" / "qdrant" / "full"
DENSE = ROOT / "artifacts" / "dense" / "embeddinggemma" / "full"


@pytest.fixture(scope="module")
def store():
    if not (QDRANT.exists() and DENSE.exists()):
        pytest.skip("build the Qdrant collection first")
    from sentineldesk.vectordb import QdrantStore
    return QdrantStore.open(QDRANT)


def test_collection_holds_every_kb_ticket_and_no_holdout(store):
    ids = np.load(DENSE / "ids.npy", allow_pickle=True)
    assert store.count() == len(ids) == 22852
    hits = store.search(np.load(DENSE / "vectors.npy")[123], 5)
    assert all(t < "TKT-22857" for t, _, _ in hits)
    assert {"ticket_id", "subject", "answer", "answer_language"} <= set(hits[0][2])


def test_qdrant_matches_in_memory_search(store):
    ids = np.load(DENSE / "ids.npy", allow_pickle=True)
    V = np.load(DENSE / "vectors.npy")
    for i in (0, 4321, 20000):
        q = V[i] + 0.01 * np.sin(np.arange(V.shape[1]))  # a nearby query, not the stored point itself
        q = q / np.linalg.norm(q)
        s = V @ q
        expected = [str(ids[j]) for j in np.argsort(-s)[:5]]
        assert [t for t, _, _ in store.search(q, 5)] == expected


def test_retriever_uses_qdrant_and_falls_back(store):
    from sentineldesk.dense import ollama_available
    from sentineldesk.retrieval import Retriever
    if not ollama_available():
        pytest.skip("Ollama not running")
    r = Retriever.load(ROOT / "artifacts" / "retrieval" / "full", dense=DENSE, vectordb=QDRANT)
    ticket = {"subject": None, "body": "Our office printer only prints blank pages since yesterday."}
    ps, mode = r.retrieve_with_mode(ticket, 5)
    assert mode == "qdrant:embeddinggemma" and len(ps) == 5
    real = r.vdb
    class Broken:
        def search(self, *a, **k):
            raise RuntimeError("vector DB down")
    r.vdb = Broken()
    try:
        ps2, mode2 = r.retrieve_with_mode(ticket, 5)
        assert mode2 == "dense:embeddinggemma" and [p.ticket_id for p in ps2] == [p.ticket_id for p in ps]
    finally:
        r.vdb = real
