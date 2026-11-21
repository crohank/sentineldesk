"""D3 entry point and RAG Engine merge logic (no network)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentineldesk.generation import Generator  # noqa: E402
from sentineldesk.rag_engine import RagEngineRetriever  # noqa: E402
from sentineldesk.retrieval import Retriever  # noqa: E402

FULL = ROOT / "artifacts" / "retrieval" / "full"


@pytest.fixture(scope="module")
def local():
    if not FULL.exists():
        pytest.skip("run scripts/build_index.py --scope full first")
    return Retriever.load(FULL)  # sparse only: no Ollama needed


def test_d3_analyze_blank_and_offtopic(local):
    from sentineldesk.d3 import D3
    d3 = D3(local=local, generator=Generator([]))
    blank = d3.analyze_ticket({"subject": "Help", "body": "  "})
    assert blank["precedents"] == [] and blank["retrieval_mode"] == "none"
    out = d3.analyze_ticket({"subject": None, "body": "Our office printer only prints blank pages since yesterday."})
    assert out["language"] == "en" and len(out["precedents"]) == 5 and out["retrieval_mode"] == "sparse"
    assert out["draft"] is not None and out["generator_mode"] in ("offline_templates", "none")


def test_rag_engine_merges_shards_and_rejects_holdout(local):
    rag = RagEngineRetriever("p", "europe-west3", ["c0", "c1"], local)
    fake = {"c0": [("TKT-00001", 0.80), ("TKT-00002", 0.60)], "c1": [("TKT-00003", 0.70), ("TKT-00001", 0.80)]}
    rag._query = lambda corpus, text, top_k: fake[corpus]
    assert rag.search("x", 3) == [("TKT-00001", 0.80), ("TKT-00003", 0.70), ("TKT-00002", 0.60)]
    assert rag.search("x", 2, exclude={"TKT-00001"}) == [("TKT-00003", 0.70), ("TKT-00002", 0.60)]
    rag._query = lambda corpus, text, top_k: [("TKT-23000", 0.9)]
    with pytest.raises(AssertionError):
        rag.search("x", 1)


def test_rag_engine_failure_falls_back_to_local(local):
    rag = RagEngineRetriever("p", "europe-west3", ["c0"], local)
    def boom(*a, **k):
        raise RuntimeError("quota")
    rag._query = boom
    precedents, mode = rag.retrieve_with_mode({"subject": None, "body": "VPN keeps disconnecting every few minutes."}, 5)
    assert mode == "sparse" and len(precedents) == 5
