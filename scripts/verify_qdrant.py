"""Prove the Qdrant vector DB returns what the evaluated in-memory search returned (D3-22).

    python scripts/verify_qdrant.py [--n 500]

Embeds N validation queries once (EmbeddingGemma via Ollama), then compares the top 5 from the
fit-scope Qdrant collection with the brute-force NumPy cosine search over the same vectors, and
checks that payload answers equal the evidence store. Writes reports/qdrant_parity.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentineldesk.dense import DenseIndex, encode  # noqa: E402
from sentineldesk.textprep import clean_for_search  # noqa: E402
from sentineldesk.vectordb import QdrantStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()
    kb = pd.read_parquet(ROOT / "data" / "processed" / "kb_train.parquet").set_index("ticket_id")
    parts = pd.read_csv(ROOT / "splits" / "kb_partitions_provisional.csv")
    q = kb.loc[parts.loc[parts["partition"] == "valid", "ticket_id"]].sample(args.n, random_state=0)
    d = DenseIndex.load(ROOT / "artifacts" / "dense" / "embeddinggemma" / "fit")
    store = QdrantStore.open(ROOT / "artifacts" / "qdrant" / "fit")
    ev = pd.read_parquet(ROOT / "artifacts" / "retrieval" / "fit" / "evidence.parquet").set_index("ticket_id")
    Q = encode([clean_for_search(s, b) for s, b in zip(q["subject"], q["body"])], "query", "embeddinggemma")

    same_ids, same_order, max_diff, payload_ok, lat = 0, 0, 0.0, True, []
    for i in range(len(Q)):
        s = d.vectors @ Q[i]
        top = np.argsort(-s)[:5]
        np_ids, np_sc = [str(d.ids[j]) for j in top], s[top]
        t0 = time.perf_counter()
        hits = store.search(Q[i], 5)
        lat.append((time.perf_counter() - t0) * 1000)
        q_ids, q_sc = [h[0] for h in hits], np.array([h[1] for h in hits])
        same_ids += set(q_ids) == set(np_ids)
        same_order += q_ids == np_ids
        max_diff = max(max_diff, float(np.abs(q_sc - np_sc).max()))
        payload_ok &= all(h[2]["answer"] == str(ev.at[h[0], "answer"]) for h in hits)
    res = {"n_queries": len(Q), "collection_points": store.count(),
           "same_top5_set": f"{same_ids}/{len(Q)}", "same_top5_order": f"{same_order}/{len(Q)}",
           "max_score_difference": round(max_diff, 7), "payload_answers_match_evidence": bool(payload_ok),
           "qdrant_search_ms": {"p50": round(float(np.percentile(lat, 50)), 2), "p95": round(float(np.percentile(lat, 95)), 2)}}
    (ROOT / "reports" / "qdrant_parity.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
