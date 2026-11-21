"""Does a cross-encoder reranker help on top of EmbeddingGemma? (D3-13)

    python scripts/eval_rerank.py [--n 500] [--pool 20]

Takes the top-`pool` EmbeddingGemma candidates for validation queries (paraphrase twins
removed, D3-09), reranks them with BAAI/bge-reranker-v2-m3, and compares label agreement
and latency before and after. Needs requirements-dense.txt and Ollama.
Writes reports/rerank_eval.json.
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

from sentineldesk.dense import DenseIndex  # noqa: E402
from sentineldesk.retrieval import Retriever  # noqa: E402
from sentineldesk.textprep import clean_for_search  # noqa: E402

RERANKER = "BAAI/bge-reranker-v2-m3"
LABELS = ["queue", "type", "priority"]


def agreement(cand_ids: np.ndarray, kb: pd.DataFrame, gold: pd.DataFrame) -> dict:
    out = {}
    for lab in LABELS:
        nb = kb[lab].reindex(cand_ids.ravel()).to_numpy().reshape(cand_ids.shape)
        g = gold[lab].to_numpy()[:, None]
        out[lab] = {"top1": round(float((nb[:, 0] == g[:, 0]).mean()), 4),
                    "any_top5": round(float((nb[:, :5] == g).any(axis=1).mean()), 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--pool", type=int, default=20)
    args = ap.parse_args()

    import torch
    from sentence_transformers import CrossEncoder
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ce = CrossEncoder(RERANKER, device=device, max_length=512)

    kb = pd.read_parquet(ROOT / "data" / "processed" / "kb_train.parquet").set_index("ticket_id")
    parts = pd.read_csv(ROOT / "splits" / "kb_partitions_provisional.csv")
    v = kb.loc[parts.loc[parts["partition"] == "valid", "ticket_id"]].sample(args.n, random_state=0)
    qtext = [clean_for_search(s, b) for s, b in zip(v["subject"], v["body"])]

    r = Retriever.load(ROOT / "artifacts" / "retrieval" / "fit")
    g = DenseIndex.load(ROOT / "artifacts" / "dense" / "embeddinggemma" / "fit")
    S, G = r.scores(qtext), g.scores(qtext)
    twin = (S >= 0.5) | (G >= float(np.percentile(G[S >= 0.6], 10)))
    e5_path = ROOT / "artifacts" / "dense" / "e5" / "fit"
    if e5_path.exists():
        E = DenseIndex.load(e5_path).scores(qtext)
        twin |= E >= float(np.percentile(E[S >= 0.6], 10))
    doc_text = {tid: clean_for_search(kb.at[tid, "subject"], kb.at[tid, "body"]) for tid in r.ids}

    res = {"reranker": RERANKER, "first_stage": "embeddinggemma", "n_queries": args.n, "pool": args.pool}
    for setting, M in [("no_twin", np.where(twin, -np.inf, G)), ("with_twins", G)]:
        pool = np.argsort(-M, axis=1)[:, :args.pool]
        before = r.ids[pool]
        after, lat = [], []
        for qi in range(args.n):
            pairs = [(qtext[qi], doc_text[t]) for t in before[qi]]
            t0 = time.perf_counter()
            ce_s = ce.predict(pairs, batch_size=args.pool, show_progress_bar=False)
            lat.append((time.perf_counter() - t0) * 1000)
            after.append(before[qi][np.argsort(-ce_s)])
        after = np.array(after)
        res[setting] = {
            "embeddinggemma": agreement(before, kb, v),
            "embeddinggemma+rerank": agreement(after, kb, v),
            "top1_changed": round(float((after[:, 0] != before[:, 0]).mean()), 4),
            "rerank_latency_ms": {"p50": round(float(np.percentile(lat, 50)), 1),
                                  "p95": round(float(np.percentile(lat, 95)), 1)},
        }
        b, a = res[setting]["embeddinggemma"], res[setting]["embeddinggemma+rerank"]
        print(f"{setting:10} queue top1 {b['queue']['top1']:.3f} -> {a['queue']['top1']:.3f} | "
              f"any5 {b['queue']['any_top5']:.3f} -> {a['queue']['any_top5']:.3f} | "
              f"type {b['type']['top1']:.3f} -> {a['type']['top1']:.3f} | prio {b['priority']['top1']:.3f} -> {a['priority']['top1']:.3f} | "
              f"top1 changed {res[setting]['top1_changed']:.0%} | rerank p50 {res[setting]['rerank_latency_ms']['p50']} ms")

    out = ROOT / "reports" / "rerank_eval.json"
    out.write_text(json.dumps(res, indent=2))
    print("->", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
