"""Score retrieval on the KB validation partition against the fit-only index (D3-07).

    python scripts/eval_retrieval.py [--mode sparse|dense|hybrid] [--backend embeddinggemma|e5]

Label agreement between a query and its neighbours is a coarse proxy for relevance,
not proof the historical answer supports the request. Never reads holdout files.
Writes reports/retrieval_<mode>[_<backend>].json.
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

from sentineldesk.retrieval import Retriever  # noqa: E402
from sentineldesk.textprep import clean_for_search  # noqa: E402

KB = ROOT / "data" / "processed" / "kb_train.parquet"
PARTS = ROOT / "splits" / "kb_partitions_provisional.csv"
INDEX = ROOT / "artifacts" / "retrieval" / "fit"
REPORTS = ROOT / "reports"
K = 5


def load_searcher(mode: str, backend: str):
    if mode == "sparse":
        r = Retriever.load(INDEX)
        return r, r.scores
    from sentineldesk.dense import DenseIndex, hybrid_scores  # optional dependency
    r = Retriever.load(INDEX)
    d = DenseIndex.load(ROOT / "artifacts" / "dense" / backend / "fit")
    assert (d.ids == r.ids).all(), "dense and sparse fit indexes cover different IDs"
    if mode == "dense":
        return r, d.scores
    return r, lambda texts: hybrid_scores(r.scores(texts), d.scores(texts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sparse", "dense", "hybrid"], default="sparse")
    ap.add_argument("--backend", choices=["embeddinggemma", "e5"], default="embeddinggemma")
    args = ap.parse_args()

    kb = pd.read_parquet(KB)
    parts = pd.read_csv(PARTS)
    valid_ids = set(parts.loc[parts["partition"] == "valid", "ticket_id"])
    q = kb[kb["ticket_id"].isin(valid_ids)].reset_index(drop=True)
    r, score_fn = load_searcher(args.mode, args.backend)
    assert not set(r.ids) & valid_ids, "validation ticket inside the fit index"
    ev = r.evidence

    texts = [clean_for_search(s, b) for s, b in zip(q["subject"], q["body"])]
    S = score_fn(texts)
    top = np.argsort(-S, axis=1)[:, :K]
    top_ids = r.ids[top]
    top_scores = np.take_along_axis(S, top, axis=1)

    res = {"mode": args.mode, "backend": None if args.mode == "sparse" else args.backend, "n_queries": len(q), "n_index": int(len(r.ids)), "k": K}
    for lab in ["queue", "type", "priority"]:
        nb = ev[lab].reindex(top_ids.ravel()).to_numpy().reshape(top_ids.shape)
        gold = q[lab].to_numpy()[:, None]
        hit = nb == gold
        share = q[lab].value_counts(normalize=True)
        res[lab] = {
            "top1_agree": round(float(hit[:, 0].mean()), 4),
            "any_top5_agree": round(float(hit.any(axis=1).mean()), 4),
            "majority_vote_top5_acc": round(float(np.mean([
                pd.Series(row).mode().iloc[0] == g for row, g in zip(nb, gold[:, 0])])), 4),
            "chance_top1": round(float((share ** 2).sum()), 4),
        }

    all_ev = pd.read_parquet(ROOT / "artifacts" / "evidence_all.parquet").set_index("ticket_id")
    qlang = all_ev.loc[q["ticket_id"], "body_language"].to_numpy()
    alang = ev["answer_language"].reindex(top_ids[:, 0]).to_numpy()
    blang = ev["body_language"].reindex(top_ids[:, 0]).to_numpy()
    res["top1_score"] = {p: round(float(np.percentile(top_scores[:, 0], p)), 4) for p in (5, 25, 50, 75, 95)}
    res["top5_score_median"] = round(float(np.median(top_scores[:, -1])), 4)
    res["cross_lingual_top1"] = {
        "query_en_hit_de_body": int(((qlang == "en") & (blang == "de")).sum()),
        "query_de_hit_en_body": int(((qlang == "de") & (blang == "en")).sum()),
        "top1_answer_lang_differs_from_query": round(float((alang != qlang).mean()), 4),
    }
    by_lang = {}
    for lang in ["en", "de"]:
        m = qlang == lang
        nbq = ev["queue"].reindex(top_ids[m, 0]).to_numpy()
        by_lang[lang] = {"n": int(m.sum()), "queue_top1_agree": round(float((nbq == q["queue"].to_numpy()[m]).mean()), 4)}
    res["by_query_language"] = by_lang

    lat = []
    for i in range(200):
        t0 = time.perf_counter()
        r.retrieve({"subject": q.at[i, "subject"], "body": q.at[i, "body"]}, K) if args.mode == "sparse" \
            else score_fn([texts[i]])
        lat.append((time.perf_counter() - t0) * 1000)
    res["latency_ms_single_query"] = {"p50": round(float(np.percentile(lat, 50)), 1),
                                      "p95": round(float(np.percentile(lat, 95)), 1)}

    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / (f"retrieval_{args.mode}.json" if args.mode == "sparse" else f"retrieval_{args.mode}_{args.backend}.json")
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("->", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
