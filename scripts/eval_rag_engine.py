"""Compare Vertex AI RAG Engine retrieval with local EmbeddingGemma on the same queries (D3-16).

    GOOGLE_CLOUD_PROJECT=... python scripts/eval_rag_engine.py [--n 300]

Both search the full KB. For each validation query we exclude the query ticket itself and,
in the "no_twin" setting, its paraphrase twins (sparse >= 0.5 or EmbeddingGemma >= its twin
cutoff, computed once locally), so both systems see exactly the same candidate set.
Also scores the authored demo tickets and an off-topic ticket for threshold guidance.
Writes reports/retrieval_rag_engine.json. Never reads holdout files.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentineldesk.rag_engine import RagEngineRetriever  # noqa: E402
from sentineldesk.retrieval import Retriever  # noqa: E402
from sentineldesk.textprep import clean_for_search  # noqa: E402

LABELS = ["queue", "type", "priority"]
AUTHORED = {
    "pw_en": "I forgot my password and can still access my registered email. How can I reset my password for the customer portal?",
    "pw_de": "Ich habe mein Passwort für das Kundenportal vergessen und habe noch Zugriff auf meine registrierte E-Mail-Adresse. Wie kann ich das Passwort zurücksetzen?",
    "printer_de": "Unser Drucker im Büro druckt seit gestern nur noch leere Seiten.",
    "invoice_en": "Where can I download last month's invoice for our subscription?",
    "vpn_de": "VPN-Ausfall im Krankenhaus. Seit heute Morgen können sich alle Mitarbeitenden unserer Klinik nicht mehr per VPN verbinden.",
    "off_topic_weather": "What is the weather in Paris tomorrow?",
    "off_topic_poem": "Write me a poem about cats.",
}


def agreement(top_ids: list[list[str]], kb: pd.DataFrame, gold: pd.DataFrame) -> dict:
    out = {}
    for lab in LABELS:
        g = gold[lab].to_numpy()
        top1 = np.mean([bool(ids) and kb.at[ids[0], lab] == gi for ids, gi in zip(top_ids, g)])
        any5 = np.mean([any(kb.at[t, lab] == gi for t in ids[:5]) for ids, gi in zip(top_ids, g)])
        out[lab] = {"top1": round(float(top1), 4), "any_top5": round(float(any5), 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    kb = pd.read_parquet(ROOT / "data" / "processed" / "kb_train.parquet").set_index("ticket_id")
    parts = pd.read_csv(ROOT / "splits" / "kb_partitions_provisional.csv")
    q = kb.loc[parts.loc[parts["partition"] == "valid", "ticket_id"]].sample(args.n, random_state=0)
    texts = [clean_for_search(s, b) for s, b in zip(q["subject"], q["body"])]

    local = Retriever.load(ROOT / "artifacts" / "retrieval" / "full", dense=ROOT / "artifacts" / "dense" / "embeddinggemma" / "full")
    rag = RagEngineRetriever.from_manifest(ROOT / "artifacts" / "rag_engine" / "full.json", local=local, timeout=15)

    S = local.scores(texts)                    # sparse
    G = local.dense.scores(texts)              # EmbeddingGemma
    ids = local.ids
    cut = float(np.percentile(G[S >= 0.6], 10))
    self_idx = {tid: i for i, tid in enumerate(ids)}

    results = {"n_queries": args.n, "corpora": rag.corpora, "twin_rule": {"sparse": 0.5, "embeddinggemma": round(cut, 4)}}
    for setting in ["self_only", "no_twin"]:
        excludes = []
        for qi, tid in enumerate(q.index):
            ex = {tid}
            if setting == "no_twin":
                ex |= set(ids[(S[qi] >= 0.5) | (G[qi] >= cut)])
            excludes.append(ex)
        # local EmbeddingGemma with the same exclusions
        g_top = []
        for qi in range(args.n):
            order = np.argsort(-G[qi])
            g_top.append([ids[i] for i in order if ids[i] not in excludes[qi]][:5])
        # RAG Engine with the same exclusions (ask for extra results to cover them)
        lat = []

        def one(qi: int) -> list[str]:
            t0 = time.perf_counter()
            hits = rag.search(texts[qi], 5, exclude=excludes[qi]) if len(excludes[qi]) < 60 else []
            lat.append((time.perf_counter() - t0) * 1000)
            return [t for t, _ in hits]

        with ThreadPoolExecutor(8) as pool:
            r_top = list(pool.map(one, range(args.n)))
        results[setting] = {
            "embeddinggemma": agreement(g_top, kb, q),
            "rag_engine": agreement(r_top, kb, q),
            "rag_engine_empty_results": int(sum(1 for t in r_top if not t)),
            "mean_excluded": round(float(np.mean([len(e) for e in excludes])), 2),
            "rag_engine_latency_ms_under_8_threads": {"p50": round(float(np.percentile(lat, 50)), 1),
                                                     "p95": round(float(np.percentile(lat, 95)), 1)},
        }
        e, r = results[setting]["embeddinggemma"], results[setting]["rag_engine"]
        print(f"{setting:9} queue top1 gemma {e['queue']['top1']:.3f} vs rag {r['queue']['top1']:.3f} | "
              f"type {e['type']['top1']:.3f} vs {r['type']['top1']:.3f} | prio {e['priority']['top1']:.3f} vs {r['priority']['top1']:.3f}")

    seq = []
    results["authored"] = {}
    for key, text in AUTHORED.items():
        t0 = time.perf_counter()
        hits = rag.search(clean_for_search(None, text), 3)
        seq.append((time.perf_counter() - t0) * 1000)
        results["authored"][key] = [{"ticket_id": t, "score": s, "subject": str(kb.at[t, "subject"])[:60]} for t, s in hits]
        print(f"{key:18} " + " | ".join(f"{t} {s:.3f} {str(kb.at[t, 'subject'])[:30]!r}" for t, s in hits))
    results["single_query_latency_ms"] = {"p50": round(float(np.percentile(seq, 50)), 1), "max": round(float(max(seq)), 1)}
    (ROOT / "reports" / "retrieval_rag_engine.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("-> reports/retrieval_rag_engine.json")


if __name__ == "__main__":
    main()
