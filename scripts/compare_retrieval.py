"""Compare retrieval modes on validation with paraphrase twins removed (D3-09), plus authored tickets.

    python scripts/compare_retrieval.py

For every validation query, KB documents that look like its paraphrase twin are dropped
from the candidate list before ranking, so modes are compared on what a fresh ticket sees.
A document is a twin if sparse >= 0.5, or a dense backend scores it at or above that
backend's own twin cutoff (10th percentile of its scores on pairs where sparse >= 0.6).
The same twin set is removed for every mode. Writes reports/retrieval_no_twin.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentineldesk.dense import DenseIndex, hybrid_scores  # noqa: E402
from sentineldesk.retrieval import Retriever  # noqa: E402
from sentineldesk.textprep import clean_for_search  # noqa: E402

KB = ROOT / "data" / "processed" / "kb_train.parquet"
PARTS = ROOT / "splits" / "kb_partitions_provisional.csv"
LABELS = ["queue", "type", "priority"]

AUTHORED = {
    "pw_en": "I forgot my password and can still access my registered email. How can I reset my password for the customer portal?",
    "pw_de": "Ich habe mein Passwort für das Kundenportal vergessen und habe noch Zugriff auf meine registrierte E-Mail-Adresse. Wie kann ich das Passwort zurücksetzen?",
    "vpn_de": "VPN-Ausfall im Krankenhaus. Seit heute Morgen können sich alle Mitarbeitenden unserer Klinik nicht mehr per VPN verbinden. Telemedizin und der Zugriff auf die elektronische Patientenakte sind an allen Standorten ausgefallen.",
    "invoice_en": "Where can I download last month's invoice for our subscription?",
    "printer_de": "Unser Drucker im Büro druckt seit gestern nur noch leere Seiten.",
}


def topk_agreement(M: np.ndarray, labs: dict, gold: pd.DataFrame, k: int = 5) -> dict:
    top = np.argsort(-M, axis=1)[:, :k]
    out = {}
    for lab, arr in labs.items():
        nb = arr[top]
        g = gold[lab].to_numpy()[:, None]
        out[lab] = {"top1": round(float((nb[:, 0] == g[:, 0]).mean()), 4),
                    "any_top5": round(float((nb == g).any(axis=1).mean()), 4)}
    return out


def main() -> None:
    kb = pd.read_parquet(KB).set_index("ticket_id")
    parts = pd.read_csv(PARTS)
    v = kb.loc[parts.loc[parts["partition"] == "valid", "ticket_id"]]
    texts = [clean_for_search(s, b) for s, b in zip(v["subject"], v["body"])]

    r = Retriever.load(ROOT / "artifacts" / "retrieval" / "fit")
    S = r.scores(texts)
    dense = {}
    for backend in ["e5", "embeddinggemma"]:
        p = ROOT / "artifacts" / "dense" / backend / "fit"
        if p.exists():
            d = DenseIndex.load(p)
            assert (d.ids == r.ids).all()
            dense[backend] = d.scores(texts)

    twin = S >= 0.5
    cutoffs = {}
    for b, D in dense.items():
        cutoffs[b] = round(float(np.percentile(D[S >= 0.6], 10)), 4)
        twin |= D >= cutoffs[b]
    labs = {lab: kb.loc[r.ids, lab].to_numpy() for lab in LABELS}

    modes = {"sparse": S}
    for b, D in dense.items():
        modes[f"dense_{b}"] = D
        modes[f"hybrid_{b}"] = hybrid_scores(S, D)

    res = {"n_queries": len(v), "twin_rule": {"sparse": 0.5, **cutoffs},
           "queries_with_twin": round(float(twin.any(axis=1).mean()), 4),
           "mean_twins_removed": round(float(twin.sum(axis=1).mean()), 2),
           "with_twins": {}, "no_twin": {}, "authored_top3": {}}
    for name, M in modes.items():
        res["with_twins"][name] = topk_agreement(M, labs, v)
        res["no_twin"][name] = topk_agreement(np.where(twin, -np.inf, M), labs, v)
        nt, wt = res["no_twin"][name], res["with_twins"][name]
        print(f"{name:22} queue top1 {wt['queue']['top1']:.3f} -> no-twin {nt['queue']['top1']:.3f} (any5 {nt['queue']['any_top5']:.3f}) "
              f"| type {nt['type']['top1']:.3f} | prio {nt['priority']['top1']:.3f}")

    full = Retriever.load(ROOT / "artifacts" / "retrieval" / "full")
    full_dense = {b: DenseIndex.load(ROOT / "artifacts" / "dense" / b / "full")
                  for b in dense if (ROOT / "artifacts" / "dense" / b / "full").exists()}
    for key, q in AUTHORED.items():
        t = [clean_for_search(None, q)]
        Sq = full.scores(t)
        res["authored_top3"][key] = {}
        cands = {"sparse": Sq}
        for b, d in full_dense.items():
            Dq = d.scores(t)
            cands[f"dense_{b}"] = Dq
            cands[f"hybrid_{b}"] = hybrid_scores(Sq, Dq)
        print(f"\n### {key}")
        for name, M in cands.items():
            top = np.argsort(-M[0])[:3]
            rows = [{"ticket_id": str(full.ids[i]), "sparse": round(float(Sq[0, i]), 3),
                     "subject": full.evidence["subject"].iloc[i]} for i in top]
            res["authored_top3"][key][name] = rows
            print(f"  {name:22}", " | ".join(f"{x['ticket_id']} s{x['sparse']:.2f} {x['subject'][:34]!r}" for x in rows))

    out = ROOT / "reports" / "retrieval_no_twin.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("\n->", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
