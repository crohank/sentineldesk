"""Compare triage variants A (TF-IDF), B (EmbeddingGemma), C (stacked) on the val thr half.

All metrics are recomputed here with the same code for every variant. Never reads test files.
Writes reports/triage_compare/{compare.md, compare.json, queue_per_class.csv}.

Decision rule (fixed before results): adopt B or C only if type or queue macro-F1 improves by
>= 0.02 over A, no head drops by more than 0.01, and p95 triage latency < 150 ms.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentineldesk.data_d1 import load_partitions  # noqa: E402
from sentineldesk.embed import Embedder  # noqa: E402
from sentineldesk.triage import TARGETS, TriagePredictor  # noqa: E402
from sentineldesk.triage_dense import DenseTriagePredictor, embed_frame  # noqa: E402
from sentineldesk.triage_metrics import ece, latency, outage_search, val_halves  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "split811"
REP = ROOT / "reports" / "triage_compare"
MIN_GAIN, MAX_DROP, MAX_P95_MS = 0.02, 0.01, 150.0
PROBES = [
    ("", "VPN down for all hospital staff since this morning; telemedicine and EMR unavailable at all sites."),
    ("VPN-Ausfall im Krankenhaus", "Seit heute Morgen können sich alle Mitarbeitenden unserer Klinik nicht mehr per "
     "VPN verbinden. Telemedizin und der Zugriff auf die elektronische Patientenakte sind an allen Standorten ausgefallen."),
    ("", "Where can I view my invoice?"),
]


def evaluate(name: str, proba: dict, thr: pd.DataFrame) -> dict:
    res = {"heads": {}}
    for t in TARGETS:
        y = thr[t].to_numpy()
        pred = proba[t].idxmax(axis=1).to_numpy()
        e, _ = ece(proba[t].max(axis=1).to_numpy(), (pred == y).astype(float))
        labels = sorted(set(y))
        by_lang = {}
        if "language" in thr:
            for lang in sorted(thr["language"].unique()):
                mk = (thr["language"] == lang).to_numpy()
                by_lang[lang] = float(f1_score(y[mk], pred[mk], average="macro"))
        res["heads"][t] = {"macro_f1": f1_score(y, pred, average="macro"), "ece": e, "macro_f1_by_language": by_lang,
                           "per_class_f1": dict(zip(labels, f1_score(y, pred, labels=labels, average=None)))}
    gold = ((thr["type"] == "Incident") & (thr["priority"] == "high")).to_numpy()
    o = outage_search(proba["type"], proba["priority"], gold)
    res["outage"] = {k: o[k] for k in ["gold_n", "classifier_recall", "classifier_precision", "target_status", "chosen"]}
    print(f"{name}: " + "  ".join(f"{t} F1={res['heads'][t]['macro_f1']:.3f}" for t in TARGETS))
    return res


def main() -> None:
    global REP
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["split811", "d1"], default="split811")
    split = ap.parse_args().split
    if split == "d1":
        # D1 canonical split: report on the whole val partition (C/alpha + temperatures used calib)
        val = load_partitions()["val"]
        thr = val
        a_dir, dense_root, REP = (ROOT / "artifacts" / "triage_d1" / "tfidf", ROOT / "artifacts" / "triage_d1",
                                  ROOT / "reports" / "triage_d1")
    else:
        val = pd.read_parquet(DATA / "val.parquet")
        _, thr = val_halves(val)
        a_dir, dense_root = ROOT / "artifacts" / "triage", ROOT / "artifacts" / "triage_dense"
    variants = {}

    def add_tfidf_lr(key, label, d):
        if not (d / "manifest.json").exists():
            print(f"skip {key}: {d} missing")
            return
        m = TriagePredictor.load(d)
        conv = m.meta.get("convergence", {})
        variants[key] = {"label": label, "model": m, "proba": m.proba(thr), "embed": False,
                         "chosen": {t: conv.get(t, {}).get("param", conv.get(t, {}).get("C", 4.0)) for t in TARGETS},
                         "param_name": "C"}

    add_tfidf_lr("A", "A · TF-IDF + LR", a_dir)
    if split == "d1":
        add_tfidf_lr("A+extra", "A+extra · TF-IDF + LR, fit + kb_extra", dense_root / "tfidf_extra")

    dense_dirs = ({"B": "embed", "B-scaled": "embed_scaled", "B-SVM": "embed_svm_scaled", "B-LGBM": "embed_lgbm",
                   "SVM+extra": "tfidf_svm_extra", "SVM-tuned+extra": "tfidf_svm_tuned_extra",
                   "LGBM+extra": "tfidf_lgbm_extra", "C": "stacked"}
                  if split == "d1" else {"B": "embed", "C": "stacked"})
    dense_labels = {"embed": "EmbeddingGemma + LR", "embed_scaled": "EmbeddingGemma → scaler → LR",
                    "embed_svm_scaled": "EmbeddingGemma → scaler → LinearSVC",
                    "embed_lgbm": "EmbeddingGemma + LightGBM", "stacked": "TF-IDF ⊕ EmbeddingGemma",
                    "tfidf_svm_extra": "TF-IDF + LinearSVC, fit + kb_extra",
                    "tfidf_svm_tuned_extra": "TF-IDF + LinearSVC (tuned), fit + kb_extra",
                    "tfidf_lgbm_extra": "TF-IDF + LightGBM, fit + kb_extra"}
    emb = None
    for v, feat in dense_dirs.items():
        d = dense_root / feat
        if not (d / "manifest.json").exists():
            print(f"skip {v}: {d} missing")
            continue
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        if man["features"] == "tfidf":  # no embedder needed
            m = DenseTriagePredictor.load(d)
            variants[v] = {"label": f"{v} · {dense_labels[feat]}", "model": m, "proba": m.proba(thr), "embed": False,
                           "chosen": man["C"], "param_name": man.get("param_name", "C")}
            continue
        cfg = man["embedder"]
        if emb is None:
            emb = Embedder(model_id=cfg["model_id"], revision=cfg["revision"], prompt_name=cfg["prompt_name"],
                           max_len=cfg["max_len"], dim=cfg["dim"])
        m = DenseTriagePredictor.load(d, embedder=emb)
        d_val, meta = embed_frame(emb, val, show_progress=False)
        assert meta["cache_hit"], "val embeddings should come from the cache"
        pos = {tid: i for i, tid in enumerate(val["ticket_id"])}
        d_thr = d_val[[pos[t] for t in thr["ticket_id"]]]
        variants[v] = {"label": f"{v} · {dense_labels[feat]}", "model": m, "proba": m.proba(thr, d_thr),
                       "embed": True, "chosen": man["C"], "param_name": man.get("param_name", "C")}

    if split == "d1":
        b_rep = REP / "val_embed.json"
        eb = json.loads(b_rep.read_text(encoding="utf-8")).get("embedding_build") if b_rep.exists() else None
        build_s = sum(x["encode_seconds"] for x in eb.values()) if eb else None
    else:
        build = REP / "embed_build.json"
        build_s = json.loads(build.read_text(encoding="utf-8"))["total_build_seconds"] if build.exists() else None
    results = {}
    for v, info in variants.items():
        r = evaluate(v, info["proba"], thr)
        r["latency"] = latency(info["model"].predict, thr)
        r["embedding_build_seconds"] = build_s if info.get("embed") else 0.0
        r["chosen"], r["param_name"] = info["chosen"], info["param_name"]
        r["probes"] = []
        for s, b in PROBES:
            p = info["model"].predict(s, b)
            r["probes"].append({"subject": s, "body": b[:60],
                                **{t: (p[t]["label"], round(p[t]["confidence"], 3)) for t in TARGETS}})
        results[v] = r

    # decision rule
    base = results["A"]["heads"]
    decision = {"rule": f"type or queue gain >= {MIN_GAIN}, no head drop > {MAX_DROP}, p95 < {MAX_P95_MS} ms",
                "candidates": {}}
    for v in [x for x in results if x != "A"]:
        delta = {t: results[v]["heads"][t]["macro_f1"] - base[t]["macro_f1"] for t in TARGETS}
        p95 = results[v]["latency"]["p95_ms"]
        ok = (max(delta["type"], delta["queue"]) >= MIN_GAIN and min(delta.values()) >= -MAX_DROP
              and p95 < MAX_P95_MS)
        decision["candidates"][v] = {"delta_macro_f1": delta, "p95_ms": p95, "qualifies": ok}
    qual = [v for v, c in decision["candidates"].items() if c["qualifies"]]
    decision["release"] = (max(qual, key=lambda v: max(decision["candidates"][v]["delta_macro_f1"]["type"],
                                                       decision["candidates"][v]["delta_macro_f1"]["queue"]))
                           if qual else "A")

    # report
    cols = list(results)
    hdr = "| metric | " + " | ".join(variants[v]["label"] for v in cols) + " |"
    lines = ([f"# Triage variant comparison (D1 split, val partition)", "",
              f"n = {len(thr)}. C (B) and temperatures chosen on calib; everything below on val.", ""]
             if split == "d1" else
             ["# Triage variant comparison (split811 val, thr half)", "",
              f"n = {len(thr)}. C/alpha and temperatures chosen on the cal half; everything below on thr.", ""]) + [
             hdr, "|---|" + "---|" * len(cols)]

    def row(name, fn):
        lines.append(f"| {name} | " + " | ".join(fn(results[v]) for v in cols) + " |")

    for t in TARGETS:
        row(f"macro-F1 {t}", lambda r, t=t: f"{r['heads'][t]['macro_f1']:.3f}")
    for t in TARGETS:
        row(f"ECE {t}", lambda r, t=t: f"{r['heads'][t]['ece']:.3f}")
    for t in TARGETS:
        row(f"chosen C / num_leaves — {t}", lambda r, t=t: f"{r['param_name']}={r['chosen'].get(t)}")
    langs = sorted({lg for r in results.values() for lg in r["heads"]["type"]["macro_f1_by_language"]})
    for t in TARGETS:
        for lg in langs:
            row(f"macro-F1 {t} · {lg}", lambda r, t=t, lg=lg: f"{r['heads'][t]['macro_f1_by_language'].get(lg, float('nan')):.3f}")
    row("Incident∧high recall (classifier)", lambda r: f"{r['outage']['classifier_recall']:.3f}")
    row("Incident∧high precision (classifier)", lambda r: f"{r['outage']['classifier_precision']:.3f}")
    row("rule 6 best recall", lambda r: f"{r['outage']['chosen']['recall']:.3f} "
        f"(@{r['outage']['chosen']['p_incident']}/{r['outage']['chosen']['p_high']})")
    row("rule 6 escalation rate", lambda r: f"{r['outage']['chosen']['escalation_rate']:.1%}")
    row("latency p50 / p95 per ticket", lambda r: f"{r['latency']['p50_ms']:.0f} / {r['latency']['p95_ms']:.0f} ms")
    row("one-time embedding build", lambda r: "—" if r["embedding_build_seconds"] in (0.0, None)
        else f"{r['embedding_build_seconds'] / 60:.1f} min")
    if langs:
        lines += ["", f"Language rows use the dataset's `language` column (noisy: ~26% of `de` rows have English "
                      f"bodies). Val counts: {thr['language'].value_counts().to_dict()}."]
    lines += ["", "## Decision", "", f"Rule: {decision['rule']}.", ""]
    for v, c in decision["candidates"].items():
        d = c["delta_macro_f1"]
        lines.append(f"- {v}: Δ type {d['type']:+.3f}, Δ queue {d['queue']:+.3f}, Δ priority {d['priority']:+.3f}, "
                     f"p95 {c['p95_ms']:.0f} ms → {'qualifies' if c['qualifies'] else 'does not qualify'}")
    lines += ["", f"**Release model: {decision['release']}**", "", "## Probe tickets", "",
              "| variant | ticket | type | queue | priority |", "|---|---|---|---|---|"]
    for v in cols:
        for p in results[v]["probes"]:
            lines.append(f"| {v} | {(p['subject'] + ' ' + p['body']).strip()[:50]}… | "
                         + " | ".join(f"{p[t][0]} ({p[t][1]:.2f})" for t in TARGETS) + " |")

    qpc = pd.DataFrame({v: results[v]["heads"]["queue"]["per_class_f1"] for v in cols})
    qpc["support"] = thr["queue"].value_counts()
    qpc.sort_values("support", ascending=False).round(3).to_csv(REP / "queue_per_class.csv")
    lines += ["", "## Queue per-class F1", "", "```",
              qpc.sort_values("support", ascending=False).round(3).to_string(), "```"]

    REP.mkdir(parents=True, exist_ok=True)
    (REP / "compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REP / "compare.json").write_text(json.dumps({"results": results, "decision": decision}, indent=2, default=str),
                                      encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
