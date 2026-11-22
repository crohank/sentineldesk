"""Train triage variant B (--features embed) or C (--features stacked).

Same partitions as scripts/train_triage.py (variant A):
  train (split811/train)  -> fit heads (and TF-IDF for stacked)
  val cal half            -> choose C (per head) and alpha (stacked, shared), fit temperatures
  val thr half            -> report metrics, rule-6 threshold search
Never reads test files. Saves artifacts/triage_dense/<features>/ and reports/triage_compare/val_<features>.*
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentineldesk.embed import Embedder  # noqa: E402
from sentineldesk.triage import TARGETS  # noqa: E402
from sentineldesk.triage_dense import DenseTriagePredictor, embed_frame, fit_tfidf, make_head, stack  # noqa: E402
from sentineldesk.triage_metrics import ece, outage_search, val_halves  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "split811"
REP = ROOT / "reports" / "triage_compare"
C_GRID = (0.1, 1.0, 10.0)
ALPHA_GRID = (0.5, 1.0, 2.0)
CONF_DEFAULT = 0.45


def fit_score(key, Xtr, ytr, Xcal, ycal, C):
    t0 = time.perf_counter()
    m = make_head(C).fit(Xtr, ytr)
    f1 = f1_score(ycal, m.predict(Xcal), average="macro")
    return key, C, f1, time.perf_counter() - t0, m


def file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["embed", "stacked"], required=True)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--n-jobs", type=int, default=3)
    args = ap.parse_args()
    REP.mkdir(parents=True, exist_ok=True)
    art = ROOT / "artifacts" / "triage_dense" / args.features

    train = pd.read_parquet(DATA / "train.parquet")
    val = pd.read_parquet(DATA / "val.parquet")
    cal, thr = val_halves(val)
    a_halves = pd.read_parquet(ROOT / "reports" / "triage" / "val_predictions.parquet")[["ticket_id", "val_half"]]
    assert set(a_halves.loc[a_halves.val_half == "cal", "ticket_id"]) == set(cal["ticket_id"]), "cal half differs from A"
    print(f"train={len(train)} cal={len(cal)} thr={len(thr)}")

    emb = Embedder(max_len=args.max_len, dim=args.dim)
    d_train, m_train = embed_frame(emb, train)
    d_val, m_val = embed_frame(emb, val)
    pos = {tid: i for i, tid in enumerate(val["ticket_id"])}
    d_cal = d_val[[pos[t] for t in cal["ticket_id"]]]
    d_thr = d_val[[pos[t] for t in thr["ticket_id"]]]
    print(f"embeddings: train cache_hit={m_train['cache_hit']} val cache_hit={m_val['cache_hit']} dim={d_train.shape[1]}")

    t_fit = time.perf_counter()
    tfidf = None
    if args.features == "embed":
        alphas = [1.0]
        mats = {1.0: (d_train, d_cal)}
    else:
        tfidf, Xtr_s = fit_tfidf(train)
        from sentineldesk.triage import normalize
        Xcal_s = tfidf.transform([normalize(s, b) for s, b in zip(cal["subject"], cal["body"])])
        alphas = list(ALPHA_GRID)
        mats = {a: (stack(Xtr_s, d_train, a), stack(Xcal_s, d_cal, a)) for a in alphas}
        print(f"stacked matrix: {mats[1.0][0].shape}, nnz={mats[1.0][0].nnz:,}")

    jobs = [delayed(fit_score)((a, t), mats[a][0], train[t].to_numpy(), mats[a][1], cal[t].to_numpy(), C)
            for a in alphas for t in TARGETS for C in C_GRID]
    print(f"grid: {len(jobs)} fits, n_jobs={args.n_jobs}")
    best: dict = {}
    grid_log = []
    for (a, t), C, f1, secs, m in Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(jobs):
        grid_log.append({"alpha": a, "head": t, "C": C, "cal_macro_f1": f1, "fit_seconds": secs})
        print(f"  alpha={a} {t:8s} C={C:<5} cal macro-F1={f1:.4f} ({secs:.0f}s)")
        if (a, t) not in best or f1 > best[(a, t)][1]:
            best[(a, t)] = (C, f1, m)
    alpha = max(alphas, key=lambda a: np.mean([best[(a, t)][1] for t in TARGETS]))
    heads = {t: best[(alpha, t)][2] for t in TARGETS}
    Cs = {t: best[(alpha, t)][0] for t in TARGETS}
    fit_secs = time.perf_counter() - t_fit
    print(f"chosen alpha={alpha} C={Cs} (grid {fit_secs:.0f}s)")
    del mats

    model = DenseTriagePredictor(features=args.features, embedder_cfg=emb.config(), heads=heads, tfidf=tfidf,
                                 alpha=alpha, C=Cs)
    model.attach_embedder(emb)
    temps = model.calibrate(cal, d_cal)
    print("temperatures:", {k: round(v, 3) for k, v in temps.items()})

    raw = model.proba(thr, d_thr, calibrated=False)
    calp = model.proba(thr, d_thr)
    report = {"variant": {"embed": "B", "stacked": "C"}[args.features], "features": args.features,
              "n": {"train": len(train), "cal": len(cal), "thr": len(thr)}, "alpha": alpha, "C": Cs,
              "temperatures": temps, "grid": grid_log, "grid_seconds": fit_secs, "heads": {}}
    for t in TARGETS:
        y = thr[t].to_numpy()
        pred = calp[t].idxmax(axis=1).to_numpy()
        correct = (pred == y).astype(float)
        e_raw, _ = ece(raw[t].max(axis=1).to_numpy(), correct)
        e_cal, bins = ece(calp[t].max(axis=1).to_numpy(), correct)
        f1 = f1_score(y, pred, average="macro")
        report["heads"][t] = {
            "macro_f1": f1, "accuracy": float(correct.mean()), "ece_raw": e_raw, "ece_calibrated": e_cal,
            "reliability": bins, "per_class": classification_report(y, pred, output_dict=True, zero_division=0),
            "confusion": {"labels": list(model.heads[t].classes_),
                          "matrix": confusion_matrix(y, pred, labels=model.heads[t].classes_).tolist()},
        }
        print(f"{t}: macro-F1={f1:.3f} acc={correct.mean():.3f} ECE raw={e_raw:.3f} cal={e_cal:.3f}")

    gold = ((thr["type"] == "Incident") & (thr["priority"] == "high")).to_numpy()
    out = outage_search(calp["type"], calp["priority"], gold)
    report["outage"] = out
    ch = out["chosen"]
    print(f"classifier Incident&high: recall={out['classifier_recall']:.3f} precision={out['classifier_precision']:.3f}")
    print(f"rule 6 ({out['target_status']}): P(Incident)>={ch['p_incident']} & P(high)>={ch['p_high']} -> "
          f"recall={ch['recall']:.3f} precision={ch['precision']:.3f} escalation={ch['escalation_rate']:.3f}")

    model.thresholds = {"confidence": {t: CONF_DEFAULT for t in TARGETS},
                        "outage_rule": {"p_incident": ch["p_incident"], "p_high": ch["p_high"]}}
    model.meta = {"split": "split811",
                  "data_sha256": {f: file_sha(DATA / f) for f in ["train.parquet", "val.parquet"]},
                  "embedding_cache": {"train": m_train["key"], "val": m_val["key"]},
                  "val_partition": {"method": "train_test_split 50/50 stratified type|priority", "seed": 42},
                  "selection": "C per head and shared alpha by cal-half macro-F1; temperatures on cal; "
                               "metrics and rule 6 on thr",
                  "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    manifest = model.save(art)
    print(f"saved {art} model_sha256={manifest['model_sha256'][:12]}")

    vp = model.predict_batch(val, d_val)
    vp["val_half"] = np.where(val["ticket_id"].isin(cal["ticket_id"]), "cal", "thr")
    for t in TARGETS:
        vp[f"{t}_true"] = val[t]
    vp.to_parquet(REP / f"val_predictions_{args.features}.parquet", index=False)
    (REP / f"val_{args.features}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
