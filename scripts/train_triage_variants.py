"""Train triage variants on D1's canonical split.

  A         --features tfidf                      TF-IDF + LR (C=4, fixed)
  B         --features embed                      EmbeddingGemma + LR (C from 0.1..10)
  B-scaled  --features embed --scaled             EmbeddingGemma -> StandardScaler -> LR (C from 0.1..1000)
  B-SVM     --features embed --head svm --scaled  EmbeddingGemma -> StandardScaler -> LinearSVC
  B-LGBM    --features embed --head lgbm          EmbeddingGemma -> LightGBM (no scaling: trees don't need it)

Partitions (data/processed/kb_partitions.parquet, from origin/d1/data-eval):
  fit   -> train heads, TF-IDF vocabulary, StandardScaler
  calib -> choose C / num_leaves per head, LightGBM early stopping, one temperature per head
  val   -> all reported metrics, rule-6 threshold search, underfitting diagnostics
Never reads test_inputs / test_labels.

Run with live logs:
  python -u scripts/train_triage_variants.py --features embed --scaled > reports/triage_d1/logs/train_embed_scaled.log 2>&1
  Get-Content -Wait reports/triage_d1/logs/train_embed_scaled.log        # PowerShell, another terminal

Each head is fit to convergence (ConvergenceWarnings are captured, not silenced; LR continues with
warm_start and SVM refits with doubled max_iter up to 32000) and is checkpointed to
artifacts/triage_d1/<variant>/heads/ as soon as it is chosen.
"""
import argparse
import contextlib
import hashlib
import json
import sys
import threading
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentineldesk.data_d1 import PARTITIONS_SHA256, load_partitions  # noqa: E402
from sentineldesk.triage import TARGETS, TriagePredictor, build_features, normalize  # noqa: E402
from sentineldesk.triage_metrics import ece, latency, outage_search  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports" / "triage_d1"
SEED = 42
MAX_ITER, ITER_CAP = 2000, 32000
LGBM_ROUNDS, LGBM_PATIENCE = 2000, 50
CONF_DEFAULT = 0.45
LC_FRACS = (0.25, 0.5)  # learning-curve subsamples (1.0 = the main model)
# (selection grid, diagnostic sweep). Sweep = grid + one step beyond the top.
GRIDS = {
    "tfidf": ((4.0,), (1.0, 4.0, 16.0, 64.0)),
    "embed": ((0.1, 1.0, 10.0), (0.1, 1.0, 10.0, 100.0, 1000.0)),
    "embed_scaled": ((0.1, 1.0, 10.0, 100.0, 1000.0), (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)),
    "embed_svm_scaled": ((0.001, 0.01, 0.1, 1.0, 10.0), (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)),
    "embed_svm": ((0.001, 0.01, 0.1, 1.0, 10.0), (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)),
    "embed_lgbm": ((15, 31, 63), (15, 31, 63, 127)),
    "tfidf_svm": ((0.03, 0.1, 0.3, 1.0), (0.03, 0.1, 0.3, 1.0, 3.0)),
    "tfidf_lgbm": ((31, 63), (31, 63, 127)),
}
LABELS = {"tfidf": "A", "embed": "B", "embed_scaled": "B-scaled", "embed_svm_scaled": "B-SVM",
          "embed_svm": "B-SVM (unscaled)", "embed_lgbm": "B-LGBM", "tfidf_svm": "TF-IDF + LinearSVC",
          "tfidf_lgbm": "TF-IDF + LightGBM"}
TRAIN_COLS = ["ticket_id", "subject", "body", "type", "queue", "priority", "language"]
T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{time.perf_counter() - T0:7.1f}s] {msg}", flush=True)


@contextlib.contextmanager
def heartbeat(label: str, every: float = 30.0, status=lambda: ""):
    stop = threading.Event()
    start = time.perf_counter()

    def run():
        while not stop.wait(every):
            log(f"  … {label}: still running ({time.perf_counter() - start:.0f}s) {status()}")
    th = threading.Thread(target=run, daemon=True)
    th.start()
    try:
        yield
    finally:
        stop.set()
        th.join()


def _fit_capture(m, X, y, **kw) -> int:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        m.fit(X, y, **kw)
    return sum(issubclass(x.category, ConvergenceWarning) for x in w)


def fit_head(head_type: str, X, y, param, X_es=None, y_es=None):
    """Fit one head to convergence. Returns (model, info). info['convergence_warnings'] is from the final round."""
    t0, rounds = time.perf_counter(), []
    if head_type == "lr":
        m = LogisticRegression(C=param, class_weight="balanced", max_iter=MAX_ITER, warm_start=True)
        while True:
            n_warn = _fit_capture(m, X, y)
            it = int(np.max(m.n_iter_))
            rounds.append({"max_iter": m.max_iter, "n_iter": it, "warnings": n_warn})
            if it < m.max_iter and n_warn == 0:
                converged = True
                break
            if m.max_iter >= ITER_CAP:
                converged = False
                break
            m.max_iter *= 2  # warm_start: continue from current coefficients
    elif head_type == "svm":
        max_iter = MAX_ITER
        while True:
            m = LinearSVC(C=param, class_weight="balanced", max_iter=max_iter, dual="auto", random_state=SEED)
            n_warn = _fit_capture(m, X, y)
            rounds.append({"max_iter": max_iter, "n_iter": int(np.max(m.n_iter_)), "warnings": n_warn})
            if n_warn == 0:
                converged = True
                break
            if max_iter >= ITER_CAP:
                converged = False
                break
            max_iter *= 2
    elif head_type == "lgbm":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=LGBM_ROUNDS, learning_rate=0.1, num_leaves=int(param),
                               colsample_bytree=0.5, class_weight="balanced", random_state=SEED, n_jobs=8,
                               verbose=-1)
        m.fit(X, y, eval_set=[(X_es, y_es)], eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(LGBM_PATIENCE, verbose=False)])
        best = int(m.best_iteration_ or LGBM_ROUNDS)
        rounds.append({"n_estimators": LGBM_ROUNDS, "best_iteration": best, "warnings": 0})
        converged = best < LGBM_ROUNDS
    else:
        raise ValueError(head_type)
    last = rounds[-1]
    return m, {"param": param, "converged": converged, "convergence_warnings": last["warnings"],
               "n_iter_total": sum(r.get("n_iter", r.get("best_iteration", 0)) for r in rounds),
               "rounds": rounds, "seconds": time.perf_counter() - t0}


def head_proba(m, X) -> np.ndarray:
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)
    return softmax(m.decision_function(X), axis=1)  # LinearSVC: uncalibrated softmax of margins


def job(tag, head_type, X, y, param, Xeval, yeval, X_es, y_es):
    m, info = fit_head(head_type, X, y, param, X_es, y_es)
    return tag, info, float(f1_score(yeval, m.predict(Xeval), average="macro"))


def file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def stratified_subsample(df: pd.DataFrame, frac: float) -> np.ndarray:
    idx, _ = train_test_split(np.arange(len(df)), train_size=frac, random_state=SEED,
                              stratify=df["type"] + "|" + df["priority"])
    return np.sort(idx)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["tfidf", "embed"], required=True)
    ap.add_argument("--head", choices=["lr", "svm", "lgbm"], default="lr")
    ap.add_argument("--scaled", action="store_true", help="StandardScaler on embeddings (linear heads only)")
    ap.add_argument("--n-jobs", type=int, default=3)
    ap.add_argument("--extra", action="store_true",
                    help="add D1's kb_extra (minus calib/val twins) to the fit partition")
    ap.add_argument("--no-diagnostics", action="store_true", help="skip the underfitting diagnostics")
    ap.add_argument("--lang", choices=["en", "de"], default=None,
                    help="keep only rows whose `language` column equals this, in fit, calib and val")
    args = ap.parse_args()
    if args.features == "tfidf" and args.scaled:
        ap.error("TF-IDF variants are not scaled")
    if args.head == "lgbm" and args.scaled:
        ap.error("LightGBM is trained on unscaled features")
    feat, head_type = args.features, args.head
    base = feat + ("" if head_type == "lr" else f"_{head_type}") + ("_scaled" if args.scaled else "")
    variant = base + ("_extra" if args.extra else "") + (f"_{args.lang}" if args.lang else "")
    grid, sweep = GRIDS[base]
    LABELS[variant] = LABELS[base] + (" +kb_extra" if args.extra else "") + (f" [{args.lang} only]" if args.lang else "")
    pname = "num_leaves" if head_type == "lgbm" else "C"
    art = ROOT / "artifacts" / "triage_d1" / variant
    (art / "heads").mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)
    log(f"variant {LABELS[variant]} ({variant}): features={feat} head={head_type} scaled={args.scaled} "
        f"grid {pname}={grid} n_jobs={args.n_jobs}")

    P = load_partitions()
    fit, calib, val = P["fit"], P["calib"], P["val"]
    log(f"data ok: partitions sha {PARTITIONS_SHA256[:12]} fit={len(fit)} calib={len(calib)} val={len(val)}")
    extra_stats = None
    if args.extra:
        from sentineldesk.data_d1 import load_kb_extra
        extra, extra_stats = load_kb_extra([calib, val])
        fit = pd.concat([fit[TRAIN_COLS], extra[TRAIN_COLS]], ignore_index=True)
        log(f"kb_extra: {extra_stats} -> fit now {len(fit)} rows (calib/val unchanged, pure kb_train)")
    if args.lang:
        fit, calib, val = (df[df["language"] == args.lang].reset_index(drop=True) for df in (fit, calib, val))
        log(f"language filter `language == {args.lang!r}` (dataset column, noisy): "
            f"fit={len(fit)} calib={len(calib)} val={len(val)}")

    # ---- features ----
    emb, scaler = None, None
    t = time.perf_counter()
    if feat == "tfidf":
        feats = build_features()
        with heartbeat("TF-IDF fit_transform"):
            X = {"fit": feats.fit_transform([normalize(s, b) for s, b in zip(fit["subject"], fit["body"])])}
        for name, df in (("calib", calib), ("val", val)):
            X[name] = feats.transform([normalize(s, b) for s, b in zip(df["subject"], df["body"])])
        log(f"TF-IDF features {X['fit'].shape} nnz={X['fit'].nnz:,} in {time.perf_counter() - t:.1f}s")
        embed_build = None
        XS = X
    else:
        from sentineldesk.embed import Embedder
        from sentineldesk.triage_dense import embed_frame
        emb = Embedder()
        log(f"embedder {emb.model_id}@{emb.revision[:8]} prompt={emb.prompt!r} dim={emb.dim} "
            f"max_len={emb.max_len} device={emb.device}")
        X, embed_build = {}, {}
        for name, df in (("fit", fit), ("calib", calib), ("val", val)):
            with heartbeat(f"embedding {name}"):
                X[name], meta = embed_frame(emb, df, show_progress=False)
            embed_build[name] = {"cache_hit": meta["cache_hit"], "encode_seconds": meta["encode_seconds"],
                                 "key": meta["key"]}
            log(f"embedded {name}: {X[name].shape} cache_hit={meta['cache_hit']} "
                f"encode={meta['encode_seconds']:.1f}s")
        if args.scaled:
            scaler = StandardScaler().fit(X["fit"])  # fit partition only
            XS = {k: scaler.transform(v) for k, v in X.items()}
            log(f"StandardScaler fit on fit partition: mean|μ|={np.abs(scaler.mean_).mean():.4f} "
                f"mean σ={scaler.scale_.mean():.4f} -> applied to calib/val (and live inputs via the predictor)")
        else:
            XS = X
    Y = {name: {h: df[h].to_numpy() for h in TARGETS} for name, df in (("fit", fit), ("calib", calib), ("val", val))}

    # ---- main heads (fit to convergence, select on calib, checkpoint each) ----
    heads, head_info, select_log = {}, {}, []
    for h in TARGETS:
        best = None
        for p in grid:
            log(f"fit head={h} {pname}={p} (rows={len(fit)})")
            with heartbeat(f"head={h} {pname}={p}"):
                m, info = fit_head(head_type, XS["fit"], Y["fit"][h], p, XS["calib"], Y["calib"][h])
            f1c = float(f1_score(Y["calib"][h], m.predict(XS["calib"]), average="macro"))
            select_log.append({"head": h, **info, "calib_macro_f1": f1c})
            status = (f"converged, convergence_warnings={info['convergence_warnings']}" if info["converged"]
                      else f"WARNING: NOT converged (convergence_warnings={info['convergence_warnings']})")
            log(f"  head={h} {pname}={p}: {status}, iters={info['n_iter_total']} ({info['seconds']:.0f}s), "
                f"calib macro-F1={f1c:.4f}")
            if best is None or f1c > best[2]:
                best = (m, info, f1c)
        m, info, f1c = best
        heads[h], head_info[h] = m, {**info, "calib_macro_f1": f1c}
        joblib.dump(m, art / "heads" / f"{h}.joblib", compress=3)
        (art / "heads" / f"{h}.json").write_text(json.dumps(head_info[h], indent=2, default=str), encoding="utf-8")
        edge = " (top of grid)" if p == grid[-1] and len(grid) > 1 and info["param"] == grid[-1] else ""
        log(f"checkpoint saved: heads/{h}.joblib ({pname}={info['param']}{edge}, converged={info['converged']})")
    total_warn = sum(head_info[h]["convergence_warnings"] for h in TARGETS)
    log(f"chosen heads: convergence_warnings total = {total_warn}"
        + ("" if total_warn == 0 else "  <-- WARNING"))

    # ---- calibrate + build predictor ----
    if feat == "tfidf" and head_type == "lr":  # variant A family: unchanged TriagePredictor class
        model = TriagePredictor(features=feats, heads=heads)
        temps = model.calibrate(calib)
        raw, calp = model.proba(val, calibrated=False), model.proba(val)
    elif feat == "tfidf":  # TF-IDF + LinearSVC / LightGBM, same Triage schema
        from sentineldesk.triage_dense import DenseTriagePredictor
        model = DenseTriagePredictor(features="tfidf", embedder_cfg={}, heads=heads, tfidf=feats,
                                     C={h: head_info[h]["param"] for h in TARGETS}, head_type=head_type)
        temps = model.calibrate(calib)
        raw, calp = model.proba(val, calibrated=False), model.proba(val)
    else:
        from sentineldesk.triage_dense import DenseTriagePredictor
        model = DenseTriagePredictor(features="embed", embedder_cfg=emb.config(), heads=heads,
                                     C={h: head_info[h]["param"] for h in TARGETS}, head_type=head_type,
                                     scaler=scaler)
        model.attach_embedder(emb)
        temps = model.calibrate(calib, X["calib"])  # raw vectors; scaler applied inside the predictor
        raw, calp = model.proba(val, X["val"], calibrated=False), model.proba(val, X["val"])
    log("temperatures: " + json.dumps({k: round(v, 3) for k, v in temps.items()}))

    # ---- validation metrics ----
    report = {"variant": LABELS[variant], "name": variant, "features": feat, "head_type": head_type,
              "scaled": bool(args.scaled), "param_name": pname, "split": "d1", "kb_extra": extra_stats,
              "n": {"fit": len(fit), "calib": len(calib), "val": len(val)}, "temperatures": temps,
              "heads_fit": head_info, "selection": select_log, "embedding_build": embed_build, "heads": {}}
    md = [f"# Triage variant {LABELS[variant]} ({variant}) — D1 split, val partition", "",
          f"fit={len(fit)} calib={len(calib)} val={len(val)}; head={head_type}, scaled={bool(args.scaled)}", "",
          f"| head | {pname} | converged (iters, warnings) | macro-F1 | acc | ECE raw | ECE cal | T |",
          "|---|---|---|---|---|---|---|---|"]
    for h in TARGETS:
        y, pred = Y["val"][h], calp[h].idxmax(axis=1).to_numpy()
        correct = (pred == y).astype(float)
        e_raw, _ = ece(raw[h].max(axis=1).to_numpy(), correct)
        e_cal, bins = ece(calp[h].max(axis=1).to_numpy(), correct)
        f1 = float(f1_score(y, pred, average="macro"))
        report["heads"][h] = {"macro_f1": f1, "accuracy": float(correct.mean()), "ece_raw": e_raw,
                              "ece_calibrated": e_cal, "reliability": bins,
                              "per_class": classification_report(y, pred, output_dict=True, zero_division=0),
                              "confusion": {"labels": list(heads[h].classes_),
                                            "matrix": confusion_matrix(y, pred, labels=heads[h].classes_).tolist()}}
        hi = head_info[h]
        md.append(f"| {h} | {hi['param']} | {hi['converged']} ({hi['n_iter_total']}, {hi['convergence_warnings']}) "
                  f"| {f1:.3f} | {correct.mean():.3f} | {e_raw:.3f} | {e_cal:.3f} | {temps[h]:.2f} |")
        log(f"val {h}: macro-F1={f1:.3f} acc={correct.mean():.3f} ECE raw={e_raw:.3f} cal={e_cal:.3f}")
    gold = ((val["type"] == "Incident") & (val["priority"] == "high")).to_numpy()
    out = outage_search(calp["type"], calp["priority"], gold)
    report["outage"] = out
    ch = out["chosen"]
    log(f"Incident&high classifier recall={out['classifier_recall']:.3f} precision={out['classifier_precision']:.3f}; "
        f"rule 6 {out['target_status']}: >= {ch['p_incident']}/{ch['p_high']} recall={ch['recall']:.3f} "
        f"escalation={ch['escalation_rate']:.3f}")
    md += ["", "## Outage (type==Incident & priority==high)", "",
           f"- gold in val: {out['gold_n']}",
           f"- classifier recall {out['classifier_recall']:.3f}, precision {out['classifier_precision']:.3f}",
           f"- rule 6 ({out['target_status']}): P(Incident) ≥ {ch['p_incident']}, P(high) ≥ {ch['p_high']} → "
           f"recall {ch['recall']:.3f}, precision {ch['precision']:.3f}, escalates {ch['escalation_rate']:.1%}"]

    # ---- save final bundle ----
    model.thresholds = {"confidence": {h: CONF_DEFAULT for h in TARGETS},
                        "outage_rule": {"p_incident": ch["p_incident"], "p_high": ch["p_high"]}}
    model.meta = {"variant": LABELS[variant], "split": "d1 (origin/d1/data-eval kb_train + kb_partitions)",
                  "kb_extra": extra_stats,
                  "partitions_sha256": PARTITIONS_SHA256, "param_name": pname,
                  "data_sha256": {"kb_train.parquet": file_sha(ROOT / "data" / "processed" / "kb_train.parquet")},
                  "convergence": {h: {k: head_info[h][k] for k in
                                      ("param", "converged", "convergence_warnings", "n_iter_total", "rounds")}
                                  for h in TARGETS},
                  "selection": f"{pname} per head from {list(grid)} by calib macro-F1; temperatures on calib; "
                               "metrics and rule 6 on val" + ("; StandardScaler fit on fit partition" if scaler else ""),
                  "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    manifest = model.save(art)
    log(f"final model saved: {art} model_sha256={manifest['model_sha256'][:12]}")

    vp = model.predict_batch(val) if feat == "tfidf" else model.predict_batch(val, X["val"])
    assert not vp["ticket_id"].str.startswith("EXT-").any()
    for h in TARGETS:
        vp[f"{h}_true"] = val[h].to_numpy()
    vp["language"] = val["language"].to_numpy()
    vp.to_parquet(REP / f"val_predictions_{variant}.parquet", index=False)

    lat = latency(model.predict, val)
    report["latency"] = lat
    log(f"latency per ticket: p50={lat['p50_ms']:.1f}ms p95={lat['p95_ms']:.1f}ms (n={lat['n']})")
    md += ["", f"Latency per ticket (predict incl. featurization): p50 {lat['p50_ms']:.1f} ms, "
               f"p95 {lat['p95_ms']:.1f} ms"]
    (REP / f"val_{variant}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (REP / f"val_{variant}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- underfitting diagnostics ----
    if args.no_diagnostics:
        log(f"done (diagnostics skipped). reports in {REP}")
        return
    log("underfitting diagnostics: fit/calib/val gap, capacity sweep, learning curve")
    diag = {"param_name": pname, "heads": {}}
    for h in TARGETS:
        m = heads[h]
        row = {}
        for name in ("fit", "calib", "val"):
            pr = head_proba(m, XS[name])
            row[name] = {"macro_f1": float(f1_score(Y[name][h], m.classes_[pr.argmax(1)], average="macro")),
                         "log_loss": float(log_loss(Y[name][h], pr, labels=m.classes_))}
        diag["heads"][h] = {"chosen": head_info[h]["param"], "scores": row,
                            "gap_fit_val": row["fit"]["macro_f1"] - row["val"]["macro_f1"],
                            "sweep": {}, "learning_curve": {1.0: row["val"]["macro_f1"]}}
        log(f"  {h}: macro-F1 fit={row['fit']['macro_f1']:.3f} calib={row['calib']['macro_f1']:.3f} "
            f"val={row['val']['macro_f1']:.3f} (gap {diag['heads'][h]['gap_fit_val']:+.3f}); "
            f"log-loss fit={row['fit']['log_loss']:.3f} val={row['val']['log_loss']:.3f}")
    known = {(r["head"], r["param"]): r["calib_macro_f1"] for r in select_log}
    jobs = []
    for h in TARGETS:
        for p in sweep:
            if (h, p) in known:
                diag["heads"][h]["sweep"][p] = known[(h, p)]
            else:
                jobs.append(delayed(job)(("sweep", h, p), head_type, XS["fit"], Y["fit"][h], p,
                                         XS["calib"], Y["calib"][h], XS["calib"], Y["calib"][h]))
        for frac in LC_FRACS:
            idx = stratified_subsample(fit, frac)
            jobs.append(delayed(job)(("lc", h, frac), head_type, XS["fit"][idx], Y["fit"][h][idx],
                                     head_info[h]["param"], XS["val"], Y["val"][h], XS["calib"], Y["calib"][h]))
    done = [0]
    log(f"  {len(jobs)} diagnostic fits, n_jobs={args.n_jobs}")
    with heartbeat("diagnostic fits", status=lambda: f"{done[0]}/{len(jobs)} done"):
        for (kind, h, v), info, f1 in Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(jobs):
            done[0] += 1
            conv = "" if info["converged"] else f" WARNING: not converged ({info['convergence_warnings']} warnings)"
            if kind == "sweep":
                diag["heads"][h]["sweep"][v] = f1
                log(f"  [{done[0]}/{len(jobs)}] sweep {h} {pname}={v}: calib macro-F1={f1:.4f} "
                    f"({info['n_iter_total']} iters, {info['seconds']:.0f}s){conv}")
            else:
                diag["heads"][h]["learning_curve"][v] = f1
                log(f"  [{done[0]}/{len(jobs)}] learning curve {h} {v:.0%} of fit: val macro-F1={f1:.4f} "
                    f"({info['seconds']:.0f}s){conv}")

    more = "larger num_leaves" if head_type == "lgbm" else "a larger C"
    md_u = [f"# Underfitting check — variant {LABELS[variant]} ({variant}), D1 split", "",
            f"Verdict rules (per head): **underfitting** = fit−val macro-F1 gap < 0.05 and {more} than the "
            "chosen one improves calib macro-F1 by ≥ 0.01; **data-limited** = val macro-F1 rises ≥ 0.01 from 50% "
            "to 100% of fit; **overfitting** = gap > 0.15; otherwise ok. The capacity sweep is diagnostic only. "
            "Learning-curve subsamples reuse the features (and scaler) fitted on the full fit partition.", "",
            f"| head | {pname} | F1 fit | F1 calib | F1 val | gap | log-loss fit / val | best larger-capacity gain | "
            "LC 25% / 50% / 100% | verdict |", "|---|---|---|---|---|---|---|---|---|---|"]
    for h in TARGETS:
        d = diag["heads"][h]
        c0, sw = d["chosen"], d["sweep"]
        base = sw.get(c0, head_info[h]["calib_macro_f1"])
        larger = {p: f - base for p, f in sw.items() if p > c0}
        gain = max(larger.values()) if larger else 0.0
        lc = d["learning_curve"]
        verdict = []
        if d["gap_fit_val"] < 0.05 and gain >= 0.01:
            verdict.append("underfitting")
        if d["gap_fit_val"] > 0.15:
            verdict.append("overfitting")
        if lc[1.0] - lc[0.5] >= 0.01:
            verdict.append("data-limited")
        d["larger_capacity_gain"] = gain
        d["verdict"] = verdict or ["ok"]
        s = d["scores"]
        md_u.append(f"| {h} | {c0} | {s['fit']['macro_f1']:.3f} | {s['calib']['macro_f1']:.3f} | "
                    f"{s['val']['macro_f1']:.3f} | {d['gap_fit_val']:+.3f} | {s['fit']['log_loss']:.3f} / "
                    f"{s['val']['log_loss']:.3f} | {gain:+.4f} | {lc[0.25]:.3f} / {lc[0.5]:.3f} / {lc[1.0]:.3f} | "
                    f"{', '.join(d['verdict'])} |")
        log(f"  verdict {h}: {', '.join(d['verdict'])}")
    md_u += ["", f"Capacity sweep (calib macro-F1 by {pname}):", ""]
    for h in TARGETS:
        md_u.append(f"- {h}: " + ", ".join(f"{pname}={p}: {f:.4f}" for p, f in sorted(diag["heads"][h]["sweep"].items())))
    (REP / f"underfit_{variant}.json").write_text(json.dumps(diag, indent=2, default=str), encoding="utf-8")
    (REP / f"underfit_{variant}.md").write_text("\n".join(md_u) + "\n", encoding="utf-8")
    log(f"done. reports in {REP}")


if __name__ == "__main__":
    main()
