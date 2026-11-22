"""Per-head hyperparameter search for TF-IDF + LinearSVC on D1's split (fit + kb_extra).

Grid per head: C {0.1,0.3,1,3,10} x class_weight {None,balanced} x loss {squared_hinge,hinge}, penalty l2.
Select per head by calib macro-F1; if the chosen C is at a grid end, extend one step that way and re-select.
Then one temperature per head on calib (decision_function scores), rule-6 search on val, comparison with A
and the fixed decision rule. TF-IDF features are exactly sentineldesk.triage.build_features().
Never reads test_inputs / test_labels.

  python -u scripts/svm_search.py > reports/triage_d1/logs/svm_search.log 2>&1
"""
import argparse
import contextlib
import json
import re
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from compare_triage import MAX_DROP, MAX_P95_MS, MIN_GAIN, evaluate  # noqa: E402
from sentineldesk.data_d1 import PARTITIONS_SHA256, load_kb_extra, load_partitions  # noqa: E402
from sentineldesk.triage import TARGETS, TriagePredictor, build_features, normalize  # noqa: E402
from sentineldesk.triage_dense import DenseTriagePredictor  # noqa: E402
from sentineldesk.triage_metrics import latency  # noqa: E402

ART = ROOT / "artifacts" / "triage_d1"
REP = ROOT / "reports" / "triage_d1"
NAME = "tfidf_svm_tuned_extra"
SEED = 42
C_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]
CW = [None, "balanced"]
LOSSES = ["squared_hinge", "hinge"]
EXTEND = {0.1: 0.03, 10.0: 30.0}
MAX_ITER0, ITER_CAP = 2000, 64000
CONF_DEFAULT = 0.45
TRAIN_COLS = ["ticket_id", "subject", "body", "type", "queue", "priority", "language"]
T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{time.perf_counter() - T0:7.1f}s] {msg}", flush=True)


@contextlib.contextmanager
def heartbeat(label, status, every=30.0):
    stop = threading.Event()

    def run():
        while not stop.wait(every):
            log(f"  … {label}: {status()}")
    th = threading.Thread(target=run, daemon=True)
    th.start()
    try:
        yield
    finally:
        stop.set()
        th.join()


def fit_svm(X, y, C, cw, loss):
    """LinearSVC to convergence: refit with doubled max_iter until no ConvergenceWarning (cap 64000)."""
    t0, max_iter = time.perf_counter(), MAX_ITER0
    while True:
        m = LinearSVC(C=C, class_weight=cw, loss=loss, penalty="l2", dual=True, max_iter=max_iter, random_state=SEED)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            m.fit(X, y)
        n_warn = sum(issubclass(x.category, ConvergenceWarning) for x in w)
        if n_warn == 0 or max_iter >= ITER_CAP:
            break
        max_iter *= 2
    return m, {"max_iter": max_iter, "n_iter": int(np.max(m.n_iter_)), "warnings": n_warn,
               "seconds": time.perf_counter() - t0}


def job(head, C, cw, loss, X, y, Xc, yc):
    m, info = fit_svm(X, y, C, cw, loss)
    f1 = float(f1_score(yc, m.predict(Xc), average="macro"))
    return {"head": head, "C": C, "class_weight": cw or "None", "loss": loss, "calib_macro_f1": f1, **info}, m


LOG_RE = re.compile(r"\]\s+\[\d+/\d+\] (\w+)\s+C=([\d.]+)\s+cw=(\w+)\s+loss=(\w+)\s+calib macro-F1=([\d.]+) "
                    r"\(max_iter=(\d+), iters=(\d+), (\d+)s\)(?:\s+WARNING: (\d+))?")


def parse_log(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = LOG_RE.search(line)
        if m:
            h, C, cw, loss, f1, mi, it, sec, warn = m.groups()
            rows.append({"head": h, "C": float(C), "class_weight": cw, "loss": loss, "calib_macro_f1": float(f1),
                         "max_iter": int(mi), "n_iter": int(it), "warnings": int(warn or 0), "seconds": float(sec)})
    return rows


def select(rows: list[dict]) -> dict:
    # best calib macro-F1; ties -> smaller C, then balanced, then squared_hinge
    return max(rows, key=lambda r: (round(r["calib_macro_f1"], 6), -r["C"], r["class_weight"] == "balanced",
                                    r["loss"] == "squared_hinge"))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--from-log", default=None,
                    help="skip the grid: read finished fits from a previous svm_search.log, refit the selected heads")
    args = ap.parse_args()
    REP.mkdir(parents=True, exist_ok=True)

    P = load_partitions()
    calib, val = P["calib"], P["val"]
    extra, extra_stats = load_kb_extra([calib, val])
    fit = pd.concat([P["fit"][TRAIN_COLS], extra[TRAIN_COLS]], ignore_index=True)
    log(f"data: partitions sha {PARTITIONS_SHA256[:12]}; kb_extra {extra_stats}; "
        f"fit={len(fit)} calib={len(calib)} val={len(val)}")

    feats = build_features()
    X = feats.fit_transform([normalize(s, b) for s, b in zip(fit["subject"], fit["body"])])
    Xc = feats.transform([normalize(s, b) for s, b in zip(calib["subject"], calib["body"])])
    log(f"TF-IDF (unchanged config) {X.shape} nnz={X.nnz:,}")

    results, models = [], {}
    selected, extensions = {}, {}

    if args.from_log:
        results = parse_log(Path(args.from_log))
        log(f"read {len(results)} finished grid fits from {args.from_log} (search stopped early by request)")
        for h in TARGETS:
            rows = [r for r in results if r["head"] == h]
            best = select(rows)
            added = sorted({r["C"] for r in rows} - set(C_GRID))
            if added:
                n_run = sum(r["C"] in added for r in rows)
                edge = 10.0 if added[0] > 10 else 0.1
                extensions[h] = {"edge_C": edge, "added_C": added[0], "new_selected_C": best["C"],
                                 "still_at_edge": best["C"] == added[0],
                                 "note": f"{n_run}/4 combinations at C={added[0]} finished before the search was "
                                         "stopped (hinge-loss fits at C=30 not run)" if n_run < 4 else ""}
            selected[h] = best
            log(f"selected {h}: C={best['C']} class_weight={best['class_weight']} loss={best['loss']} "
                f"calib macro-F1={best['calib_macro_f1']:.4f}")
        # refit only the selected settings (same data, features, solver, seed)
        jobs = [delayed(job)(h, s["C"], None if s["class_weight"] == "None" else s["class_weight"], s["loss"], X,
                             fit[h].to_numpy(), Xc, calib[h].to_numpy()) for h, s in selected.items()]
        with heartbeat("refit selected heads", lambda: "running"):
            for r, m in Parallel(n_jobs=3)(jobs):
                models[(r["head"], r["C"], r["class_weight"], r["loss"])] = m
                s = selected[r["head"]]
                log(f"refit {r['head']}: calib macro-F1={r['calib_macro_f1']:.4f} (search log {s['calib_macro_f1']:.4f}), "
                    f"max_iter={r['max_iter']}, iters={r['n_iter']}, warnings={r['warnings']}")
                assert abs(r["calib_macro_f1"] - s["calib_macro_f1"]) < 1e-3, "refit does not reproduce the search"
                selected[r["head"]] = {**s, **{k: r[k] for k in ("max_iter", "n_iter", "warnings")}}

    def run(specs):
        jobs = [delayed(job)(h, C, cw, loss, X, fit[h].to_numpy(), Xc, calib[h].to_numpy()) for h, C, cw, loss in specs]
        done = [0]
        log(f"running {len(jobs)} fits, n_jobs={args.n_jobs}")
        with heartbeat("grid", lambda: f"{done[0]}/{len(jobs)} done"):
            for r, m in Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(jobs):
                done[0] += 1
                results.append(r)
                models[(r["head"], r["C"], r["class_weight"], r["loss"])] = m
                warn = "" if r["warnings"] == 0 else f"  WARNING: {r['warnings']} ConvergenceWarning(s) remain"
                log(f"  [{done[0]}/{len(jobs)}] {r['head']:8s} C={r['C']:<5} cw={r['class_weight']:<8} "
                    f"loss={r['loss']:<13} calib macro-F1={r['calib_macro_f1']:.4f} "
                    f"(max_iter={r['max_iter']}, iters={r['n_iter']}, {r['seconds']:.0f}s){warn}")

    if not args.from_log:
        run([(h, C, cw, loss) for h in TARGETS for C in C_GRID for cw in CW for loss in LOSSES])
    for h in ([] if args.from_log else TARGETS):
        best = select([r for r in results if r["head"] == h])
        if best["C"] in EXTEND:
            newC = EXTEND[best["C"]]
            log(f"{h}: selected C={best['C']} is at the grid edge -> extending with C={newC}")
            run([(h, newC, cw, loss) for cw in CW for loss in LOSSES])
            best2 = select([r for r in results if r["head"] == h])
            extensions[h] = {"edge_C": best["C"], "added_C": newC, "new_selected_C": best2["C"],
                             "still_at_edge": best2["C"] == newC}
            best = best2
        selected[h] = best
        log(f"selected {h}: C={best['C']} class_weight={best['class_weight']} loss={best['loss']} "
            f"calib macro-F1={best['calib_macro_f1']:.4f}")

    heads = {h: models[(s["head"], s["C"], s["class_weight"], s["loss"])] for h, s in selected.items()}
    key = lambda r: (r["head"], r["C"], r["class_weight"], r["loss"])  # noqa: E731
    model = DenseTriagePredictor(features="tfidf", embedder_cfg={}, heads=heads, tfidf=feats,
                                 C={h: selected[h]["C"] for h in TARGETS}, head_type="svm")
    temps = model.calibrate(calib)
    log("temperatures (calib, decision_function): " + json.dumps({k: round(v, 3) for k, v in temps.items()}))

    a = TriagePredictor.load(ART / "tfidf")
    res = {"A": evaluate("A", a.proba(val), val), "SVM-tuned": evaluate("SVM-tuned", model.proba(val), val)}
    ch = res["SVM-tuned"]["outage"]["chosen"]
    model.thresholds = {"confidence": {h: CONF_DEFAULT for h in TARGETS},
                        "outage_rule": {"p_incident": ch["p_incident"], "p_high": ch["p_high"]}}
    model.meta = {"variant": "TF-IDF + LinearSVC (tuned) +kb_extra", "split": "d1 (origin/d1/data-eval)",
                  "partitions_sha256": PARTITIONS_SHA256, "kb_extra": extra_stats, "param_name": "C",
                  "settings": {h: {k: selected[h][k] for k in ("C", "class_weight", "loss", "max_iter", "warnings",
                                                                "calib_macro_f1")} for h in TARGETS},
                  "selection": "per head by calib macro-F1 over C x class_weight x loss; temperatures on calib; "
                               "rule 6 on val", "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    manifest = model.save(ART / NAME)
    log(f"saved {ART / NAME} model_sha256={manifest['model_sha256'][:12]}")
    vp = model.predict_batch(val)
    for h in TARGETS:
        vp[f"{h}_true"] = val[h].to_numpy()
    vp.to_parquet(REP / f"val_predictions_{NAME}.parquet", index=False)

    lat = {"A": latency(a.predict, val), "SVM-tuned": latency(model.predict, val)}
    delta = {t: res["SVM-tuned"]["heads"][t]["macro_f1"] - res["A"]["heads"][t]["macro_f1"] for t in TARGETS}
    qualifies = (max(delta["type"], delta["queue"]) >= MIN_GAIN and min(delta.values()) >= -MAX_DROP
                 and lat["SVM-tuned"]["p95_ms"] < MAX_P95_MS)
    log("delta vs A: " + json.dumps({k: round(v, 4) for k, v in delta.items()})
        + f"; p95 {lat['SVM-tuned']['p95_ms']:.1f} ms -> {'QUALIFIES' if qualifies else 'does not qualify'}")

    # ---- report ----
    warn_rows = [r for r in results if r["warnings"]]
    md = ["# TF-IDF + LinearSVC hyperparameter search (D1 split, fit + kb_extra)", "",
          f"fit = {len(fit)} (D1 fit {len(P['fit'])} + kb_extra {extra_stats['kept']}; {extra_stats['dropped']} "
          f"calib/val twins dropped), calib = {len(calib)}, val = {len(val)}. TF-IDF features unchanged "
          f"({X.shape[1]:,} columns). penalty=l2, dual=True. Selection by calib macro-F1 per head; "
          "everything in the comparison is on val.", "",
          f"ConvergenceWarnings remaining after raising max_iter (cap {ITER_CAP:,}): "
          + ("none." if not warn_rows else ", ".join(f"{r['head']} C={r['C']} {r['class_weight']} {r['loss']} "
                                                     f"({r['warnings']})" for r in warn_rows)), "",
          "## 1. Full grid — calib macro-F1", ""]
    for h in TARGETS:
        Cs = sorted({r["C"] for r in results if r["head"] == h})
        md += [f"### {h}", "", "| class_weight | loss | " + " | ".join(f"C={c}" for c in Cs) + " |",
               "|---|---|" + "---|" * len(Cs)]
        for cw in ["None", "balanced"]:
            for loss in LOSSES:
                cells = []
                for c in Cs:
                    r = next((r for r in results if (r["head"], r["C"], r["class_weight"], r["loss"]) == (h, c, cw, loss)), None)
                    if r is None:
                        cells.append("—")
                        continue
                    mark = "**" if key(r) == key(selected[h]) else ""
                    cells.append(f"{mark}{r['calib_macro_f1']:.4f}{mark}" + (" ⚠" if r["warnings"] else ""))
                md.append(f"| {cw} | {loss} | " + " | ".join(cells) + " |")
        md.append("")
    md += ["## 2. Selected settings", "", "| head | C | class_weight | loss | calib macro-F1 | max_iter | T |",
           "|---|---|---|---|---|---|---|"]
    for h in TARGETS:
        s = selected[h]
        md.append(f"| {h} | {s['C']} | {s['class_weight']} | {s['loss']} | {s['calib_macro_f1']:.4f} | "
                  f"{s['max_iter']} | {temps[h]:.3f} |")
    for h, e in extensions.items():
        md.append(f"\n- {h}: C={e['edge_C']} was at the grid edge; added C={e['added_C']} → selected C={e['new_selected_C']}"
                  + (" (still at the new edge; not extended further)" if e["still_at_edge"] else "")
                  + (f". Note: {e['note']}." if e.get("note") else ""))
    if not extensions:
        md.append("\nNo selected C was at a grid end, so no extension was needed.")

    md += ["", "## 3. Comparison on val (n = %d)" % len(val), "", "| metric | A · TF-IDF + LR | SVM-tuned · TF-IDF + LinearSVC +kb_extra |",
           "|---|---|---|"]

    def row(name, fn):
        md.append(f"| {name} | {fn('A')} | {fn('SVM-tuned')} |")
    for t in TARGETS:
        row(f"macro-F1 {t}", lambda v, t=t: f"{res[v]['heads'][t]['macro_f1']:.3f}")
    for t in TARGETS:
        row(f"ECE {t}", lambda v, t=t: f"{res[v]['heads'][t]['ece']:.3f}")
    row("Incident∧high recall (classifier)", lambda v: f"{res[v]['outage']['classifier_recall']:.3f}")
    row("Incident∧high precision (classifier)", lambda v: f"{res[v]['outage']['classifier_precision']:.3f}")
    row("rule 6 best recall", lambda v: f"{res[v]['outage']['chosen']['recall']:.3f} "
        f"(@{res[v]['outage']['chosen']['p_incident']}/{res[v]['outage']['chosen']['p_high']})")
    row("rule 6 escalation rate", lambda v: f"{res[v]['outage']['chosen']['escalation_rate']:.1%}")
    for t in TARGETS:
        for lg in sorted(val["language"].unique()):
            row(f"macro-F1 {t} · {lg}", lambda v, t=t, lg=lg: f"{res[v]['heads'][t]['macro_f1_by_language'][lg]:.3f}")
    row("latency p95 per ticket", lambda v: f"{lat[v]['p95_ms']:.1f} ms")
    md += ["", f"Language rows use the dataset's `language` column (noisy). Val counts: "
               f"{val['language'].value_counts().to_dict()}.", "",
           "## 4. Decision", "",
           f"Rule (fixed): type or queue macro-F1 gain ≥ {MIN_GAIN}, no head drop > {MAX_DROP}, p95 < {MAX_P95_MS:.0f} ms.",
           "", f"Δ vs A: type {delta['type']:+.3f}, queue {delta['queue']:+.3f}, priority {delta['priority']:+.3f}; "
               f"p95 {lat['SVM-tuned']['p95_ms']:.1f} ms → **{'qualifies' if qualifies else 'does not qualify'}**."]
    (REP / "svm_search.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (REP / "svm_search.json").write_text(json.dumps(
        {"grid": results, "selected": selected, "extensions": extensions, "temperatures": temps,
         "comparison": res, "latency": lat, "delta_vs_A": delta, "qualifies": qualifies, "kb_extra": extra_stats},
        indent=2, default=str), encoding="utf-8")
    log(f"done. report: {REP / 'svm_search.md'}")


if __name__ == "__main__":
    main()
