"""Train the triage predictor on data/split811/train, calibrate + choose thresholds on val.

val is split 50/50 (stratified on type x priority, seed 42):
  cal half -> one temperature per head
  thr half -> reported metrics, confidence-coverage curves, outage-rule thresholds

Writes artifacts/triage/{model.joblib, manifest.json} and reports/triage/*.
Never reads test files.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentineldesk.triage import TARGETS, TriagePredictor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "split811"
ART = ROOT / "artifacts" / "triage"
REP = ROOT / "reports" / "triage"
SEED = 42
CONF_DEFAULT = 0.45
OUTAGE_GRID = np.round(np.arange(0.30, 0.701, 0.05), 2)
OUTAGE_RECALL_TARGET = 0.90


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    rows, total = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            gap = abs(conf[m].mean() - correct[m].mean())
            total += m.mean() * gap
            rows.append({"bin": b, "lo": edges[b], "hi": edges[b + 1], "n": int(m.sum()),
                         "conf": float(conf[m].mean()), "acc": float(correct[m].mean())})
    return float(total), rows


def file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    REP.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(DATA / "train.parquet")
    val = pd.read_parquet(DATA / "val.parquet")
    strat = val["type"] + "|" + val["priority"]
    cal, thr = train_test_split(val, test_size=0.5, random_state=SEED, stratify=strat)
    print(f"train={len(train)} cal={len(cal)} thr={len(thr)}")

    t0 = time.perf_counter()
    model = TriagePredictor.fit(train)
    print(f"fit: {time.perf_counter() - t0:.1f}s")
    temps = model.calibrate(cal)
    print("temperatures:", {k: round(v, 3) for k, v in temps.items()})

    raw = model.proba(thr, calibrated=False)
    calp = model.proba(thr, calibrated=True)
    report: dict = {"n": {"train": len(train), "cal": len(cal), "thr": len(thr)}, "temperatures": temps, "heads": {}}
    lines = ["# Triage — validation report (thr half of split811 val)", "",
             f"train={len(train)} cal={len(cal)} thr={len(thr)}", "",
             "| head | macro-F1 | acc | majority acc | ECE raw | ECE calibrated | T |", "|---|---|---|---|---|---|---|"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, t in zip(axes, TARGETS):
        y = thr[t].to_numpy()
        pred = calp[t].idxmax(axis=1).to_numpy()
        majority = train[t].value_counts().idxmax()
        f1 = f1_score(y, pred, average="macro")
        acc = float((pred == y).mean())
        maj_acc = float((y == majority).mean())
        maj_f1 = f1_score(y, np.full_like(y, majority), average="macro")
        correct = (pred == y).astype(float)
        e_raw, _ = ece(raw[t].max(axis=1).to_numpy(), correct)
        e_cal, bins = ece(calp[t].max(axis=1).to_numpy(), correct)

        conf = calp[t].max(axis=1).to_numpy()
        curve = []
        for c in np.round(np.arange(0.30, 0.901, 0.05), 2):
            keep = conf >= c
            curve.append({"threshold": float(c), "coverage": float(keep.mean()),
                          "accuracy_on_covered": float(correct[keep].mean()) if keep.any() else None})

        report["heads"][t] = {
            "macro_f1": f1, "accuracy": acc, "majority_label": majority, "majority_accuracy": maj_acc,
            "majority_macro_f1": maj_f1, "ece_raw": e_raw, "ece_calibrated": e_cal, "reliability": bins,
            "confidence_curve": curve,
            "per_class": classification_report(y, pred, output_dict=True, zero_division=0),
            "confusion": {"labels": list(model.heads[t].classes_),
                          "matrix": confusion_matrix(y, pred, labels=model.heads[t].classes_).tolist()},
        }
        lines.append(f"| {t} | {f1:.3f} | {acc:.3f} | {maj_acc:.3f} | {e_raw:.3f} | {e_cal:.3f} | {temps[t]:.2f} |")
        print(f"{t}: macro-F1={f1:.3f} acc={acc:.3f} (majority acc {maj_acc:.3f}, macro-F1 {maj_f1:.3f}) "
              f"ECE raw={e_raw:.3f} cal={e_cal:.3f}")

        ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1)
        ax.plot([b["conf"] for b in bins], [b["acc"] for b in bins], marker="o", label="calibrated")
        _, braw = ece(raw[t].max(axis=1).to_numpy(), correct)
        ax.plot([b["conf"] for b in braw], [b["acc"] for b in braw], marker="x", alpha=0.6, label="raw")
        ax.set_title(f"{t} (ECE {e_cal:.3f})")
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.legend()
    fig.tight_layout()
    fig.savefig(REP / "reliability.png", dpi=110)

    # Per-head confidence coverage
    lines += ["", "## Confidence threshold → coverage / accuracy on covered", "",
              "| threshold | " + " | ".join(f"{t} cov | {t} acc" for t in TARGETS) + " |",
              "|---|" + "---|---|" * len(TARGETS)]
    for i, c in enumerate(report["heads"]["type"]["confidence_curve"]):
        cells = []
        for t in TARGETS:
            r = report["heads"][t]["confidence_curve"][i]
            cells.append(f"{r['coverage']:.2f} | {r['accuracy_on_covered']:.3f}" if r["accuracy_on_covered"] is not None
                         else f"{r['coverage']:.2f} | -")
        lines.append(f"| {c['threshold']:.2f} | " + " | ".join(cells) + " |")

    # Outage: classifier joint prediction (gate rule 2) + probability rule (gate rule 6)
    gold = ((thr["type"] == "Incident") & (thr["priority"] == "high")).to_numpy()
    cls_hit = ((calp["type"].idxmax(axis=1) == "Incident") & (calp["priority"].idxmax(axis=1) == "high")).to_numpy()
    p_inc = calp["type"]["Incident"].to_numpy()
    p_high = calp["priority"]["high"].to_numpy()
    cls_rec = float(cls_hit[gold].mean())
    cls_prec = float(gold[cls_hit].mean()) if cls_hit.any() else 0.0
    print(f"classifier Incident&high: recall={cls_rec:.3f} ({int(cls_hit[gold].sum())}/{int(gold.sum())}) "
          f"precision={cls_prec:.3f}")
    grid = []
    for a in OUTAGE_GRID:
        for b in OUTAGE_GRID:
            fire = cls_hit | ((p_inc >= a) & (p_high >= b))
            grid.append({"p_incident": float(a), "p_high": float(b),
                         "recall": float(fire[gold].mean()), "precision": float(gold[fire].mean()),
                         "escalation_rate": float(fire.mean())})
    ok = [g for g in grid if g["recall"] >= OUTAGE_RECALL_TARGET]
    if ok:
        chosen = max(ok, key=lambda g: (g["precision"], g["recall"]))
        status = "met"
    else:
        chosen = max(grid, key=lambda g: (g["recall"], g["precision"]))
        status = "not met (best attainable recall)"
    print(f"outage rule ({status}): P(Incident)>={chosen['p_incident']} & P(high)>={chosen['p_high']} -> "
          f"combined recall={chosen['recall']:.3f} precision={chosen['precision']:.3f} "
          f"escalation={chosen['escalation_rate']:.3f}")
    report["outage"] = {"gold_n": int(gold.sum()), "classifier_recall": cls_rec, "classifier_precision": cls_prec,
                        "target_recall": OUTAGE_RECALL_TARGET, "target_status": status, "chosen": chosen, "grid": grid}
    lines += ["", "## Outage (type==Incident & priority==high)", "",
              f"- gold on thr half: {int(gold.sum())}",
              f"- classifier joint prediction: recall {cls_rec:.3f} ({int(cls_hit[gold].sum())}/{int(gold.sum())}), "
              f"precision {cls_prec:.3f}",
              f"- probability rule, target recall {OUTAGE_RECALL_TARGET:.0%} {status}: "
              f"P(Incident) ≥ {chosen['p_incident']}, P(high) ≥ {chosen['p_high']}",
              f"- combined gate (classifier OR rule): recall {chosen['recall']:.3f}, precision {chosen['precision']:.3f}, "
              f"escalates {chosen['escalation_rate']:.1%} of tickets"]

    model.thresholds = {
        "confidence": {t: CONF_DEFAULT for t in TARGETS},
        "outage_rule": {"p_incident": chosen["p_incident"], "p_high": chosen["p_high"]},
    }
    model.meta = {
        "split": "split811",
        "data_sha256": {f: file_sha(DATA / f) for f in ["train.parquet", "val.parquet"]},
        "val_partition": {"method": "train_test_split 50/50 stratified type|priority", "seed": SEED},
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    manifest = model.save(ART)
    print(f"saved {ART} model_sha256={manifest['model_sha256'][:12]}")

    vp = model.predict_batch(val)
    vp["val_half"] = np.where(val["ticket_id"].isin(cal["ticket_id"]), "cal", "thr")
    for t in TARGETS:
        vp[f"{t}_true"] = val[t]
    vp.to_parquet(REP / "val_predictions.parquet", index=False)

    (REP / "val_metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (REP / "val_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
