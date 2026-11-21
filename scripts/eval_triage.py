"""Holdout evaluation of the triage predictor on data/split811 test.

Predicts from test_inputs only, writes + hashes predictions, and only then opens test_labels.
Writes reports/triage/test_predictions.parquet, test_metrics.json, test_report.md.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentineldesk.triage import TARGETS, TriagePredictor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_triage import ece  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "split811"
ART = ROOT / "artifacts" / "triage"
REP = ROOT / "reports" / "triage"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    REP.mkdir(parents=True, exist_ok=True)
    model = TriagePredictor.load(ART)

    inputs = pd.read_parquet(DATA / "test_inputs.parquet")
    assert list(inputs.columns) == ["ticket_id", "subject", "body"], inputs.columns
    preds = model.predict_batch(inputs)
    pred_file = REP / "test_predictions.parquet"
    preds.to_parquet(pred_file, index=False)
    pred_sha = hashlib.sha256(pred_file.read_bytes()).hexdigest()
    print(f"predictions: {len(preds)} rows, sha256={pred_sha[:16]}")

    # ---- labels opened only after predictions are frozen ----
    labels = pd.read_parquet(DATA / "test_labels.parquet")
    assert preds["ticket_id"].is_unique and set(preds["ticket_id"]) == set(labels["ticket_id"])
    df = labels.merge(preds, on="ticket_id", validate="one_to_one")
    train_major = json.loads((REP / "val_metrics.json").read_text(encoding="utf-8"))["heads"]

    report = {"n": len(df), "predictions_sha256": pred_sha, "model_sha256": json.loads(
        (ART / "manifest.json").read_text(encoding="utf-8"))["model_sha256"], "heads": {}}
    lines = ["# Triage — holdout report (split811 test)", "", f"n = {len(df)}  ·  predictions sha256 `{pred_sha[:16]}`",
             "", "| head | macro-F1 | acc | majority macro-F1 | majority acc | ECE |", "|---|---|---|---|---|---|"]
    for t in TARGETS:
        y, p = df[t].to_numpy(), df[f"{t}_pred"].to_numpy()
        maj = train_major[t]["majority_label"]
        f1 = f1_score(y, p, average="macro")
        maj_f1 = f1_score(y, np.full_like(y, maj), average="macro")
        e, bins = ece(df[f"{t}_conf"].to_numpy(), (p == y).astype(float))
        classes = model.heads[t].classes_
        report["heads"][t] = {
            "macro_f1": f1, "accuracy": float((p == y).mean()), "majority_label": maj,
            "majority_macro_f1": maj_f1, "majority_accuracy": float((y == maj).mean()), "ece": e, "reliability": bins,
            "per_class": classification_report(y, p, output_dict=True, zero_division=0),
            "confusion": {"labels": list(classes), "matrix": confusion_matrix(y, p, labels=classes).tolist()},
        }
        lines.append(f"| {t} | {f1:.3f} | {(p == y).mean():.3f} | {maj_f1:.3f} | {(y == maj).mean():.3f} | {e:.3f} |")
        print(f"{t}: macro-F1={f1:.3f} acc={(p == y).mean():.3f} majority macro-F1={maj_f1:.3f} ECE={e:.3f}")

    gold = ((df["type"] == "Incident") & (df["priority"] == "high")).to_numpy()
    cls_hit = ((df["type_pred"] == "Incident") & (df["priority_pred"] == "high")).to_numpy()
    rule = model.thresholds["outage_rule"]
    fire = cls_hit | ((df["p_type_Incident"] >= rule["p_incident"]) & (df["p_priority_high"] >= rule["p_high"])).to_numpy()
    out = {
        "gold_n": int(gold.sum()),
        "classifier": {"tp": int((cls_hit & gold).sum()), "predicted": int(cls_hit.sum()),
                       "recall": float(cls_hit[gold].mean()), "precision": float(gold[cls_hit].mean())},
        "combined_gate": {"rule": rule, "tp": int((fire & gold).sum()), "fired": int(fire.sum()),
                          "recall": float(fire[gold].mean()), "precision": float(gold[fire].mean()),
                          "escalation_rate": float(fire.mean())},
    }
    report["outage"] = out
    c, g = out["classifier"], out["combined_gate"]
    print(f"High-priority Incident recall (classifier): {c['recall']:.3f} ({c['tp']}/{out['gold_n']}), "
          f"precision {c['precision']:.3f} ({c['tp']}/{c['predicted']})")
    print(f"Combined gate: recall {g['recall']:.3f} ({g['tp']}/{out['gold_n']}), precision {g['precision']:.3f}, "
          f"escalates {g['escalation_rate']:.1%}")
    lines += ["", "## High-priority Incident (type==Incident & priority==high)", "",
              f"- classifier recall **{c['recall']:.3f}** ({c['tp']}/{out['gold_n']}), "
              f"precision {c['precision']:.3f} ({c['tp']}/{c['predicted']})",
              f"- classifier OR probability rule (P(Incident) ≥ {rule['p_incident']}, P(high) ≥ {rule['p_high']}): "
              f"recall {g['recall']:.3f} ({g['tp']}/{out['gold_n']}), precision {g['precision']:.3f}, "
              f"escalates {g['fired']}/{len(df)} ({g['escalation_rate']:.1%})"]

    (REP / "test_metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (REP / "test_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
