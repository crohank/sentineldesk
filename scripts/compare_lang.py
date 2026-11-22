"""Compare variant A (trained on en+de) vs A-en (trained on English rows only) on the same English val rows.

English = D1 val partition rows with language == "en" (dataset column; noisy). Never reads test files.
Writes reports/triage_d1/compare_en.md.
"""
import sys
from pathlib import Path

from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentineldesk.data_d1 import load_partitions  # noqa: E402
from sentineldesk.triage import TARGETS, TriagePredictor  # noqa: E402
from sentineldesk.triage_metrics import ece, outage_search  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "triage_d1"


def score(model: TriagePredictor, df) -> dict:
    p = model.proba(df)
    out = {}
    for t in TARGETS:
        y, pred = df[t].to_numpy(), p[t].idxmax(axis=1).to_numpy()
        out[f"macro-F1 {t}"] = f1_score(y, pred, average="macro")
        out[f"ECE {t}"] = ece(p[t].max(axis=1).to_numpy(), (pred == y).astype(float))[0]
    gold = ((df["type"] == "Incident") & (df["priority"] == "high")).to_numpy()
    o = outage_search(p["type"], p["priority"], gold)
    out["Incident∧high recall"], out["Incident∧high precision"] = o["classifier_recall"], o["classifier_precision"]
    out["rule 6 best recall"], out["rule 6 escalation"] = o["chosen"]["recall"], o["chosen"]["escalation_rate"]
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    val = load_partitions()["val"]
    en = val[val["language"] == "en"].reset_index(drop=True)
    a, a_en = TriagePredictor.load(ART / "tfidf"), TriagePredictor.load(ART / "tfidf_en")
    sa, se = score(a, en), score(a_en, en)
    lines = ["# A vs A-en on English val rows", "",
             f"n = {len(en)} val rows with language == 'en'. A: trained on all fit rows (en+de); "
             "A-en: trained on English fit rows only. Same features, C=4, temperatures refit on calib (A-en: English "
             "calib rows). Rule-6 thresholds re-searched on these rows for both.", "",
             "| metric | A (en+de) | A-en | Δ |", "|---|---|---|---|"]
    for k in sa:
        pct = "escalation" in k
        f = (lambda v: f"{v:.1%}") if pct else (lambda v: f"{v:.3f}")
        lines.append(f"| {k} | {f(sa[k])} | {f(se[k])} | {se[k] - sa[k]:+.3f} |")
    text = "\n".join(lines) + "\n"
    (ROOT / "reports" / "triage_d1" / "compare_en.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
