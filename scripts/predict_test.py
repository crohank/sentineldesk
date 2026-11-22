"""Write D1-format holdout predictions (docs/prediction_format.md on origin/d1/data-eval) for each
finished D2 triage variant. Reads only data/processed/test_inputs.parquet; never test labels.

    python scripts/predict_test.py [--variants tfidf tfidf_svm_tuned_extra ...]

-> data/predictions/d2_<variant>.parquet, scored with D1's scripts/run_eval.py.
decision_status: escalate_oncall when the variant's rule-6 thresholds (chosen on val) fire,
human_review when any head is below its confidence threshold, else auto_draft.
The release model was fixed on val (reports/triage_d1/compare.md) before any test scoring.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentineldesk.triage import TARGETS, TriagePredictor  # noqa: E402
from sentineldesk.triage_dense import DenseTriagePredictor, embed_frame  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "triage_d1"
INPUTS = ROOT / "data" / "processed" / "test_inputs.parquet"
INPUTS_SHA256 = "535ba1e086290e6ef92c4f1d31b3f8b27d81dd0739c1242fa921842dfab14b98"
OUT = ROOT / "data" / "predictions"
DEFAULT = ["tfidf", "tfidf_extra", "embed", "tfidf_svm_extra", "tfidf_svm_tuned_extra"]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=DEFAULT)
    args = ap.parse_args()

    sha = hashlib.sha256(INPUTS.read_bytes()).hexdigest()
    if sha != INPUTS_SHA256:
        raise RuntimeError(f"test_inputs sha256 {sha[:12]} != D1's {INPUTS_SHA256[:12]}")
    test = pd.read_parquet(INPUTS)
    assert list(test.columns) == ["ticket_id", "subject", "body"], test.columns
    OUT.mkdir(parents=True, exist_ok=True)

    emb = None
    for v in args.variants:
        d = ART / v
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        t0 = time.perf_counter()
        if "features" not in man:  # variant A family (TriagePredictor)
            m = TriagePredictor.load(d)
            proba = m.proba(test)
        elif man.get("features") == "tfidf":
            m = DenseTriagePredictor.load(d)
            proba = m.proba(test)
        else:
            from sentineldesk.embed import Embedder
            cfg = man["embedder"]
            if emb is None:
                emb = Embedder(model_id=cfg["model_id"], revision=cfg["revision"], prompt_name=cfg["prompt_name"],
                               max_len=cfg["max_len"], dim=cfg["dim"])
            m = DenseTriagePredictor.load(d, embedder=emb)
            dense, _ = embed_frame(emb, test)
            proba = m.proba(test, dense)
        per_ticket_ms = (time.perf_counter() - t0) * 1000 / len(test)

        thr = man["thresholds"]
        out = pd.DataFrame({"ticket_id": test["ticket_id"]})
        for t in TARGETS:
            out[f"pred_{t}"] = proba[t].idxmax(axis=1).astype(str).to_numpy()
            out[f"conf_{t}"] = proba[t].max(axis=1).to_numpy()
        oncall = ((proba["type"]["Incident"] >= thr["outage_rule"]["p_incident"])
                  & (proba["priority"]["high"] >= thr["outage_rule"]["p_high"])).to_numpy()
        low = pd.concat([out[f"conf_{t}"] < thr["confidence"][t] for t in TARGETS], axis=1).any(axis=1).to_numpy()
        out["decision_status"] = ["escalate_oncall" if o else "human_review" if lw else "auto_draft"
                                  for o, lw in zip(oncall, low)]
        out["latency_ms"] = per_ticket_ms  # batch time / n (amortized, not end-to-end)
        path = OUT / f"d2_{v}.parquet"
        out.to_parquet(path, index=False)
        print(f"{v}: {len(out)} rows -> {path.relative_to(ROOT).as_posix()}  "
              f"status {out['decision_status'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
