"""Train D2's classifier on the canonical masked fit rows and measure fast-path coverage."""
import json
from pathlib import Path
import pandas as pd
from sentineldesk.safety.pii import mask_pii
from sentineldesk.triage import TriagePredictor, TARGETS


def main():
    root = Path("data/processed")
    frame = pd.read_parquet(root/"kb_train.parquet", columns=["ticket_id","subject","body",*TARGETS]).merge(
        pd.read_parquet(root/"kb_partitions.parquet"), on="ticket_id", validate="one_to_one")
    for field in ("subject","body"):
        frame[field] = frame[field].fillna("").map(lambda text: mask_pii(text).text)
    model = TriagePredictor.fit(frame.query("partition == 'fit'"))
    model.calibrate(frame.query("partition == 'calib'"))
    model.thresholds = {"min_head_confidence": 0.90}
    val = frame.query("partition == 'val'")
    pred = model.predict_batch(val)
    chosen = pred[[f"{h}_conf" for h in TARGETS]].min(axis=1) >= model.thresholds["min_head_confidence"]
    actual = (val.type == "Incident") & (val.priority == "high")
    predicted = (pred.type_pred == "Incident") & (pred.priority_pred == "high")
    model.meta = {"validation_rows":len(val), "fast_rows":int(chosen.sum()),
                  "correct_all_three":int((chosen & pd.concat([pred[f"{h}_pred"] == val[h] for h in TARGETS],axis=1).all(axis=1)).sum()),
                  "fast_high_incidents":int((chosen & actual).sum()),
                  "fast_high_incidents_detected":int((chosen & actual & predicted).sum())}
    print(json.dumps(model.meta), flush=True)
    model.save("artifacts/fast_path")


if __name__ == "__main__":
    main()
