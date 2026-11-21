"""Collect raw managed predictions and fit per-head temperatures on the D1 split."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sqlite3
import time
from threading import Lock

import httpx
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax

from scripts.score import calibration_block
from sentineldesk.classifier_prompt import HEADS

URL = "https://raw-calibration---sentineldesk-wwnd44xj6q-uc.a.run.app"
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/predictions/managed-00001-h9d-calibration.sqlite"
RESULT = ROOT / "reports/llm/managed_calibration.json"


_lock = Lock()
_next_request = 0.0


def pace():
    global _next_request
    with _lock:
        wait = max(0, _next_request - time.monotonic())
        _next_request = time.monotonic() + wait + 0.75
    time.sleep(wait)


def predict(client, row):
    for attempt in range(4):
        try:
            pace()
            response = client.post(URL + "/api/triage", json={"subject": row.subject or "", "body": row.body})
            response.raise_for_status()
            result = response.json()
            predictions = result["triage"]
            if predictions is None or len(result["precedents"]) != 5:
                raise ValueError("Incomplete managed prediction")
            if any(predictions[h]["calibrated"] is not False for h in HEADS):
                raise RuntimeError("Endpoint no longer emits raw probabilities")
            return json.dumps(predictions)
        except (httpx.HTTPError, ValueError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def collect(rows, db, workers):
    done = {r[0] for r in db.execute("select id from predictions")}
    missing = [r for r in rows.itertuples() if r.ticket_id not in done]
    started = time.monotonic()
    with httpx.Client(timeout=150) as client, ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(predict, client, r): r.ticket_id for r in missing}
        for i, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            try:
                payload = future.result()
            except (httpx.HTTPError, ValueError) as exc:
                print(key, type(exc).__name__, flush=True)
                continue
            with db:
                db.execute("insert or replace into predictions values (?,?)", (key, payload))
            if i == 8 or i % 100 == 0:
                print(f"{i}/{len(missing)} replies in {time.monotonic()-started:.1f}s", flush=True)


def metrics(logits, truth, selected, t):
    logs = log_softmax(logits / t, axis=1)
    n = np.arange(len(truth))
    return {"nll": float(-logs[n, truth].mean()),
            "ece": calibration_block(np.exp(logs[n, selected]), selected == truth)["ece"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pilot", type=int)
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    data = ROOT / "data/processed"
    kb = pd.read_parquet(data / "kb_train.parquet", columns=["ticket_id", "subject", "body", *HEADS])
    frame = kb.merge(pd.read_parquet(data / "kb_partitions.parquet"), validate="one_to_one", on="ticket_id")
    calib = frame.query("partition == 'calib'").sort_values("ticket_id")
    val = frame.query("partition == 'val'").sample(n=300, random_state=42)
    assert len(calib) == 2286
    rows = val if args.validation_only else pd.concat([calib, val])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(CACHE) as db:
        db.execute("create table if not exists predictions(id text primary key, payload text not null)")
        collect(rows.head(args.pilot) if args.pilot else rows, db, args.workers)
        predictions = {k: json.loads(v) for k,v in db.execute("select id,payload from predictions")}
    if args.pilot:
        assert set(rows.head(args.pilot).ticket_id) <= predictions.keys()
        return
    if set(rows.ticket_id) - predictions.keys():
        raise RuntimeError("Missing predictions. Resume with the same command.")
    if args.validation_only:
        print(f"Verified {len(rows)} validation predictions")
        return
    report = {"model": "gemini-3.8-flash", "revision": "sentineldesk-00001-h9d", "k": 5,
              "n_calib": len(calib), "n_validation": len(val), "heads": {}}
    for h in HEADS:
        labels = sorted(kb[h].unique())
        raw = [predictions[k][h] for k in rows.ticket_id]
        logits = np.log(np.clip([[p["probabilities"][l] for l in labels] for p in raw], 1e-12, 1))
        truth = np.array([labels.index(l) for l in rows[h]])
        selected = np.array([labels.index(p["label"]) for p in raw])
        # D2's temperature scaling objective, applied to log probabilities.
        n = len(calib)
        def nll(t):
            return -log_softmax(logits[:n] / t, axis=1)[np.arange(n), truth[:n]].mean()
        t = float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)
        report["heads"][h] = {"temperature": t, "validation_before": metrics(logits[n:],truth[n:],selected[n:],1),
                              "validation_after": metrics(logits[n:],truth[n:],selected[n:],t)}
    RESULT.write_text(json.dumps(report, indent=2) + "\n")
    print(RESULT.read_text())


if __name__ == "__main__":
    main()
