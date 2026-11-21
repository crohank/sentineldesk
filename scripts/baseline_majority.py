"""Majority-label baseline: predicts the most common kb_train label per head for every holdout ticket.

Reads only kb_train and test_inputs. Writes data/predictions/majority.parquet.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "predictions" / "majority.parquet"
TARGETS = ["type", "queue", "priority"]


def main() -> None:
    kb = pd.read_parquet(PROC / "kb_train.parquet", columns=TARGETS)
    ids = pd.read_parquet(PROC / "test_inputs.parquet", columns=["ticket_id"])
    pred = ids.copy()
    for t in TARGETS:
        pred[f"pred_{t}"] = kb[t].value_counts().index[0]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT).as_posix()} ({len(pred)} rows):",
          {t: pred[f'pred_{t}'].iloc[0] for t in TARGETS})


if __name__ == "__main__":
    main()
