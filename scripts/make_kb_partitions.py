"""Provisional internal KB partition (D3-08): fit / calib / valid = 80 / 10 / 10.

Stratified by type x priority, seed 42. Reads only kb_train.parquet. Replace with
D1's manifest once it is published.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "data" / "processed" / "kb_train.parquet"
OUT = ROOT / "splits" / "kb_partitions_provisional.csv"
SEED = 42


def main() -> None:
    kb = pd.read_parquet(KB, columns=["ticket_id", "type", "priority"])
    strat = kb["type"] + "|" + kb["priority"]
    fit, rest = train_test_split(kb, test_size=0.2, random_state=SEED, stratify=strat)
    calib, valid = train_test_split(rest, test_size=0.5, random_state=SEED, stratify=strat.loc[rest.index])
    parts = pd.concat([
        fit.assign(partition="fit"), calib.assign(partition="calib"), valid.assign(partition="valid"),
    ])[["ticket_id", "partition"]].sort_values("ticket_id")
    assert len(parts) == len(kb) and parts["ticket_id"].is_unique
    assert parts["ticket_id"].max() <= "TKT-22856", "holdout ID in KB partition"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    parts.to_csv(OUT, index=False)
    print(parts["partition"].value_counts().to_dict(), "->", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
