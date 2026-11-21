"""D2 working split: official HF loader -> filter -> dedup -> 80/10/10.

Starts from the problem statement's snippet (datadownloadsnippet.txt), which stacks all
three CSVs in the HF repo (~61,765 rows). Changes vs. the snippet:
  * rows without type/answer or with labels outside the main taxonomy are dropped
    (removes the German-normalized file: no type/answer, 42 queues, 5 priorities);
  * duplicates on normalized subject+body are removed BEFORE shuffling, so a ticket
    cannot appear in two splits (the 20k file repeats many main-file tickets);
  * the shuffled data is cut 80/10/10 into train / val / test.

Outputs in data/split811/:
  train.parquet        all columns incl. answer (training + RAG)
  val.parquet          all columns except answer (calibration + threshold selection)
  test_inputs.parquet  ticket_id, subject, body
  test_labels.parquet  ticket_id, type, queue, priority, language (eval script only)
Test answers are never written to disk.

This is separate from D1's canonical split (scripts/make_split.py -> data/processed/).
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "split811"
SEED = 42
TYPES = {"Incident", "Request", "Problem", "Change"}
PRIORITIES = {"high", "medium", "low"}
QUEUES = {
    "Technical Support", "Product Support", "Customer Service", "IT Support",
    "Billing and Payments", "Returns and Exchanges", "Service Outages and Maintenance",
    "Sales and Pre-Sales", "Human Resources", "General Inquiry",
}
LABELS = ["type", "queue", "priority", "language"]


def frame_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    df = load_dataset("Tobi-Bueck/customer-support-tickets")["train"].to_pandas()
    print(f"loaded: {len(df)} rows")

    for col in ["subject", "body", "answer"]:
        df[col] = df[col].fillna("").str.replace("\\n", "\n", regex=False)

    steps = [
        ("empty body", df["body"].str.strip() != ""),
        ("missing type / unknown type", df["type"].isin(TYPES)),
        ("empty answer", df["answer"].str.strip() != ""),
        ("priority not high/medium/low", df["priority"].isin(PRIORITIES)),
        ("queue outside main taxonomy", df["queue"].isin(QUEUES)),
    ]
    keep = pd.Series(True, index=df.index)
    for name, mask in steps:
        dropped = int((keep & ~mask).sum())
        keep &= mask
        print(f"drop {name}: {dropped}")
    df = df[keep]
    print(f"after filters: {len(df)}")

    key = df["subject"].str.strip().str.lower() + "\n" + df["body"].str.strip().str.lower()
    dup = key.duplicated(keep="first")
    print(f"drop duplicate subject+body: {int(dup.sum())}")
    df = df[~dup]
    body_dups = int(df["body"].str.strip().str.lower().duplicated().sum())
    print(f"remaining body-only duplicates (different subject, kept): {body_dups}")

    assert set(df["type"]) == TYPES, set(df["type"])
    assert set(df["queue"]) == QUEUES, set(df["queue"])
    assert set(df["priority"]) == PRIORITIES, set(df["priority"])

    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df["ticket_id"] = [f"TKT-{i:05d}" for i in range(len(df))]
    n = len(df)
    i80, i90 = int(n * 0.8), int(n * 0.9)
    train, val, test = df.iloc[:i80], df.iloc[i80:i90], df.iloc[i90:]

    ids = [set(x["ticket_id"]) for x in (train, val, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    k = [set(x["subject"].str.strip().str.lower() + "\n" + x["body"].str.strip().str.lower()) for x in (train, val, test)]
    assert not (k[0] & k[1] or k[0] & k[2] or k[1] & k[2])

    OUT.mkdir(parents=True, exist_ok=True)
    train.to_parquet(OUT / "train.parquet", index=False)
    val.drop(columns=["answer"]).to_parquet(OUT / "val.parquet", index=False)
    test_inputs = test[["ticket_id", "subject", "body"]]
    test_labels = test[["ticket_id", *LABELS]]
    test_inputs.to_parquet(OUT / "test_inputs.parquet", index=False)
    test_labels.to_parquet(OUT / "test_labels.parquet", index=False)

    print(f"\ndedup={n} train={len(train)} val={len(val)} test={len(test)}")
    print(f"ids: train {train.ticket_id.iloc[0]}..{train.ticket_id.iloc[-1]}, "
          f"val {val.ticket_id.iloc[0]}..{val.ticket_id.iloc[-1]}, "
          f"test {test.ticket_id.iloc[0]}..{test.ticket_id.iloc[-1]}")
    print("hashes:", {"train": frame_hash(train), "val": frame_hash(val.drop(columns=["answer"])),
                      "test_inputs": frame_hash(test_inputs), "test_labels": frame_hash(test_labels)})
    print("\nlabel share drift vs train (max abs pct-points):")
    for c in LABELS:
        a = train[c].value_counts(normalize=True)
        gaps = {name: round(float((a - s[c].value_counts(normalize=True)).abs().max() * 100), 2)
                for name, s in (("val", val), ("test", test))}
        print(f"  {c}: {gaps}")


if __name__ == "__main__":
    main()
