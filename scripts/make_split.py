"""Build the locked 80/20 split from the main CSV.

Outputs in data/processed/:
  kb_train.parquet    all columns incl. answer, for training and RAG indexing
  test_inputs.parquet ticket_id, subject, body only (what inference may see)
  test_labels.parquet ticket_id, type, queue, priority, language (eval script only)

The holdout answer column is never written to disk.

Only the main 5-2-50 file is used. `load_dataset(...)` on the HF repo concatenates
all three CSVs (61,765 rows): the 20k file repeats 8,395 bodies from the main file,
so a random split over the concatenation leaks test tickets into the KB, and the
German file uses a different queue/priority taxonomy with no type or answer.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
OUT = ROOT / "data" / "processed"
SEED = 42
LABELS = ["type", "queue", "priority", "language"]


def main() -> None:
    df = pd.read_csv(SRC)
    n_raw = len(df)

    df["subject"] = df["subject"].fillna("")
    df["answer"] = df["answer"].fillna("")
    # ~4% of bodies and answers carry literal "\n" escapes instead of newlines.
    for col in ["subject", "body", "answer"]:
        df[col] = df[col].str.replace("\\n", "\n", regex=False)
    # 15 bodies repeat after strip+lowercase; keeping both could put one in each split.
    key = df["body"].str.strip().str.lower()
    df = df[~key.duplicated(keep="first")]

    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df["ticket_id"] = [f"TKT-{i:05d}" for i in range(len(df))]
    split_idx = int(len(df) * 0.8)
    kb = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    OUT.mkdir(parents=True, exist_ok=True)
    kb.to_parquet(OUT / "kb_train.parquet", index=False)
    test[["ticket_id", "subject", "body"]].to_parquet(OUT / "test_inputs.parquet", index=False)
    test[["ticket_id", *LABELS]].to_parquet(OUT / "test_labels.parquet", index=False)

    print(f"raw={n_raw} dedup={len(df)} kb={len(kb)} test={len(test)}")
    print(f"kb ids {kb.ticket_id.iloc[0]}..{kb.ticket_id.iloc[-1]}, test ids {test.ticket_id.iloc[0]}..{test.ticket_id.iloc[-1]}")


if __name__ == "__main__":
    main()
