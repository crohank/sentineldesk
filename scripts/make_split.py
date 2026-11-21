"""Build the locked 80/20 split from the main CSV.

Outputs in --out (default data/processed/):
  kb_train.parquet    all columns incl. answer, for training and RAG indexing
  test_inputs.parquet ticket_id, subject, body only (what inference may see)
  test_labels.parquet ticket_id, type, queue, priority, language (eval script only)

The holdout answer column is never written to disk.

Only the main 5-2-50 file is used. `load_dataset(...)` on the HF repo concatenates
all three CSVs (61,765 rows): the 20k file repeats 8,395 bodies from the main file,
so a random split over the concatenation leaks test tickets into the KB, and the
German file uses a different queue/priority taxonomy with no type or answer.

The split is locked: the source hash, row counts, boundary IDs and one spot-checked
ticket must match the values below, or the script raises and writes nothing.
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
OUT = ROOT / "data" / "processed"
SEED = 42
LABELS = ["type", "queue", "priority", "language"]

EXPECTED_SRC_SHA256 = "f187c090e59581c2bbf3aa1377c8db4dd647464ecf2ae51bf8966e42e0ed6bc0"
EXPECTED = {
    "raw rows": 28587,
    "duplicates removed": 15,
    "retained rows": 28572,
    "kb rows": 22857,
    "test rows": 5715,
    "kb first id": "TKT-00000",
    "kb last id": "TKT-22856",
    "test first id": "TKT-22857",
    "test last id": "TKT-28571",
}
SPOT_ID = "TKT-09746"
SPOT_EXPECTED = {"language": "en", "type": "Problem", "queue": "Billing and Payments", "priority": "high"}


class SplitMismatch(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(observed: dict, expected: dict, what: str) -> None:
    bad = [f"  {k}: expected {expected[k]!r}, observed {observed.get(k)!r}"
           for k in expected if observed.get(k) != expected[k]]
    if bad:
        raise SplitMismatch(f"{what} does not match the locked split, nothing written:\n" + "\n".join(bad))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the locked 80/20 split.")
    ap.add_argument("--out", type=Path, default=OUT, help=f"output directory (default {OUT})")
    out = ap.parse_args().out

    # A different file would still parse and split cleanly, just into a different mapping.
    check({"sha256": sha256(SRC)}, {"sha256": EXPECTED_SRC_SHA256}, f"source {SRC}")

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

    check({
        "raw rows": n_raw,
        "duplicates removed": n_raw - len(df),
        "retained rows": len(df),
        "kb rows": len(kb),
        "test rows": len(test),
        "kb first id": kb.ticket_id.iloc[0],
        "kb last id": kb.ticket_id.iloc[-1],
        "test first id": test.ticket_id.iloc[0],
        "test last id": test.ticket_id.iloc[-1],
    }, EXPECTED, "split shape")
    spot = df.loc[df.ticket_id == SPOT_ID, list(SPOT_EXPECTED)]
    check(spot.iloc[0].to_dict() if len(spot) == 1 else {}, SPOT_EXPECTED, f"spot check {SPOT_ID}")

    out.mkdir(parents=True, exist_ok=True)
    kb.to_parquet(out / "kb_train.parquet", index=False)
    test[["ticket_id", "subject", "body"]].to_parquet(out / "test_inputs.parquet", index=False)
    test[["ticket_id", *LABELS]].to_parquet(out / "test_labels.parquet", index=False)

    print(f"raw={n_raw} dedup={len(df)} kb={len(kb)} test={len(test)}")
    print(f"kb ids {kb.ticket_id.iloc[0]}..{kb.ticket_id.iloc[-1]}, test ids {test.ticket_id.iloc[0]}..{test.ticket_id.iloc[-1]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
