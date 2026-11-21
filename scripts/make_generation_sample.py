"""Preselect 100 holdout ticket_ids for the generation audit, fixed before any scoring.

Reads only test_inputs.parquet. Writes reports/generation_sample_ids.txt, one ID per line, sorted.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEST_INPUTS = ROOT / "data" / "processed" / "test_inputs.parquet"
OUT = ROOT / "reports" / "generation_sample_ids.txt"
SEED = 42
N = 100


def main() -> None:
    # Sorting first makes the draw independent of row order in the parquet.
    ids = (
        pd.read_parquet(TEST_INPUTS, columns=["ticket_id"])["ticket_id"]
        .sort_values()
        .reset_index(drop=True)
    )
    if not ids.is_unique:
        raise RuntimeError("duplicate ticket_id in test_inputs")
    sample = sorted(ids.sample(n=N, random_state=SEED))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(sample) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(sample)} ids to {OUT.relative_to(ROOT)}: {sample[0]}..{sample[-1]}")


if __name__ == "__main__":
    main()
