"""Guards that keep holdout tickets out of anything built from the KB split.

From the repo root:

    from scripts.leakage import assert_kb_only, holdout_ids

    assert_kb_only(index_ticket_ids)  # raises LeakageError on any non-KB ID
"""
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
NEAR_DUP_IDS = ROOT / "reports" / "near_dup_holdout_ids.txt"


class LeakageError(AssertionError):
    pass


@lru_cache(maxsize=None)
def _ids(path: Path) -> frozenset[str]:
    return frozenset(pd.read_parquet(path, columns=["ticket_id"]).ticket_id)


def kb_ids(processed_dir: Path = PROCESSED) -> frozenset[str]:
    """kb_train IDs plus the approved 20k-file extension (EXT- IDs, D1-38) when it exists."""
    ids = _ids(Path(processed_dir) / "kb_train.parquet")
    extra = Path(processed_dir) / "kb_extra.parquet"
    return ids | _ids(extra) if extra.exists() else ids


def holdout_ids(processed_dir: Path = PROCESSED) -> frozenset[str]:
    return _ids(Path(processed_dir) / "test_inputs.parquet")


def near_dup_holdout_ids(path: Path = NEAR_DUP_IDS) -> frozenset[str]:
    """Holdout IDs with a near-duplicate KB body, from scripts/dup_audit.py."""
    return frozenset(line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())


def assert_kb_only(ids: Iterable, processed_dir: Path = PROCESSED, context: str = "") -> None:
    """Raise LeakageError if any ID is a holdout ticket or is not in the KB (kb_train + kb_extra)."""
    given = {str(i) for i in ids}
    kb = kb_ids(processed_dir)
    holdout = given & holdout_ids(processed_dir)
    unknown = given - kb - holdout
    if not holdout and not unknown:
        return
    where = f"{context}: " if context else ""
    parts = []
    if holdout:
        parts.append(f"{len(holdout)} holdout IDs (e.g. {', '.join(sorted(holdout)[:5])})")
    if unknown:
        parts.append(f"{len(unknown)} IDs not in the KB (e.g. {', '.join(sorted(unknown)[:5])})")
    raise LeakageError(f"{where}{' and '.join(parts)}")
