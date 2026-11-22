"""Load D1's canonical KB split (origin/d1/data-eval) as fit / calib / val frames.

Files (copied from D1's branch into the gitignored data/processed/):
  kb_train.parquet       22,857 KB tickets (locked 80/20 split, seed 42)
  kb_partitions.parquet  ticket_id -> fit | calib | val (80/10/10, stratified type|priority, seed 42)
Never touches test_inputs / test_labels.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PARTITIONS_SHA256 = "7e70ffd4794be9f9d331bbd578378bc5219f1e164053ec2026c24c465db7f4c4"  # reports/kb_partitions_summary.md
EXPECTED_COUNTS = {"fit": 18285, "calib": 2286, "val": 2286}


def load_partitions(proc: Path = PROC) -> dict[str, pd.DataFrame]:
    part_file = proc / "kb_partitions.parquet"
    sha = hashlib.sha256(part_file.read_bytes()).hexdigest()
    if sha != PARTITIONS_SHA256:
        raise RuntimeError(f"kb_partitions.parquet sha256 {sha[:12]} != D1's {PARTITIONS_SHA256[:12]}")
    kb = pd.read_parquet(proc / "kb_train.parquet")
    parts = pd.read_parquet(part_file)
    if not parts["ticket_id"].is_unique or set(parts["ticket_id"]) != set(kb["ticket_id"]):
        raise RuntimeError("partitions are not a one-to-one cover of kb_train ticket_ids")
    if not kb["ticket_id"].str.startswith("TKT-").all():
        raise RuntimeError("unexpected non-TKT ids in kb_train")
    df = kb.merge(parts, on="ticket_id", validate="one_to_one")
    out = {p: df[df["partition"] == p].drop(columns="partition").reset_index(drop=True) for p in EXPECTED_COUNTS}
    counts = {p: len(v) for p, v in out.items()}
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"partition counts {counts} != {EXPECTED_COUNTS}")
    return out


TWIN_COSINE = 0.8  # D1's twin rule (scripts/twin_audit.py): word 1-2 gram TF-IDF cosine on body


def load_kb_extra(exclude_twins_of: list[pd.DataFrame], proc: Path = PROC) -> tuple[pd.DataFrame, dict]:
    """D1's approved 20k-file KB extension (EXT- ids), minus tickets that duplicate or paraphrase
    any ticket in `exclude_twins_of` (our calib/val partitions). D1 already removed holdout twins.
    Returns (rows, stats). Use for training the fit stage only.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    extra = pd.read_parquet(proc / "kb_extra.parquet")
    if not extra["ticket_id"].str.startswith("EXT-").all():
        raise RuntimeError("kb_extra contains non-EXT ids")
    ref = pd.concat(exclude_twins_of, ignore_index=True)
    key = lambda s: s.str.strip().str.lower()  # noqa: E731
    exact = key(extra["body"]).isin(set(key(ref["body"])))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit(pd.concat([extra["body"], ref["body"]]))
    E, R = vec.transform(extra["body"]), vec.transform(ref["body"])
    max_cos = np.zeros(len(extra))
    for i in range(0, len(extra), 2000):  # chunked to bound memory
        max_cos[i:i + 2000] = (E[i:i + 2000] @ R.T).max(axis=1).toarray().ravel()
    twin = exact.to_numpy() | (max_cos >= TWIN_COSINE)
    stats = {"kb_extra_rows": len(extra), "exact_body_matches": int(exact.sum()),
             f"cosine_ge_{TWIN_COSINE}": int((max_cos >= TWIN_COSINE).sum()), "dropped": int(twin.sum()),
             "kept": int((~twin).sum())}
    return extra[~twin].reset_index(drop=True), stats
