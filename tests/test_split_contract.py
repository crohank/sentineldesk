"""Contract for the locked 80/20 split in data/processed.

The split is re-derived here from the raw CSV without importing scripts/make_split.py,
so a change to that script that alters the output fails these tests.
"""
import hashlib
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
PROC = ROOT / "data" / "processed"
RAW_SHA256 = "f187c090e59581c2bbf3aa1377c8db4dd647464ecf2ae51bf8966e42e0ed6bc0"
LABELS = ["type", "queue", "priority", "language"]
TEST_INPUT_COLS = ["ticket_id", "subject", "body"]
TEST_LABEL_COLS = ["ticket_id", *LABELS]


def norm(s: pd.Series) -> pd.Series:
    return s.str.strip().str.lower()


def decode(s: pd.Series) -> pd.Series:
    return s.str.replace("\\n", "\n", regex=False)


@pytest.fixture(scope="module")
def split():
    files = [PROC / f for f in ("kb_train.parquet", "test_inputs.parquet", "test_labels.parquet")]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        pytest.skip(f"processed split missing: {missing}")
    return tuple(pd.read_parquet(f) for f in files)


@pytest.fixture(scope="module")
def source():
    if not RAW.exists():
        pytest.skip(f"raw CSV missing: {RAW.name}")
    return pd.read_csv(RAW)


@pytest.fixture(scope="module")
def expected(source):
    d = source.copy()
    d["subject"] = d["subject"].fillna("")
    d["answer"] = d["answer"].fillna("")
    for c in ["subject", "body", "answer"]:
        d[c] = decode(d[c])
    dups = norm(d["body"]).duplicated(keep="first")
    d = d[~dups].sample(frac=1.0, random_state=42).reset_index(drop=True)
    d["ticket_id"] = [f"TKT-{i:05d}" for i in range(len(d))]
    cut = int(len(d) * 0.8)
    return {"n_dups": int(dups.sum()), "all": d, "kb": d.iloc[:cut].reset_index(drop=True), "test": d.iloc[cut:].reset_index(drop=True)}


def test_source_sha256():
    if not RAW.exists():
        pytest.skip(f"raw CSV missing: {RAW.name}")
    assert hashlib.sha256(RAW.read_bytes()).hexdigest() == RAW_SHA256


def test_source_rows(source):
    assert len(source) == 28587


def test_rederivation_drops_15_duplicates(expected):
    assert expected["n_dups"] == 15
    assert len(expected["all"]) == 28572


def test_row_counts(split):
    kb, ti, tl = split
    assert len(kb) == 22857
    assert len(ti) == len(tl) == 5715


def test_id_ranges(split):
    kb, ti, tl = split
    assert kb.ticket_id.tolist() == [f"TKT-{i:05d}" for i in range(22857)]
    assert ti.ticket_id.tolist() == [f"TKT-{i:05d}" for i in range(22857, 28572)]


def test_holdout_column_sets(split):
    kb, ti, tl = split
    assert list(ti.columns) == TEST_INPUT_COLS
    assert list(tl.columns) == TEST_LABEL_COLS
    assert "answer" in kb.columns


def test_ids_unique_and_disjoint(split):
    kb, ti, tl = split
    assert kb.ticket_id.is_unique and ti.ticket_id.is_unique and tl.ticket_id.is_unique
    assert not set(kb.ticket_id) & set(ti.ticket_id)


def test_inputs_and_labels_aligned(split):
    _, ti, tl = split
    assert (ti.ticket_id.to_numpy() == tl.ticket_id.to_numpy()).all()


def test_kb_matches_rederivation(split, expected):
    kb = split[0]
    exp = expected["kb"]
    assert list(kb.columns) == list(exp.columns)
    pd.testing.assert_frame_equal(kb.reset_index(drop=True), exp)


def test_test_inputs_match_rederivation(split, expected):
    pd.testing.assert_frame_equal(split[1].reset_index(drop=True), expected["test"][TEST_INPUT_COLS])


def test_test_labels_match_rederivation(split, expected):
    pd.testing.assert_frame_equal(split[2].reset_index(drop=True), expected["test"][TEST_LABEL_COLS])


def test_no_nulls_in_text(split):
    kb, ti, _ = split
    assert kb[["subject", "body", "answer"]].notna().all().all()
    assert ti[["subject", "body"]].notna().all().all()


def test_no_literal_backslash_n(split):
    kb, ti, _ = split
    for s in [kb.subject, kb.body, kb.answer, ti.subject, ti.body]:
        assert not s.str.contains("\\n", regex=False).any()


def test_no_normalized_body_duplicates(split):
    kb, ti, _ = split
    assert pd.concat([norm(kb.body), norm(ti.body)]).is_unique


def test_no_exact_cross_split_bodies(split):
    kb, ti, _ = split
    assert not set(norm(kb.body)) & set(norm(ti.body))


def test_test_labels_unchanged_vs_source(split, source):
    _, ti, tl = split
    src = source.assign(k=norm(decode(source.body))).drop_duplicates("k")
    m = tl.assign(k=norm(ti.body)).merge(src[["k", *LABELS]], on="k", suffixes=("", "_src"))
    assert len(m) == 5715
    for c in LABELS:
        assert (m[c] == m[f"{c}_src"]).all(), c


def test_spot_check_tkt_09746(split):
    r = split[0].set_index("ticket_id").loc["TKT-09746"]
    assert (r.language, r.type, r.queue, r.priority) == ("en", "Problem", "Billing and Payments", "high")
    assert r.subject.startswith("Problem with Several Subscriptions")
