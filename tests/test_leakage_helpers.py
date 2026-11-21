import numpy as np
import pandas as pd
import pytest

from scripts import leakage
from scripts.leakage import LeakageError, assert_kb_only, holdout_ids, kb_ids, near_dup_holdout_ids


@pytest.fixture
def toy_dir(tmp_path):
    pd.DataFrame({"ticket_id": ["TKT-00000", "TKT-00001", "TKT-00002"], "body": ["a", "b", "c"]}).to_parquet(tmp_path / "kb_train.parquet")
    pd.DataFrame({"ticket_id": ["TKT-00003", "TKT-00004"], "subject": ["", ""], "body": ["d", "e"]}).to_parquet(tmp_path / "test_inputs.parquet")
    return tmp_path


def test_id_sets(toy_dir):
    assert kb_ids(toy_dir) == {"TKT-00000", "TKT-00001", "TKT-00002"}
    assert holdout_ids(toy_dir) == {"TKT-00003", "TKT-00004"}


def test_kb_only_passes(toy_dir):
    assert_kb_only(["TKT-00000", "TKT-00002", "TKT-00002"], toy_dir)
    assert_kb_only([], toy_dir)
    assert_kb_only(pd.Series(["TKT-00001"]), toy_dir)
    assert_kb_only(np.array(["TKT-00001"]), toy_dir)


def test_holdout_id_raises(toy_dir):
    with pytest.raises(LeakageError, match=r"1 holdout IDs \(e\.g\. TKT-00004\)"):
        assert_kb_only(["TKT-00000", "TKT-00004"], toy_dir)


def test_unknown_id_raises(toy_dir):
    with pytest.raises(LeakageError, match="not in the KB"):
        assert_kb_only(["TKT-00000", "TKT-99999"], toy_dir)


def test_context_in_message(toy_dir):
    with pytest.raises(LeakageError, match="^bm25 index: "):
        assert_kb_only(["TKT-00003"], toy_dir, context="bm25 index")


def test_leakage_error_is_assertion_error(toy_dir):
    with pytest.raises(AssertionError):
        assert_kb_only(["TKT-00003"], toy_dir)


def test_near_dup_file_parsing(tmp_path):
    p = tmp_path / "ids.txt"
    p.write_text("TKT-1\n\nTKT-2\n", encoding="utf-8")
    assert near_dup_holdout_ids(p) == {"TKT-1", "TKT-2"}


@pytest.fixture
def real_split():
    if not (leakage.PROCESSED / "kb_train.parquet").exists() or not (leakage.PROCESSED / "test_inputs.parquet").exists():
        pytest.skip("processed split missing")


def test_real_split_sizes_and_disjoint(real_split):
    has_extra = (leakage.PROCESSED / "kb_extra.parquet").exists()
    assert len(kb_ids()) == 22857 + (11489 if has_extra else 0)
    assert len(holdout_ids()) == 5715
    assert not kb_ids() & holdout_ids()


def test_extra_ids_accepted_and_disjoint(real_split):
    extra = leakage.PROCESSED / "kb_extra.parquet"
    if not extra.exists():
        pytest.skip("kb_extra.parquet not built")
    ext = set(pd.read_parquet(extra, columns=["ticket_id"]).ticket_id)
    assert all(i.startswith("EXT-") for i in ext)
    assert not ext & holdout_ids()
    assert_kb_only(sorted(ext)[:50] + ["TKT-00000"])


def test_real_kb_passes_and_holdout_fails(real_split):
    assert_kb_only(kb_ids())
    with pytest.raises(LeakageError):
        assert_kb_only([*sorted(kb_ids())[:10], "TKT-22857"])


def test_real_near_dup_ids_are_holdout(real_split):
    if not leakage.NEAR_DUP_IDS.exists():
        pytest.skip("near-dup report missing")
    flagged = near_dup_holdout_ids()
    assert flagged
    assert flagged <= holdout_ids()
