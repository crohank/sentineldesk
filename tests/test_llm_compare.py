import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from scripts import llm_compare as cmp


def test_id_set_mismatch_fails():
    ok = pd.DataFrame({"ticket_id": ["a", "b", "c"]})
    cmp.assert_same_ids({"ok": ok}, ["c", "b", "a"])
    with pytest.raises(cmp.CompareError, match="1 missing"):
        cmp.assert_same_ids({"short": ok.iloc[:2]}, ["a", "b", "c"])
    with pytest.raises(cmp.CompareError, match="1 extra"):
        cmp.assert_same_ids({"swapped": pd.DataFrame({"ticket_id": ["a", "b", "z"]})}, ["a", "b", "c"])
    with pytest.raises(cmp.CompareError, match="duplicate"):
        cmp.assert_same_ids({"dup": pd.DataFrame({"ticket_id": ["a", "a", "b"]})}, ["a", "b"])


def test_macro_f1_matches_sklearn_default():
    rng = np.random.default_rng(0)
    gold = rng.choice(list("abcd"), 200)
    pred = rng.choice(list("abcde"), 200)  # "e" only in pred, which sklearn counts as a zero-F1 label
    assert cmp.macro_f1(gold, pred) == pytest.approx(f1_score(gold, pred, average="macro", zero_division=0))


def test_bootstrap_ci_contains_zero_for_identical_predictions():
    rng = np.random.default_rng(1)
    gold = rng.choice(list("abc"), 150)
    pred = np.where(rng.uniform(size=150) < 0.7, gold, rng.choice(list("abc"), 150))
    b = cmp.paired_bootstrap(gold, pred, pred, n_boot=200)
    assert b["diff"] == 0 and b["lo"] <= 0 <= b["hi"] and not b["excludes_0"]


def test_bootstrap_ci_excludes_zero_for_a_much_better_model():
    rng = np.random.default_rng(2)
    gold = rng.choice(list("abc"), 300)
    b = cmp.paired_bootstrap(gold, gold, rng.choice(list("abc"), 300), n_boot=200)
    assert b["lo"] > 0 and b["excludes_0"]


def test_vote_breaks_ties_by_nearest_neighbour():
    assert cmp.vote(["b", "a", "a", "b", "c"]) == ("b", 0.4)
    assert cmp.vote(["c", "a", "a"]) == ("a", pytest.approx(2 / 3))


def test_twin_flag_on_toy_example():
    pool = pd.DataFrame({
        "ticket_id": ["P0", "P1", "P2", "P3"],
        "subject": ["printer jammed", "printer offline", "invoice wrong", "invoice late"],
        "body": ["the office printer is jammed again today", "the office printer is offline again",
                 "my invoice shows the wrong amount", "my invoice arrived late"],
    })
    queries = pd.DataFrame({
        "ticket_id": ["Q0", "Q1"],
        "subject": ["printer jammed", "vpn down"],
        "body": ["the office printer is jammed again today", "cannot connect to the vpn from home"],
    })
    t = cmp.twin_flags(pool, queries).set_index("ticket_id")
    assert t.loc["Q0", "is_twin"] and t.loc["Q0", "twin_id"] == "P0"
    assert t.loc["Q0", "twin_sim"] == pytest.approx(1.0)
    assert not t.loc["Q1", "is_twin"]
