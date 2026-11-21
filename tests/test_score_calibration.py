import numpy as np
import pytest

from scripts.score import calibration_block


def test_calibrated_bins_give_zero_ece():
    conf = [0.25] * 4 + [0.55] * 20 + [0.95] * 20
    correct = [1, 0, 0, 0] + [1] * 11 + [0] * 9 + [1] * 19 + [0]
    c = calibration_block(conf, correct)
    assert c["ece"] == pytest.approx(0.0, abs=1e-12)
    assert [b["n"] for b in c["bins"]] == [0, 0, 4, 0, 0, 20, 0, 0, 0, 20]


def test_sampled_calibrated_confidence_has_small_ece():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.3, 1.0, 200_000)
    correct = rng.uniform(size=conf.size) < conf
    assert calibration_block(conf, correct)["ece"] < 0.01


def test_overconfident_ece_is_mean_gap():
    # Bin (0.8, 0.9]: conf 0.9, accuracy 0.6 on 10 tickets. Bin (0.6, 0.7]: conf 0.7, accuracy 0.5 on 30.
    conf = [0.9] * 10 + [0.7] * 30
    correct = [1] * 6 + [0] * 4 + [1] * 15 + [0] * 15
    c = calibration_block(conf, correct)
    assert c["ece"] == pytest.approx(10 / 40 * 0.3 + 30 / 40 * 0.2)
    b = c["bins"][8]
    assert (b["n"], b["accuracy"]["num"]) == (10, 6)
    assert b["gap"] == pytest.approx(-0.3)


def test_bin_edges_are_right_closed():
    c = calibration_block([0.0, 0.1, 0.1000001, 1.0], [1, 1, 1, 1])
    assert [b["n"] for b in c["bins"]] == [2, 1, 0, 0, 0, 0, 0, 0, 0, 1]
