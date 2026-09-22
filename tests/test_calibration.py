"""Nothing is called calibrated until it is measured to be."""

import random

from mimir.eval.calibration import (
    MIN_SAMPLES, Sample, brier, calibrate, ece, fit_temperature, render, samples_from,
)


def test_below_the_minimum_sample_no_temperature_is_written():
    rows = [Sample(f"s{i}", "retrieval", 0.9, True) for i in range(MIN_SAMPLES - 1)]
    rep = calibrate(rows)["retrieval"]
    assert rep.status == "insufficient" and rep.temperature is None
    assert "insufficient" in render({"retrieval": rep})


def test_overconfident_decisions_get_a_temperature_above_one():
    """p=0.9 claimed, right 60% of the time: the number should come down."""
    rng = random.Random(1)
    rows = [Sample(f"s{i}", "next", 0.9, rng.random() < 0.6) for i in range(200)]
    rep = calibrate(rows)["next"]
    assert rep.samples == 200
    assert rep.temperature > 1.0
    assert rep.status == "improved"
    assert rep.ece_after < rep.ece_before


def test_already_calibrated_decisions_are_left_unchanged():
    rng = random.Random(2)
    rows = []
    for i in range(300):
        p = rng.choice([0.6, 0.7, 0.8, 0.9])
        rows.append(Sample(f"s{i}", "target", p, rng.random() < p))
    rep = calibrate(rows)["target"]
    assert rep.status in ("unchanged", "improved")
    assert abs(rep.temperature - 1.0) < 0.6


def test_folds_are_grouped_by_session():
    """A session's decisions never straddle the split."""
    from mimir.eval.calibration import _folds

    rows = [Sample(f"s{i % 7}", "f", 0.5, True) for i in range(70)]
    for train, test in _folds(rows):
        assert not ({s.session_id for s in train} & {s.session_id for s in test})


def test_ece_is_zero_for_a_perfect_oracle_and_large_for_a_confident_liar():
    assert ece([(1.0, True)] * 10) == 0.0
    assert ece([(0.99, False)] * 10) > 0.9


def test_fit_temperature_recovers_one_for_calibrated_input():
    pairs = [(0.8, True)] * 8 + [(0.8, False)] * 2
    assert abs(fit_temperature(pairs) - 1.0) < 0.35


def test_samples_join_results_to_decision_logs_and_skip_unacted():
    results = [{"session_id": "a", "passed": True}, {"session_id": "b", "passed": False}]
    sessions = {
        "a": {"metadata": {"decisions": [{"field": "retrieval", "probability": 0.7, "acted": True}]}},
        "b": {"metadata": {"decisions": [{"field": "retrieval", "probability": 0.6, "acted": False},
                                          {"field": "next", "probability": 0.8, "acted": True}]}},
    }
    rows = samples_from(results, sessions)
    assert [(s.field, s.correct) for s in rows] == [("retrieval", True), ("next", False)]
