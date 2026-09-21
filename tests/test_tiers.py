"""The tier table must report the same corpus the harness ran."""

import json

from mimir.eval.tiers import Tier, load, render


def _write(tmp_path, name, rows, **payload):
    path = tmp_path / name
    path.write_text(json.dumps({"results": rows, **payload}))
    return path


def test_pending_cases_are_excluded_rather_than_counted_as_passes(tmp_path):
    """A pending case leaves ``passed`` False but never ran.

    Counting it gave the tier a case it did not attempt. Both tiers got the
    same phantom rows, so the comparison looked consistent while every
    reported total was larger than the run.
    """
    rows = [
        {"case_id": "a", "passed": True, "pending": ""},
        {"case_id": "b", "passed": False, "pending": ""},
        {"case_id": "c", "passed": False, "pending": "decided, not built"},
    ]
    tier = load(_write(tmp_path, "t.json", rows), "laptop", set())
    assert tier.total == 2
    assert tier.passed == 1
    assert tier.pending == 1


def test_pending_exclusion_is_stated_not_silent(tmp_path):
    rows = [
        {"case_id": "a", "passed": True, "pending": ""},
        {"case_id": "b", "passed": False, "pending": "decided, not built"},
    ]
    tier = load(_write(tmp_path, "t.json", rows), "laptop", set())
    text = render([tier])
    assert "pending" in text
    assert "laptop 1" in text


def test_the_resolved_model_is_preferred_over_the_routing_alias(tmp_path):
    """Two tiers can share an alias. They cannot share a model string."""
    path = _write(
        tmp_path,
        "t.json",
        [{"case_id": "a", "passed": True, "pending": ""}],
        model_alias="deep",
        model="qwen2.5:7b",
    )
    assert load(path, "mini", set()).model == "qwen2.5:7b"


def test_a_report_with_no_model_says_so_instead_of_reading_blank(tmp_path):
    """An unnamed tier is a defect to surface, not an empty column to skim."""
    path = _write(tmp_path, "t.json", [{"case_id": "a", "passed": True, "pending": ""}])
    tier = load(path, "mini", set())
    assert tier.model == "unrecorded"
    assert "unrecorded" in render([tier])


def test_deterministic_cases_are_split_out_of_the_model_rate(tmp_path):
    rows = [
        {"case_id": "det", "passed": True, "pending": ""},
        {"case_id": "m1", "passed": True, "pending": ""},
        {"case_id": "m2", "passed": False, "pending": ""},
    ]
    tier = load(_write(tmp_path, "t.json", rows), "laptop", {"det"})
    assert tier.rate == round(2 / 3, 3)
    assert tier.model_total == 2
    assert tier.model_rate == 0.5
