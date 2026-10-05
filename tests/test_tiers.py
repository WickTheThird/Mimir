"""The tier table must report the same corpus the harness ran."""

import json

from mimir.eval.tiers import (
    Tier,
    load,
    pair_table,
    render,
    shared_failures,
    stable_failures,
)


def _write(tmp_path, name, rows, **payload):
    path = tmp_path / name
    path.write_text(json.dumps({"results": rows, **payload}))
    return path


def test_pending_cases_are_excluded_rather_than_counted_as_passes(tmp_path):
    """A pending case leaves ``passed`` False but never ran."""
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


def test_pair_table_marks_a_pair_split_unless_both_sides_pass(tmp_path):
    rows = [
        {"case_id": "p-a", "passed": True, "pending": "", "pair": "p"},
        {"case_id": "p-b", "passed": False, "pending": "", "pair": "p"},
        {"case_id": "q-a", "passed": True, "pending": "", "pair": "q"},
        {"case_id": "q-b", "passed": True, "pending": "", "pair": "q"},
    ]
    tier = load(_write(tmp_path, "t.json", rows), "laptop", set())
    text = pair_table([tier])
    assert "p" in text and "split" in text
    assert "both" in text


def test_a_half_tested_pair_reads_not_applicable_rather_than_split(tmp_path):
    """One side missing is an untested pair, not a failed one."""
    rows = [{"case_id": "p-a", "passed": True, "pending": "", "pair": "p"}]
    tier = load(_write(tmp_path, "t.json", rows), "laptop", set())
    assert "n/a" in pair_table([tier])


def test_shared_failures_counts_agreement_over_cases_every_tier_ran(tmp_path):
    a = load(
        _write(
            tmp_path,
            "a.json",
            [
                {"case_id": "1", "passed": True, "pending": ""},
                {"case_id": "2", "passed": False, "pending": ""},
                {"case_id": "3", "passed": True, "pending": ""},
            ],
        ),
        "laptop",
        set(),
    )
    b = load(
        _write(
            tmp_path,
            "b.json",
            [
                {"case_id": "1", "passed": True, "pending": ""},
                {"case_id": "2", "passed": False, "pending": ""},
                {"case_id": "3", "passed": False, "pending": ""},
            ],
        ),
        "mini",
        set(),
    )
    agreeing, compared, failed_everywhere = shared_failures([a, b])
    assert (agreeing, compared) == (2, 3)
    assert failed_everywhere == ["2"]


def test_shared_failures_ignores_cases_one_tier_never_ran(tmp_path):
    a = load(
        _write(tmp_path, "a.json", [{"case_id": "1", "passed": False, "pending": ""}]),
        "laptop",
        set(),
    )
    b = load(
        _write(
            tmp_path,
            "b.json",
            [
                {"case_id": "1", "passed": False, "pending": ""},
                {"case_id": "2", "passed": False, "pending": ""},
            ],
        ),
        "mini",
        set(),
    )
    assert shared_failures([a, b]) == (1, 1, ["1"])


def _run(tmp_path, name, verdicts):
    rows = [
        {"case_id": c, "passed": v, "pending": ""} for c, v in verdicts.items()
    ]
    return load(_write(tmp_path, f"{name}.json", rows), name, set())


def test_stable_failures_caveats_when_there_are_too_few_runs(tmp_path):
    """Two runs called three cases structural that a third run passed."""
    a = _run(tmp_path, "a", {"1": False, "2": False, "3": False, "4": False, "5": False})
    b = _run(tmp_path, "b", {"1": False, "2": False, "3": False, "4": False, "5": False})
    failing, caveat = stable_failures([a, b])
    assert failing == ["1", "2", "3", "4", "5"]
    assert caveat is not None
    assert "2 run(s)" in caveat
    assert "structural" in caveat


def test_stable_failures_is_uncaveated_once_the_replicates_are_there(tmp_path):
    runs = [_run(tmp_path, n, {"1": False, "2": True}) for n in ("a", "b", "c")]
    failing, caveat = stable_failures(runs)
    assert failing == ["1"]
    assert caveat is None


def test_a_case_that_passes_in_any_run_is_not_a_stable_failure(tmp_path):
    runs = [
        _run(tmp_path, "a", {"1": False}),
        _run(tmp_path, "b", {"1": False}),
        _run(tmp_path, "c", {"1": True}),
    ]
    assert stable_failures(runs)[0] == []


def test_stable_failures_only_considers_cases_every_run_attempted(tmp_path):
    runs = [
        _run(tmp_path, "a", {"1": False}),
        _run(tmp_path, "b", {"1": False, "2": False}),
        _run(tmp_path, "c", {"1": False, "2": False}),
    ]
    assert stable_failures(runs)[0] == ["1"]


def test_no_runs_is_a_caveat_rather_than_an_empty_finding(tmp_path):
    assert stable_failures([])[1] is not None
