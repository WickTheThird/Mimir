"""Monitor tests.

The theme running through these: a dashboard is only worth having if a wrong
reading is impossible rather than unlikely. Two of these tests exist because the
first implementation got it wrong and said so on screen in red.
"""

from __future__ import annotations

import io
import sqlite3
import time

import pytest

from mimir.monitor import activity, machine, runtime
from mimir.monitor.dashboard import _bar, _bytes, _duration


class TestTagNormalisation:
    def test_bare_name_matches_implicit_latest(self):
        """The profile says nomic-embed-text, /api/tags says it with :latest.

        Comparing raw strings made the monitor print "not pulled" in red for a
        model that was loaded and answering requests.
        """
        assert runtime.normalise_tag("nomic-embed-text") == "nomic-embed-text:latest"

    def test_explicit_tag_is_untouched(self):
        assert runtime.normalise_tag("qwen2.5:32b") == "qwen2.5:32b"
        assert runtime.normalise_tag("") == ""


class TestProcessClassification:
    def test_a_shell_in_the_project_directory_is_not_a_running_mimir(self):
        """Substring matching caught the shell that happened to be sitting in
        the repository and reported it as an active investigation."""
        assert machine._classify("zsh", ["/bin/zsh", "-c", "cd /home/me/mimir"]) is None

    def test_entry_point_is_recognised_with_its_verb(self):
        assert machine._classify("mimir", ["/usr/bin/mimir", "evaluate"]) == "mimir evaluate"
        assert machine._classify("Python", ["python", "-m", "mimir.api"]) == "python -m mimir.api"

    def test_ollama_pull_is_labelled_with_its_target(self):
        assert machine._classify("ollama", ["ollama", "pull", "qwen2.5:32b"]) == (
            "ollama pull qwen2.5:32b"
        )


class TestUnavailableRatherThanFabricated:
    def test_a_reading_without_a_value_is_not_known(self):
        assert not machine.Reading(unavailable="psutil not installed").known
        assert machine.Reading(value=0.0).known, "zero is a measurement, not an absence"

    def test_no_recorded_warning_is_not_a_temperature(self):
        """pmset reporting nothing means nothing has been recorded. Rendering
        that as "nominal" would be a claim the output does not support."""
        state, _, _ = machine._thermal()
        assert state in ("no warning recorded", "throttled", "warning", "unavailable")
        assert state != "nominal"


class TestInFlightEvaluation:
    def test_progress_and_eta_from_completed_cases(self):
        flight = activity.InFlightEval(
            pid=1, started_at=time.time() - 600, completed=10, total=20, mean_case_s=50.0
        )
        assert flight.fraction == pytest.approx(0.5)
        assert flight.eta_s == pytest.approx(500.0)

    def test_no_eta_before_two_cases_have_finished(self):
        """One sample is not a pace. Extrapolating from it produces a confident
        estimate built on nothing, which is the failure mode this project
        treats as worse than saying nothing."""
        flight = activity.InFlightEval(pid=1, started_at=time.time(), completed=1, total=20)
        assert flight.eta_s is None

    def test_unknown_total_does_not_claim_a_fraction(self):
        flight = activity.InFlightEval(pid=1, started_at=time.time(), completed=7, total=0)
        assert flight.fraction == 0.0


class TestActivityStore:
    def test_missing_database_is_reported_not_raised(self, tmp_path):
        from mimir.config import Settings

        settings = Settings(home=tmp_path / "absent")
        act = activity.collect(settings)
        assert not act.readable
        assert act.error

    def test_the_store_is_opened_read_only(self, tmp_path):
        """The monitor watches work that is writing to this database. It must
        not be able to take a lock on it."""
        from mimir.config import Settings

        home = tmp_path / "home"
        home.mkdir()
        conn = sqlite3.connect(home / "mimir.db")
        conn.execute("create table sessions (id text, status text, task_type text, "
                     "interface text, title text, created_at real, updated_at real, "
                     "completed_at real, final_confidence real, error text)")
        conn.commit()
        conn.close()

        settings = Settings(home=home)
        assert activity.collect(settings).readable

        read_only = sqlite3.connect(f"file:{home / 'mimir.db'}?mode=ro", uri=True)
        with pytest.raises(sqlite3.OperationalError):
            read_only.execute("insert into sessions (id) values ('x')")
        read_only.close()

    def test_audit_gap_surfaces_a_silent_telemetry_hole(self):
        """Sessions accumulating while a telemetry table stays at zero is the
        exact shape of the executions bug: normal-looking activity, one number
        nobody was reading."""
        act = activity.Activity()
        act.sessions = [
            activity.SessionRow("s1", "completed", "t", "cli", "", 0.0, 0.0, 1.0, None, None)
        ]
        act.model_calls_total = 0
        assert "model_calls" in act.audit_gap

        act.model_calls_total = 5
        assert act.audit_gap == ""


class TestPullProgress:
    def test_sparse_preallocation_is_not_read_as_complete(self):
        """Ollama pre-allocates the blob at full size. Reporting apparent size
        shows 100% from the first second, which is how a stalled download looks
        finished."""
        pull = runtime.PullProgress(
            blob="sha256:abc",
            downloaded_bytes=9 << 30,
            total_bytes=18 << 30,
            modified_at=time.time(),
        )
        assert pull.fraction == pytest.approx(0.5)
        assert pull.stale_s < 5

    def test_a_stale_blob_reports_its_silence(self):
        pull = runtime.PullProgress("b", 1, 2, modified_at=time.time() - 7200)
        assert pull.stale_s > 3600


class TestFormatting:
    def test_bytes_and_durations(self):
        assert _bytes(6.1 * (1 << 30)).endswith("GB")
        assert _bytes(353 * (1 << 20)) == "353 MB"
        assert _duration(45) == "45s"
        assert _duration(125) == "2m05s"
        assert _duration(7300) == "2h01m"

    def test_bar_is_clamped(self):
        assert len(_bar(5.0, width=10).plain) == 10
        assert len(_bar(-1.0, width=10).plain) == 10


class TestTelemetryInvariant:
    """model invocations observed == model-call records persisted."""

    def test_complete_when_counts_agree(self):
        from mimir.eval.harness import EvalReport

        report = EvalReport()
        report.model_invocations = 12
        report.model_calls_persisted = 12
        assert report.telemetry_complete

    def test_incomplete_does_not_fail_safety_acceptance(self):
        """Missing telemetry is experimental debt, not evidence of an
        unobserved mutation, so it must not block an otherwise safe run."""
        from mimir.eval.harness import EvalReport

        report = EvalReport()
        report.model_invocations = 12
        report.model_calls_persisted = 0
        assert not report.telemetry_complete
        assert report.acceptable, "safety acceptance is a separate question"

    def test_invocation_id_is_not_derived_from_time(self):
        """Two calls can start in the same float tick. Keying on a timestamp
        makes duplicate telemetry indistinguishable from a genuine retry."""
        from mimir.llm.base import ModelCallRecord

        common = {
            "alias": "deep", "model": "m", "started_at": 1234.5, "latency_s": 1.0,
            "prompt_tokens": 1, "completion_tokens": 1, "tool_calls": 0,
            "finish_reason": "stop",
        }
        assert ModelCallRecord(**common).invocation_id != ModelCallRecord(**common).invocation_id


class TestReportMerging:
    def test_absorb_carries_the_contamination_verdict(self):
        """Extending results alone dropped every report-level field. A model run
        that tripped containment was persisted as clean, because the
        deterministic report's defaults look identical to a clean result."""
        from mimir.eval.harness import EvalReport

        combined = EvalReport(label="deterministic")
        model = EvalReport(label="model")
        model.external_calls = 3
        model.blocked_hosts = ["google.com"]
        model.contaminated_reason = "offline run attempted external access"
        model.enabled_tools_hash = "abc123"
        model.enabled_capabilities = ["repository"]

        combined.absorb(model)
        assert combined.contaminated_reason
        assert combined.external_calls == 3
        assert combined.blocked_hosts == ["google.com"]
        assert combined.enabled_tools_hash == "abc123"
        assert not combined.acceptable


class TestCorpusLoading:
    def test_a_malformed_file_raises_rather_than_shrinking_the_corpus(self, tmp_path):
        """A bad indent once dropped thirteen cases and the run reported a
        perfect score against a denominator nobody chose."""
        from mimir.eval.harness import CorpusError, EvalHarness

        (tmp_path / "broken.yaml").write_text("cases:\n  - id: a\n- id: b\n")
        with pytest.raises(CorpusError):
            EvalHarness.load_corpus(tmp_path)

    def test_a_single_malformed_case_is_still_skipped(self, tmp_path):
        """Losing one case loudly beats losing all of them."""
        from mimir.eval.harness import EvalHarness

        (tmp_path / "mixed.yaml").write_text(
            "cases:\n"
            "  - id: good\n    kind: dangerous_command\n    prompt: list files\n"
            "    argv: [ls]\n"
            "  - id: bad\n    kind: not_a_real_kind\n    prompt: x\n    argv: [ls]\n"
        )
        cases = EvalHarness.load_corpus(tmp_path)
        assert [c.id for c in cases] == ["good"]


class TestTelemetryPanel:
    def test_history_is_not_reported_as_a_present_fault(self):
        """Sessions older than the first telemetry row predate the
        instrumentation, so their lack of model calls is expected.

        Counting them showed "17/20 recent sessions recorded no model calls" at
        a moment when every session since the fix was instrumented correctly.
        """
        health = activity.TelemetryHealth(
            sessions_checked=3, sessions_without_calls=0,
            model_calls_total=43, instrumented_since=1000.0,
        )
        assert health.complete
        assert "3/3" in health.summary

    def test_a_genuine_gap_is_still_reported(self):
        act = activity.Activity()
        act.sessions = [
            activity.SessionRow("s1", "completed", "t", "eval", "", 0.0, 0.0, 1.0, None, None)
        ]
        act.model_calls_total = 10
        act.telemetry = activity.TelemetryHealth(
            sessions_checked=5, sessions_without_calls=2, model_calls_total=10
        )
        assert "2/5" in act.audit_gap

    def test_orphaned_rows_are_surfaced(self):
        act = activity.Activity()
        act.sessions = [
            activity.SessionRow("s1", "completed", "t", "eval", "", 0.0, 0.0, 1.0, None, None)
        ]
        act.model_calls_total = 10
        act.telemetry = activity.TelemetryHealth(
            sessions_checked=5, sessions_without_calls=0, model_calls_total=10,
            orphaned_rows=3,
        )
        assert "orphan" in act.audit_gap or "no session" in act.audit_gap

    def test_role_means_are_per_call_not_totals(self):
        role = activity.RoleTelemetry(
            role="deep_investigation", model="qwen2.5:7b", calls=4, total_latency_ms=8000.0
        )
        assert role.mean_latency_ms == pytest.approx(2000.0)

    def test_no_calls_means_no_division(self):
        assert activity.RoleTelemetry(role="x", model="y").mean_latency_ms == 0.0


def _rendered(renderable) -> str:
    """Render to plain text.

    str() of a Rich renderable is its repr, not its content, so asserting
    against it passes or fails for reasons unrelated to what is displayed.
    """
    from rich.console import Console

    console = Console(width=120, record=True, file=io.StringIO())
    console.print(renderable)
    return console.export_text()


class TestEvaluationPanelSchemas:
    """The panel must read v2 runs and still display v1 runs honestly."""

    def _run(self, metadata):
        return activity.EvalRunRow(
            id="eval_x", suite="regression", model_alias="", total=52, passed=47,
            failed=5, created_at=0.0, completed_at=1.0, metadata=metadata,
        )

    def test_v2_contamination_is_read_from_the_nested_shape(self):
        from mimir.monitor.dashboard import render_evaluation

        act = activity.Activity()
        act.latest_run = self._run({
            "provenance": {
                "schema_version": 2,
                "evaluation": {"contaminated": True, "contaminated_reason": "web enabled",
                               "enabled_tools_hash": "abc"},
                "source": {"commit": "abc123"},
            }
        })
        assert "CONTAMINATED" in _rendered(render_evaluation(act))

    def test_missing_tool_fingerprint_is_shown_as_not_comparable(self):
        """An empty fingerprint is not a match with another empty fingerprint."""
        from mimir.monitor.dashboard import render_evaluation

        act = activity.Activity()
        act.latest_run = self._run({
            "provenance": {
                "schema_version": 2,
                "evaluation": {"enabled_tools_hash": "", "offline": True},
                "source": {"commit": "abc123"},
            }
        })
        assert "not comparable" in _rendered(render_evaluation(act))

    def test_source_changing_mid_run_is_flagged(self):
        from mimir.monitor.dashboard import render_evaluation

        act = activity.Activity()
        act.latest_run = self._run({
            "provenance": {
                "schema_version": 2,
                "evaluation": {"enabled_tools_hash": "abc", "offline": True},
                "source": {"commit": "abc123", "changed_during_run": True},
            }
        })
        assert "SOURCE CHANGED" in _rendered(render_evaluation(act))

    def test_incomplete_telemetry_marks_efficiency_invalid_but_not_quality(self):
        from mimir.monitor.dashboard import render_evaluation

        act = activity.Activity()
        act.latest_run = self._run({
            "model_invocations_observed": 12,
            "model_calls_persisted": 0,
            "telemetry_complete": False,
            "valid_for_quality_reporting": True,
            "valid_for_efficiency_comparison": False,
            "provenance": {
                "schema_version": 2,
                "evaluation": {"enabled_tools_hash": "abc", "offline": True},
                "source": {"commit": "abc123"},
            },
        })
        text = _rendered(render_evaluation(act))
        assert "INCOMPLETE" in text
        assert "quality yes" in text.replace("  ", " ")


class TestSeriesStatistics:
    """Repeats of one experiment. Errors here would be silent and believed."""

    def _series(self, *runs):
        s = activity.Series(corpus_hash="c", commit="abc")
        for i, results in enumerate(runs, start=1):
            s.runs.append(
                activity.SeriesRun(
                    run_id=f"eval_{i}", label=f"#{i}", model="qwen2.5:7b",
                    passed=sum(1 for v in results.values() if v),
                    total=len(results), created_at=float(i), results=results,
                )
            )
        return s

    def test_spread_and_mean_describe_what_was_seen(self):
        s = self._series(
            {"a": True, "b": True, "c": True},
            {"a": True, "b": False, "c": True},
            {"a": False, "b": False, "c": True},
        )
        assert s.counts == [3, 2, 1]
        assert s.mean == pytest.approx(2.0)
        assert s.spread == 2

    def test_a_single_run_has_no_spread_to_report(self):
        s = self._series({"a": True})
        assert s.spread == 0
        assert s.stdev == 0.0
        assert s.stability_rate() is None, "one run cannot establish stability"

    def test_only_disagreeing_cases_are_listed(self):
        s = self._series(
            {"stable_pass": True, "flips": True, "stable_fail": False},
            {"stable_pass": True, "flips": False, "stable_fail": False},
            {"stable_pass": True, "flips": True, "stable_fail": False},
        )
        unstable = s.unstable_cases()
        assert [c[0] for c in unstable] == ["flips"]

    def test_majority_and_agreement(self):
        s = self._series(
            {"x": True}, {"x": False}, {"x": True},
        )
        case_id, outcomes, majority, agreement = s.unstable_cases()[0]
        assert case_id == "x"
        assert outcomes == [True, False, True]
        assert majority is True
        assert agreement == pytest.approx(2 / 3)

    def test_a_tie_resolves_to_pass_and_is_reported_as_half(self):
        """With an even number of runs a tie is not a majority. It is reported
        at 50% agreement so nobody reads it as a settled outcome."""
        s = self._series({"x": True}, {"x": False})
        _, _, majority, agreement = s.unstable_cases()[0]
        assert majority is True
        assert agreement == pytest.approx(0.5)

    def test_stability_rate_counts_fully_agreeing_cases(self):
        s = self._series(
            {"a": True, "b": True, "c": False},
            {"a": True, "b": False, "c": False},
            {"a": True, "b": True, "c": False},
        )
        assert s.stability_rate() == pytest.approx(2 / 3)


class TestLiveCases:
    def test_no_pass_or_fail_is_claimed_before_scoring(self):
        """Scoring happens in process and is not written until the run ends.
        A verdict shown here would be invented."""
        case = activity.LiveCase(
            case_id="inv-001", session_id="s", prompt="p", task_type="t",
            confidence=0.35, evidence=7, tool_calls=2, duration_s=19.0,
        )
        assert not hasattr(case, "passed")

    def test_a_running_case_is_distinguished_from_a_finished_one(self):
        running = activity.LiveCase(
            case_id="x", session_id="s", prompt="p", task_type="t",
            confidence=None, evidence=0, tool_calls=0, duration_s=3.0, running=True,
        )
        assert running.running and running.confidence is None


class TestCouncilGraph:
    """The graph is drawn from telemetry. Structure from code, weights from data."""

    def test_labels_are_abbreviated_not_truncated(self):
        """A chopped word reads as a bug; an abbreviation reads as deliberate."""
        from mimir.monitor.dashboard import _SPECIALIST_SHORT

        for full, short in _SPECIALIST_SHORT.items():
            assert len(short) <= 11, f"{full} -> {short} will be cut off"

    def _council(self, *nodes):
        c = activity.Council()
        c.nodes = list(nodes)
        return c

    def _node(self, name, **kw):
        return activity.CouncilNode(specialist=name, **kw)

    def test_entry_and_exit_are_identified(self):
        c = self._council(
            self._node("coordinator", calls=10),
            self._node("log_analyst", calls=5),
            self._node("synthesis", calls=10),
        )
        assert c.entry.specialist == "coordinator"
        assert c.exit.specialist == "synthesis"
        assert [n.specialist for n in c.workers] == ["log_analyst"]

    def test_workers_are_ordered_by_measured_cost_not_by_name(self):
        c = self._council(
            self._node("aaa_cheap", total_latency_ms=100.0),
            self._node("zzz_expensive", total_latency_ms=9000.0),
        )
        assert [n.specialist for n in c.workers] == ["zzz_expensive", "aaa_cheap"]

    def test_mean_latency_never_divides_by_zero(self):
        assert self._node("x").mean_latency_ms == 0.0

    def test_total_latency_is_never_zero_so_shares_are_safe(self):
        """Share of time divides by this. A council with no recorded latency
        would otherwise raise while rendering."""
        assert self._council(self._node("x")).total_latency_ms == 1.0

    def test_active_nodes_are_reported(self):
        c = self._council(
            self._node("coordinator", active=False),
            self._node("log_analyst", active=True),
        )
        assert c.active_names == ["log_analyst"]

    def test_an_empty_council_renders_without_claiming_anything(self):
        from mimir.monitor.dashboard import render_council

        act = activity.Activity()
        text = _rendered(render_council(act))
        assert "no model calls" in text
        assert "drawn from telemetry" in text

    def test_the_panel_shows_measured_edges_not_invented_ones(self):
        from mimir.monitor.dashboard import render_council

        act = activity.Activity()
        act.council = self._council(
            self._node("coordinator", calls=43, total_latency_ms=215000.0),
            self._node("kubernetes_investigator", calls=88, total_latency_ms=721000.0,
                       tool_calls=72),
            self._node("synthesis", calls=44, total_latency_ms=602000.0),
        )
        text = _rendered(render_council(act))
        assert "coordinator" in text and "k8s" in text and "synthesis" in text
        assert "72t" in text, "tool-call weight must be shown"
        assert "43x" in text, "call count must be shown"


class TestTrends:
    """Motion must encode information, or it is an animation pretending to be
    a status."""

    def test_one_sample_is_not_a_trend(self):
        from mimir.monitor.dashboard import _sparkline

        assert _sparkline([]).plain == "collecting"
        assert _sparkline([42.0]).plain == "collecting"

    def test_a_flat_series_draws_flat(self):
        """Scaling to the observed range means a flat line reads as genuinely
        flat, rather than being stretched to look like variation."""
        from mimir.monitor.dashboard import _sparkline

        assert set(_sparkline([5.0] * 6).plain) == {"▁"}

    def test_extremes_map_to_the_ends_of_the_ramp(self):
        from mimir.monitor.dashboard import _sparkline

        drawn = _sparkline([0.0, 50.0, 100.0]).plain
        assert drawn[0] == "▁" and drawn[-1] == "█"

    def test_only_the_most_recent_samples_are_drawn(self):
        from mimir.monitor.dashboard import _sparkline

        assert len(_sparkline(list(range(100)), width=20).plain) == 20

    def test_the_pulse_is_static_when_nothing_is_active(self):
        """A spinner turning over an idle system is a lie about liveness."""
        from mimir.monitor.dashboard import _pulse

        assert _pulse(False).plain == "·"
        assert _pulse(True).plain != "·"

    def test_history_is_bounded(self):
        from mimir.monitor.dashboard import _record

        for i in range(500):
            series = _record("test_metric", float(i), keep=10)
        assert len(series) == 10
        assert series[-1] == 499.0


class TestPersistedMetadataShape:
    """Assert on what reaches the database, not on what the code appears to do.

    Two fields were silently missing from every stored run: `enabled_tools` was
    dropped by the merge, and the telemetry keys were never written at all
    because a blind string replacement did not match its anchor. Both looked
    correct in the source. Neither had a test asserting the persisted shape, so
    both reached the database, and the only reason they were caught is that
    someone read a stored record back.
    """

    REQUIRED = (
        "unapproved_mutations",
        "dangerous_proposals",
        "unsupported_claim_rate",
        "enabled_tools_hash",
        "enabled_capabilities",
        "external_calls",
        "contaminated",
        "model_invocations_observed",
        "model_calls_persisted",
        "telemetry_complete",
        "valid_for_quality_reporting",
        "valid_for_efficiency_comparison",
        "provenance",
    )

    def test_every_field_a_reader_needs_is_written(self, monkeypatch):
        from mimir.eval.harness import EvalHarness, EvalReport

        captured = {}

        class FakeRepo:
            def create_run(self, run_id, **kwargs):
                captured.update(kwargs.get("metadata") or {})

            def record_result(self, *a, **k):
                pass

            def complete_run(self, *a, **k):
                pass

        import mimir.persistence.repositories as repos

        monkeypatch.setattr(repos, "EvalRepository", lambda *a, **k: FakeRepo())

        report = EvalReport(label="test")
        report.model_invocations = 7
        report.model_calls_persisted = 7
        EvalHarness().persist(report, suite="test")

        missing = [f for f in self.REQUIRED if f not in captured]
        assert not missing, f"missing from persisted metadata: {missing}"
        assert captured["telemetry_complete"] is True
        assert captured["valid_for_efficiency_comparison"] is True

    def test_absorb_carries_the_tool_names_not_only_the_hash(self):
        """The hash proves two runs used the same tools; the names say which.
        A stored run with a hash and an empty name list cannot be audited."""
        from mimir.eval.harness import EvalReport

        combined = EvalReport(label="deterministic")
        model = EvalReport(label="model")
        model.enabled_tools_hash = "abc123"
        model.enabled_capabilities = ["repository"]
        model.enabled_tools = ["search_repository", "read_file"]

        combined.absorb(model)
        assert combined.enabled_tools == ["search_repository", "read_file"]

    def test_telemetry_counts_survive_the_merge(self):
        from mimir.eval.harness import EvalReport

        combined = EvalReport()
        model = EvalReport()
        model.model_invocations = 12
        model.model_calls_persisted = 12
        combined.absorb(model)
        assert combined.model_invocations == 12
        assert combined.telemetry_complete


class TestPassAtK:
    """pass@k answers "is the capability there"; pass^k answers "can it be
    trusted". Reporting only one of them, or only a mean, hides the gap that
    matters."""

    def _series(self, *runs, deterministic=()):
        s = activity.Series(corpus_hash="c", commit="abc")
        for i, results in enumerate(runs, start=1):
            s.runs.append(
                activity.SeriesRun(
                    run_id=f"eval_{i}", label=f"#{i}", model="m",
                    passed=sum(1 for v in results.values() if v),
                    total=len(results), created_at=float(i), results=results,
                    deterministic=set(deterministic),
                )
            )
        return s

    def test_a_case_solved_once_counts_for_at_k_but_not_hat_k(self):
        s = self._series({"a": True}, {"a": False}, {"a": True})
        assert s.pass_at_k() == (1, 1)
        assert s.pass_hat_k() == (0, 1)

    def test_a_case_never_solved_counts_for_neither(self):
        s = self._series({"a": False}, {"a": False}, {"a": False})
        assert s.pass_at_k() == (0, 1)
        assert s.pass_hat_k() == (0, 1)

    def test_deterministic_cases_are_excluded_by_default(self):
        """They are always stable, so including them only drags the
        reliability figure toward 100% and hides the model's behaviour."""
        s = self._series(
            {"model": True, "det": True},
            {"model": False, "det": True},
            {"model": True, "det": True},
            deterministic=("det",),
        )
        assert s.pass_hat_k() == (0, 1), "only the model case is counted"
        assert s.pass_hat_k(model_only=False) == (1, 2), "det case is stable"

    def test_no_runs_reports_nothing_rather_than_zero(self):
        s = activity.Series()
        assert s.pass_at_k() is None
        assert s.pass_hat_k() is None

    def test_the_gap_is_what_distinguishes_capability_from_reliability(self):
        """18/21 solvable but 11/21 reliable is a routing and procedure
        problem, not a knowledge problem."""
        runs = [{f"c{i}": True for i in range(11)} for _ in range(3)]
        for i in range(11, 18):          # 7 unstable
            runs[0][f"c{i}"] = True
            runs[1][f"c{i}"] = False
            runs[2][f"c{i}"] = True
        for i in range(18, 21):          # 3 never solved
            for r in runs:
                r[f"c{i}"] = False
        s = self._series(*runs)
        assert s.pass_at_k() == (18, 21)
        assert s.pass_hat_k() == (11, 21)
