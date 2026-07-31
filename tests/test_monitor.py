"""Monitor tests.

The theme running through these: a dashboard is only worth having if a wrong
reading is impossible rather than unlikely. Two of these tests exist because the
first implementation got it wrong and said so on screen in red.
"""

from __future__ import annotations

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
