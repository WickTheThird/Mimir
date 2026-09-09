"""The working set.

The store holds everything ever learned and the index retrieves any of it.
Neither knows what is currently relevant, so these cover the layer that does.
"""

from __future__ import annotations

import time

import pytest

from mimir.knowledge.bank import ACTIVE_THRESHOLD, HALF_LIFE_S, MemoryBank, _activation


@pytest.fixture
def bank(tmp_path):
    return MemoryBank(tmp_path / "bank.db")


class TestActivation:
    def test_use_raises_it(self, bank):
        bank.touch(["a"], query="x")
        first = bank.working_set()[0].activation
        bank.touch(["a"], query="x")
        assert bank.working_set()[0].activation > first

    def test_time_lowers_it(self):
        now = time.time()
        assert _activation(4, now - HALF_LIFE_S, now) == pytest.approx(2.0, rel=0.01)
        assert _activation(4, now - 2 * HALF_LIFE_S, now) == pytest.approx(1.0, rel=0.01)

    def test_a_single_accidental_recall_decays_out_on_its_own(self):
        """Otherwise anything ever touched stays in mind for good."""
        now = time.time()
        assert _activation(1, now - 3 * HALF_LIFE_S, now) < ACTIVE_THRESHOLD

    def test_repeated_use_survives_much_longer(self):
        now = time.time()
        assert _activation(8, now - 3 * HALF_LIFE_S, now) >= ACTIVE_THRESHOLD

    def test_many_chunks_of_one_note_are_one_recall(self, bank):
        """A search returning six chunks of the same document has recalled one
        thing, and counting six would make long notes permanent."""
        bank.touch(["a", "a", "a"], query="x")
        assert bank.working_set()[0].hits == 1


class TestForgetting:
    def test_what_decayed_leaves_the_working_set(self, bank):
        bank.touch(["old"], query="x")
        bank._db.execute(
            "UPDATE activations SET last_recall = ?", (time.time() - 5 * HALF_LIFE_S,)
        )
        bank._db.commit()
        assert bank.forget() == 1
        assert bank.working_set() == []

    def test_forgetting_drops_the_activation_not_the_note(self, bank, tmp_path):
        """Losing the note that something was once relevant is the point.
        Losing the note itself would be destruction."""
        bank.touch(["imports/x"], query="q")
        bank._db.execute("UPDATE activations SET last_recall = 0")
        bank._db.commit()
        bank.forget()
        assert bank._db.execute("SELECT COUNT(*) FROM work").fetchone()[0] == 0
        # the store is untouched: the bank has no write path into it
        assert not hasattr(bank, "store")

    def test_a_fresh_recall_is_kept(self, bank):
        bank.touch(["new"], query="x")
        assert bank.forget() == 0


class TestWorkingSet:
    def test_it_is_ordered_by_activation(self, bank):
        bank.touch(["weak"], query="x")
        for _ in range(4):
            bank.touch(["strong"], query="x")
        assert next(r.doc_id for r in bank.working_set()) == "strong"

    def test_it_is_capped(self, bank):
        from mimir.knowledge.bank import MAX_WORKING_SET

        for i in range(MAX_WORKING_SET + 20):
            bank.touch([f"doc-{i}"], query="x")
        assert len(bank.working_set()) == MAX_WORKING_SET


class TestLedger:
    def test_it_groups_what_was_done_by_project(self, bank):
        bank._db.executemany(
            "INSERT INTO work (doc_id, project, kind, title, happened, source, stale) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                ("a", "messaging-whatsapp", "project", "deploy", "2026-09-02", "s", 1),
                ("b", "messaging-whatsapp", "project", "billing", "2026-03-10", "s", 1),
                ("c", "PERS-mimir", "project", "thesis", "2026-08-08", "s", 0),
            ],
        )
        bank._db.commit()
        projects = bank.projects()
        assert projects[0]["project"] == "messaging-whatsapp"
        assert projects[0]["notes"] == 2
        assert projects[0]["latest"] == "2026-09-02"
        assert projects[0]["unverified"] == 2

    def test_asking_about_one_returns_it_newest_first(self, bank):
        bank._db.executemany(
            "INSERT INTO work (doc_id, project, kind, title, happened) VALUES (?,?,?,?,?)",
            [("a", "p", "k", "older", "2026-01-01"), ("b", "p", "k", "newer", "2026-06-01")],
        )
        bank._db.commit()
        assert [r["title"] for r in bank.about("p")] == ["newer", "older"]
