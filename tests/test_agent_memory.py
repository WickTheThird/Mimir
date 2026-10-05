"""Curated agent memory import."""

from __future__ import annotations

import pytest

from mimir.knowledge.agent_memory import (
    AgentMemoryImporter,
    _project_of,
    discover_memory_files,
    parse_memory_file,
)
from mimir.knowledge.store import Confidence, KnowledgeStore, VerificationStatus


@pytest.fixture
def store(tmp_path):
    store = KnowledgeStore(root=tmp_path / "knowledge")
    store.ensure_layout()
    return store


def _memory(tmp_path, name, body, frontmatter=True):
    path = tmp_path / name
    head = (
        "---\nname: a-slug\ndescription: what this note says\n"
        "metadata:\n  type: project\n---\n"
        if frontmatter
        else ""
    )
    path.write_text(head + body)
    return path


class TestParsing:
    def test_the_description_becomes_the_title(self, tmp_path):
        memory = parse_memory_file(_memory(tmp_path, "a.md", "The fact itself."))
        assert memory is not None
        assert memory.title == "what this note says"
        assert memory.kind == "project"
        assert memory.body == "The fact itself."

    def test_a_file_without_frontmatter_falls_back_to_its_heading(self, tmp_path):
        memory = parse_memory_file(
            _memory(tmp_path, "b.md", "# Deploy mechanics\n\nTwo steps.", frontmatter=False)
        )
        assert memory is not None
        assert memory.title == "Deploy mechanics"

    def test_an_empty_file_is_not_a_memory(self, tmp_path):
        assert parse_memory_file(_memory(tmp_path, "c.md", "   ", frontmatter=False)) is None

    def test_something_too_large_to_be_one_fact_is_refused(self, tmp_path):
        """A manual that happens to live in a memory directory would dominate the layer meant for single facts."""
        path = _memory(tmp_path, "d.md", "x" * 70_000, frontmatter=False)
        assert parse_memory_file(path) is None

    def test_the_project_keeps_the_half_that_identifies_it(self):
        """Taking the last hyphenated word of the flattened path gives "whatsapp", which is the half that does not identify anything."""
        from pathlib import Path

        path = Path(
            "/Users/x/.claude/projects/-Users-filipb-Documents-messaging-whatsapp/memory/a.md"
        )
        assert _project_of(path) == "messaging-whatsapp"
        assert _project_of(
            Path("/h/.claude/projects/-Users-filipb-Documents-PERS-mimir/memory/a.md")
        ) == "PERS-mimir"


class TestImport:
    def test_it_lands_in_imports_as_unverified(self, tmp_path, store):
        """Curated or not, nothing reaches stable memory without review."""
        importer = AgentMemoryImporter(store)
        doc_id = importer.write_note(parse_memory_file(_memory(tmp_path, "a.md", "A fact.")))

        assert doc_id.startswith("imports/agent-memory/")
        document = store.get(doc_id)
        assert document is not None
        assert document.metadata.verification_status is VerificationStatus.UNVERIFIED

    def test_it_outranks_a_transcript_summary_and_nothing_else(self, tmp_path, store):
        """A transcript summary is an extract a program made."""
        importer = AgentMemoryImporter(store)
        doc_id = importer.write_note(parse_memory_file(_memory(tmp_path, "a.md", "A fact.")))
        assert store.get(doc_id).metadata.confidence is Confidence.MEDIUM

    def test_a_secret_in_a_title_is_redacted_like_one_in_a_body(self, tmp_path, store):
        """A title is what search returns before anyone opens the note, so a secret there is more exposed than one in a body, not less."""
        path = tmp_path / "creds.md"
        path.write_text(
            "---\ndescription: token is ghp_aBcD1234567890aBcD1234567890aBcDef\n---\n"
            "body mentions ghp_aBcD1234567890aBcD1234567890aBcDef too\n"
        )
        doc_id = AgentMemoryImporter(store).write_note(parse_memory_file(path))
        document = store.get(doc_id)
        assert "ghp_aBcD1234567890aBcD1234567890aBcDef" not in document.metadata.title
        assert "ghp_aBcD1234567890aBcD1234567890aBcDef" not in document.body

    def test_an_index_file_is_not_imported(self, tmp_path):
        """It lists the other notes, so importing it duplicates every fact as a one-line stub that then competes with the real note in retrieval."""
        (tmp_path / "memory").mkdir()
        for name in ("MEMORY.md", "real-note.md"):
            (tmp_path / "memory" / name).write_text("# x\n\nbody\n")
        # discover_memory_files filters by name, so check the filter itself.
        from mimir.knowledge.agent_memory import _INDEX_NAMES

        assert "memory.md" in _INDEX_NAMES

    def test_a_run_reports_what_it_skipped(self, tmp_path, store):
        good = _memory(tmp_path, "a.md", "A fact.")
        empty = _memory(tmp_path, "b.md", "", frontmatter=False)
        result = AgentMemoryImporter(store).run([good, empty])
        assert result.files_seen == 2
        assert result.notes_written == 1
        assert result.skipped == 1
        assert not result.errors


class TestDiscovery:
    def test_it_looks_where_these_tools_actually_keep_memory(self, tmp_path):
        project = tmp_path / ".claude" / "projects" / "-Users-a-Documents-x" / "memory"
        project.mkdir(parents=True)
        (project / "note.md").write_text("# x\n\nbody\n")
        (project / "MEMORY.md").write_text("# index\n\n- [x](note.md)\n")
        (tmp_path / ".claude" / "CLAUDE.md").write_text("# global\n\nprefs\n")

        found = discover_memory_files(tmp_path)
        names = {p.name for p in found}
        assert "note.md" in names
        assert "CLAUDE.md" in names
        assert "MEMORY.md" not in names
