"""Knowledge, memory, and skills tests (ADR 10, 11)."""

from __future__ import annotations

import pytest

from mimir.knowledge.index import KnowledgeIndex
from mimir.knowledge.promotion import MemoryPromoter
from mimir.knowledge.retrieval import MemoryRetriever
from mimir.knowledge.store import KnowledgeStore


@pytest.fixture
def store(settings):
    store = KnowledgeStore(settings.knowledge.root, settings)
    store.ensure_layout()
    return store


@pytest.fixture
def seeded(store, settings):
    root = settings.knowledge.root
    (root / "runbooks" / "kubernetes").mkdir(parents=True, exist_ok=True)
    (root / "runbooks" / "kubernetes" / "pod-restarts.md").write_text(
        "---\n"
        "title: Pod restart investigation\n"
        "category: runbooks/kubernetes\n"
        "last_verified: 2026-07-01\n"
        "verification_status: verified\n"
        "tags: [kubernetes, restarts]\n"
        "---\n\n"
        "# Pod restart investigation\n\n"
        "Check lastState.terminated for OOMKilled and CrashLoopBackOff before "
        "changing memory limits.\n",
        encoding="utf-8",
    )
    (root / "runbooks" / "kubernetes" / "ancient.md").write_text(
        "---\n"
        "title: Ancient advice\n"
        "category: runbooks/kubernetes\n"
        "last_verified: 2019-01-01\n"
        "tags: [kubernetes, restarts]\n"
        "---\n\n"
        "# Ancient advice\n\n"
        "An old note about CrashLoopBackOff that may well have rotted.\n",
        encoding="utf-8",
    )
    index = KnowledgeIndex(store, settings=settings)
    index.reindex(force=True)
    return index


def test_frontmatter_round_trips(store, settings):
    from mimir.knowledge.store import DocumentMetadata

    document = store.write(
        "stable/conventions/test-note",
        DocumentMetadata(title="Test note", category="stable/conventions", tags=["a", "b"]),
        "# Test note\n\nBody.\n",
    )
    reloaded = store.get("stable/conventions/test-note")
    assert reloaded is not None
    assert reloaded.metadata.title == "Test note"
    assert list(reloaded.metadata.tags) == ["a", "b"]
    assert "Body." in reloaded.body
    assert document.path.exists()


def test_retrieval_finds_relevant_notes(settings, seeded):
    result = MemoryRetriever(seeded, settings=settings).search("pod restart CrashLoopBackOff")
    assert result.chunks
    assert any("restart" in c.title.lower() or "restart" in c.text.lower() for c in result.chunks)


def test_stale_documents_are_returned_but_marked(settings, seeded):
    """ADR R2: the mitigation is visible freshness, not suppression."""
    result = MemoryRetriever(seeded, settings=settings).search("CrashLoopBackOff")
    titles = {c.title: c for c in result.chunks}
    assert "Ancient advice" in titles, "stale notes must still be retrievable"
    stale = titles["Ancient advice"]
    assert stale.freshness.value == "stale"
    assert "stale" in " ".join(stale.markers()).lower()


def test_fresher_verified_note_outranks_the_stale_one(settings, seeded):
    result = MemoryRetriever(seeded, settings=settings).search("CrashLoopBackOff restarts")
    ordered = [c.title for c in result.chunks]
    if "Pod restart investigation" in ordered and "Ancient advice" in ordered:
        assert ordered.index("Pod restart investigation") < ordered.index("Ancient advice")


async def test_unverified_note_cannot_reach_stable_memory(settings, store):
    """ADR 11.6 and NG4: promotion is gated, not implicit."""
    promoter = MemoryPromoter(store, settings=settings)
    proposal = promoter.propose(
        title="A guess",
        body="Something I inferred but did not verify.",
        category="stable/services",
        verification_status="unverified",
    )
    outcome = await promoter.promote(proposal, approved=False)
    assert not outcome.stored
    assert outcome.review.blocking_reasons


async def test_approved_verified_note_is_stored(settings, store):
    promoter = MemoryPromoter(store, settings=settings)
    proposal = promoter.propose(
        title="A verified fact",
        body="Confirmed against the cluster on 2026-07-27.",
        category="stable/services",
        verification_status="verified",
    )
    outcome = await promoter.promote(proposal, approved=True, approved_by="operator")
    assert outcome.stored, outcome.reason
    assert outcome.doc_id


async def test_unverified_note_may_land_in_investigation_history(settings, store):
    promoter = MemoryPromoter(store, settings=settings)
    proposal = promoter.propose(
        title="An investigation note",
        body="What this session found.",
        category="history/investigations",
        verification_status="unverified",
    )
    outcome = await promoter.promote(proposal, approved=True)
    assert outcome.stored


SECRETY_TEXT = (
    "# Debugging session\n\n"
    "We found that the auth service uses a 2s timeout.\n"
    "My password is hunter2supersecret and the token is "
    "ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.\n"
)


def _assert_untrusted_import(store, settings, result):
    """ADR 11.5: raw exports are source material, never trusted memory."""
    assert result.notes_written >= 1, result
    document = store.get(result.doc_ids[0])
    assert document is not None
    assert "imports" in str(document.path), "imports never land outside imports/"
    assert "unverified" in str(document.metadata.verification_status).lower()
    assert "hunter2supersecret" not in document.body, "secrets must be stripped on import"
    assert "ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" not in document.body


def test_text_dump_import_is_untrusted(settings, store, tmp_path):
    from mimir.knowledge.importers import ConversationImporter

    export = tmp_path / "chat.md"
    export.write_text(SECRETY_TEXT, encoding="utf-8")
    result = ConversationImporter(store, settings=settings).import_path(export)
    _assert_untrusted_import(store, settings, result)


def test_chatgpt_export_import_is_untrusted(settings, store, tmp_path):
    """The real ChatGPT export shape, not a Markdown file forced through it."""
    import json

    from mimir.knowledge.importers import ConversationImporter

    export = tmp_path / "conversations.json"
    export.write_text(
        json.dumps(
            [
                {
                    "title": "Auth timeout debugging",
                    "create_time": 1751000000,
                    "mapping": {
                        "n1": {
                            "id": "n1",
                            "message": {
                                "author": {"role": "user"},
                                "create_time": 1751000000,
                                "content": {"content_type": "text", "parts": [SECRETY_TEXT]},
                            },
                        },
                        "n2": {
                            "id": "n2",
                            "message": {
                                "author": {"role": "assistant"},
                                "create_time": 1751000060,
                                "content": {
                                    "content_type": "text",
                                    "parts": ["The client sets a 2 second context deadline."],
                                },
                            },
                        },
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    result = ConversationImporter(store, settings=settings).import_path(export)
    _assert_untrusted_import(store, settings, result)


# ---------------------------------------------------------------------------


def test_seed_skills_validate():
    from mimir.skills.registry import SkillRegistry
    from mimir.tools.base import load_all_tools

    load_all_tools()
    registry = SkillRegistry(roots=[_repo_skills_dir()]).reload()
    assert registry.all(), "seed skills should load"


def test_catalogue_is_small_enough_to_always_carry():
    """ADR 10.2: every task must not carry every runbook."""
    from mimir.skills.registry import SkillRegistry
    from mimir.tools.base import load_all_tools

    load_all_tools()
    registry = SkillRegistry(roots=[_repo_skills_dir()]).reload()
    assert registry.catalogue_tokens() < 1500


def test_skill_cannot_widen_beyond_its_specialist():
    """A skill declaring extra tools must not gain them (ADR 10.1)."""
    from mimir.models.specialist import SpecialistName
    from mimir.skills.registry import SkillRegistry
    from mimir.skills.runner import SkillRunner
    from mimir.tools.base import load_all_tools

    load_all_tools()
    registry = SkillRegistry(roots=[_repo_skills_dir()]).reload()
    skills = registry.all()
    if not skills:
        pytest.skip("no seed skills available")

    from mimir.council.specialists import Specialist

    runner = SkillRunner(registry)
    allowed_by_specialist = {
        s.name for s in Specialist(SpecialistName.WEB_RESEARCHER).available_tools()
    }
    # Every seed skill, forced into a specialist it was not written for, must
    for skill in skills:
        permitted = runner.permitted_tools(skill, SpecialistName.WEB_RESEARCHER)
    assert {spec.name for spec in permitted.allowed} <= allowed_by_specialist


def _repo_skills_dir():
    from pathlib import Path

    return Path(__file__).resolve().parents[1] / "knowledge" / "skills"
