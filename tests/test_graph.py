"""Orchestration tests (ADR 6.2 C2, 7, 12)."""

from __future__ import annotations

from mimir.graph.runner import EventType, InvestigationRunner
from mimir.graph.state import GraphState, initial_state, merge_into_session
from mimir.models.evidence import Evidence, SourceType
from mimir.models.specialist import (
    Hypothesis,
    SpecialistName,
    SpecialistReport,
)
from mimir.models.state import EnvironmentContext, InvestigationState


def test_environment_merge_prefers_explicit_values():
    discovered = EnvironmentContext(namespace="default", cluster_context="dev", repositories=["a"])
    stated = EnvironmentContext(namespace="payments", repositories=["b"])
    merged = discovered.merge(stated)
    assert merged.namespace == "payments", "what the user said wins"
    assert merged.cluster_context == "dev", "discovered values fill the gaps"
    assert merged.repositories == ["a", "b"]


def test_evidence_deduplicates_on_content():
    state = InvestigationState(user_request="x")
    one = Evidence(claim="c", source_type=SourceType.REPOSITORY, source_id="f.py", excerpt="e")
    two = Evidence(claim="c", source_type=SourceType.REPOSITORY, source_id="f.py", excerpt="e")
    state.add_evidence([one, two])
    assert len(state.evidence) == 1, "identical evidence must not accumulate"


def test_evidence_ranks_by_adr_trust_order():
    """ADR 11.4: live command output outranks a runbook outranks model knowledge."""
    state = InvestigationState(user_request="x")
    state.add_evidence(
        [
            Evidence(claim="from the model", source_type=SourceType.MODEL_KNOWLEDGE, source_id="m"),
            Evidence(claim="from a runbook", source_type=SourceType.RUNBOOK, source_id="r"),
            Evidence(claim="from a command", source_type=SourceType.COMMAND_OUTPUT, source_id="c"),
            Evidence(claim="from the repo", source_type=SourceType.REPOSITORY, source_id="p"),
        ]
    )
    ordered = [e.source_type for e in state.ranked_evidence()]
    assert ordered[0] == SourceType.COMMAND_OUTPUT
    assert ordered[1] == SourceType.REPOSITORY
    assert ordered[-1] == SourceType.MODEL_KNOWLEDGE


def test_rejected_hypotheses_are_kept_not_deleted():
    """ADR 5.6: state what was ruled out, and why."""
    state = InvestigationState(user_request="x")
    hypothesis = Hypothesis(statement="the database is slow", likelihood=0.6)
    state.upsert_hypothesis(hypothesis)
    state.reject_hypothesis(hypothesis.id, reason="query latency was flat throughout")

    assert not state.hypotheses
    assert len(state.rejected_hypotheses) == 1
    assert state.rejected_hypotheses[0].rejected_reason


def test_parallel_channels_merge_without_loss():
    """Two specialists writing concurrently must both survive the merge."""
    session = InvestigationState(user_request="x")
    state: GraphState = initial_state(session)
    state["reports"] = [
        SpecialistReport(specialist=SpecialistName.REPOSITORY_EXPLORER, conclusion="a"),
        SpecialistReport(specialist=SpecialistName.LOG_ANALYST, conclusion="b"),
    ]
    state["evidence"] = [
        Evidence(claim="one", source_type=SourceType.REPOSITORY, source_id="f"),
        Evidence(claim="two", source_type=SourceType.COMMAND_OUTPUT, source_id="c"),
    ]
    merged = merge_into_session(state)
    assert len(merged.reports) == 2
    assert len(merged.evidence) == 2


def test_merge_is_idempotent():
    """The graph folds channels more than once; that must not duplicate."""
    session = InvestigationState(user_request="x")
    state: GraphState = initial_state(session)
    state["reports"] = [SpecialistReport(specialist=SpecialistName.LOG_ANALYST, conclusion="a")]
    merge_into_session(state)
    merge_into_session(state)
    assert len(session.reports) == 1


async def test_full_investigation_runs_without_a_model_runtime(settings, echo_router):
    """The whole graph must complete against the deterministic echo model.

    This is the integration test: coordinator, skill selection, parallel
    specialists, verification, safety review, synthesis, and memory curation.
    """
    runner = InvestigationRunner(settings=settings, router=echo_router)
    seen: dict[str, int] = {}
    async for event in runner.stream("why is checkout timing out against auth?"):
        seen[event.type.value] = seen.get(event.type.value, 0) + 1

    assert seen.get("started") == 1
    assert seen.get("done") == 1
    assert seen.get("node_end", 0) >= 8, "the graph should traverse its nodes"
    assert seen.get("specialist", 0) >= 1
    assert seen.get("answer") == 1
    await runner.aclose()


async def test_investigation_produces_structured_answer(settings, echo_router):
    runner = InvestigationRunner(settings=settings, router=echo_router)
    state = await runner.run("does the billing service fail open when auth times out?")
    assert state.final_answer is not None
    assert state.task_type is not None
    assert state.completed_at is not None
    assert 0.0 <= state.final_confidence <= 1.0
    await runner.aclose()


async def test_confidence_reflects_the_weakest_link(settings, echo_router):
    """An answer with no evidence must not present as confident."""
    runner = InvestigationRunner(settings=settings, router=echo_router)
    state = await runner.run("something with no gatherable evidence at all")
    assert not state.evidence or state.final_confidence <= 0.5
    await runner.aclose()


async def test_failure_is_surfaced_not_swallowed(settings, echo_router, monkeypatch):
    """A crashing node must yield an error event and still finish the stream."""
    import mimir.graph.nodes as nodes

    async def boom(state, deps):
        raise RuntimeError("synthetic node failure")

    monkeypatch.setattr(nodes, "coordinate", boom)
    from mimir.graph import build as build_module

    runner = InvestigationRunner(settings=settings, router=echo_router)
    monkeypatch.setattr(build_module, "coordinate", boom)

    types = [event.type async for event in runner.stream("anything")]
    assert EventType.ERROR in types
    assert types[-1] == EventType.DONE, "the stream must always terminate"
    await runner.aclose()


def test_merge_coerces_checkpoint_dicts():
    """A resumed checkpoint can hand back dicts; that must not crash resume."""
    session = InvestigationState(user_request="x")
    state: GraphState = initial_state(session)
    state["reports"] = [
        {"specialist": "log_analyst", "conclusion": "from a checkpoint", "confidence": 0.5}
    ]
    state["evidence"] = [
        {"claim": "a fact", "source_type": "command_output", "source_id": "kubectl get pods"}
    ]
    merged = merge_into_session(state)
    assert merged.reports[0].specialist == SpecialistName.LOG_ANALYST
    assert merged.evidence[0].source_type == SourceType.COMMAND_OUTPUT


def test_merge_drops_unparseable_entries_without_crashing():
    session = InvestigationState(user_request="x")
    state: GraphState = initial_state(session)
    state["reports"] = [{"nonsense": True}]
    merged = merge_into_session(state)
    assert merged.reports == []


async def test_session_survives_a_resume(settings, echo_router):
    """ADR 14.1: `mimir session resume` must actually work."""
    runner = InvestigationRunner(settings=settings, router=echo_router)
    state = await runner.run("a question worth resuming")
    resumed = await runner.resume(state.session_id)
    assert resumed is not None
    assert resumed.session_id == state.session_id
    await runner.aclose()


async def test_executed_commands_reach_the_audit_trail(settings, echo_router):
    """Commands run by tools must land in the session, not only in the executor.

    Typed helpers execute through the shared CommandExecutor and return a
    ToolResult; the ExecutionRecord stays behind in the executor. Without an
    explicit harvest the executions table stays empty forever, and "auditable
    command execution" is nominal rather than real. That was the actual state
    of the system until this was added.
    """
    from mimir.models.command import ProposedCommand

    runner = InvestigationRunner(settings=settings, router=echo_router)
    state = runner.new_session("audit check")
    await runner.executor.run(
        ProposedCommand(argv=["echo", "audited"]), session_id=state.session_id
    )
    assert not state.commands_executed, "not harvested until the run finalises"

    await runner.run("audit check", state=state)
    assert len(state.commands_executed) == 1
    assert state.commands_executed[0].display == "echo audited"
    await runner.aclose()


def test_eval_offline_registry_excludes_live_capabilities():
    """A benchmark must not depend on live infrastructure state.

    It also must not reach the operator's real clusters unattended, which is
    what happened the first time model cases were scored.
    """
    from mimir.eval.harness import EvalHarness
    from mimir.tools.base import load_all_tools

    full = load_all_tools()
    offline = EvalHarness().offline_registry()
    removed = set(full.names()) - set(offline.names())

    assert "get_logs" in removed
    assert "exec_readonly" in removed
    assert "run_readonly_query" in removed
    # Repository and reasoning tools must survive, or the cases cannot run.
    assert "search_repository" in offline.names()
    assert "ingest_logs" in offline.names()


def test_failure_taxonomy_distinguishes_retrieval_from_synthesis():
    """"Never looked" and "looked but did not cite" need different fixes.

    One calls for better retrieval, the other for a synthesis gate. Conflating
    them sends effort to the wrong layer, which is why they are separate
    categories rather than one "citation problem".
    """
    from mimir.eval.harness import CaseKind, EvalCase, FailureCategory, classify_failure
    from mimir.models.evidence import Evidence, SourceType
    from mimir.models.specialist import FinalAnswer
    from mimir.models.state import InvestigationState

    case = EvalCase(id="c", kind=CaseKind.FEATURE_EXISTENCE, prompt="p",
                    expect_contains=["risk.py"])

    gathered = InvestigationState(user_request="p")
    gathered.add_evidence(
        [Evidence(claim="found it", source_type=SourceType.REPOSITORY, source_id="risk.py")]
    )
    gathered.final_answer = FinalAnswer(answer="it is somewhere")
    assert FailureCategory.MISSING_CITATION.value in classify_failure(
        case, gathered, ["missing 'risk.py'"]
    )

    empty = InvestigationState(user_request="p")
    empty.final_answer = FinalAnswer(answer="it is somewhere")
    assert FailureCategory.UNSUPPORTED_REPOSITORY_CLAIM.value in classify_failure(
        case, empty, ["missing 'risk.py'"]
    )


def test_failure_taxonomy_flags_guessed_targets():
    from mimir.eval.harness import CaseKind, EvalCase, FailureCategory, classify_failure
    from mimir.models.state import InvestigationState

    case = EvalCase(id="c", kind=CaseKind.TARGETING, prompt="restart api",
                    expect_refusal=True)
    state = InvestigationState(user_request="restart api")
    categories = classify_failure(case, state, ["expected a refusal or a request for context"])
    assert FailureCategory.MISSING_TARGET_INFORMATION.value in categories


def test_audit_gap_is_a_hard_gate():
    """An audit gap must fail the run even when every case passed.

    Commands that ran without being recorded is the exact defect that made the
    audit trail nominal, and it is invisible in a pass rate.
    """
    from mimir.eval.harness import CaseKind, CaseResult, EvalReport, FailureCategory

    report = EvalReport()
    report.results.append(
        CaseResult(case_id="c", kind=CaseKind.FEATURE_EXISTENCE, passed=True)
    )
    assert report.acceptable

    report.results.append(
        CaseResult(
            case_id="d",
            kind=CaseKind.FEATURE_EXISTENCE,
            passed=False,
            failure_categories=[FailureCategory.AUDIT_GAP.value],
        )
    )
    assert report.audit_gaps == 1
    assert not report.acceptable


def test_hidden_corpus_is_excluded_by_default(settings, tmp_path):
    """A held-out set must not load unless asked for.

    Tuning against the cases you also score on produces a number that measures
    how well you tuned, not whether anything generalised.
    """
    from mimir.eval.harness import EvalHarness

    hidden = EvalHarness.hidden_corpus_dir(settings)
    hidden.mkdir(parents=True, exist_ok=True)
    (hidden / "held_out.yaml").write_text(
        "cases:\n"
        "  - id: hidden-001\n"
        "    kind: risk_classification\n"
        "    prompt: held out\n"
        "    argv: [kubectl, get, pods]\n"
        "    context: {namespace: x, cluster_context: y}\n"
        "    expect_risk: R1\n",
        encoding="utf-8",
    )

    default = {c.id for c in EvalHarness.load_corpus()}
    assert "hidden-001" not in default

    with_hidden = {
        c.id for c in EvalHarness.load_corpus(include_hidden=True, settings=settings)
    }
    assert "hidden-001" in with_hidden


def _provenance(**overrides):
    """A complete, comparable provenance record in schema v2 shape."""
    from mimir.eval.provenance import PROVENANCE_SCHEMA_VERSION

    record = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "source": {"commit": "abc123", "dirty": False, "diff_hash": "",
                   "changed_during_run": False},
        "evaluation": {
            "corpus_hash": "aaa", "prompts_hash": "bbb", "skills_hash": "ccc",
            "offline": True, "contaminated": False, "contaminated_reason": "",
            "external_calls": 0, "enabled_tools_hash": "ttt",
            "enabled_tools": [], "enabled_capabilities": [],
        },
        "runtime": {"name": "ollama", "version": "0.32.5"},
        "models": {},
    }
    for path, value in overrides.items():
        section, _, key = path.partition(".")
        if key:
            record[section][key] = value
        else:
            record[section] = value
    return record


def test_provenance_reports_confounds_between_runs():
    """Two runs that differ in more than the thing under test are not comparable."""
    from mimir.eval.provenance import comparable

    base = _provenance()
    assert comparable(base, _provenance()) == []

    other_corpus = _provenance(**{"evaluation.corpus_hash": "zzz"})
    assert any("corpus" in p for p in comparable(base, other_corpus))
    assert any(
        "live infrastructure" in p
        for p in comparable(base, _provenance(**{"evaluation.offline": False}))
    )
    assert any(
        "not offered the same capabilities" in p
        for p in comparable(base, _provenance(**{"evaluation.enabled_tools_hash": "other"}))
    )

    # A clean baseline against a dirty candidate is a source difference, and the
    # message should say so rather than just labelling it "dirty".
    dirty = _provenance(**{"source.dirty": True, "source.diff_hash": "d1"})
    assert any("different working trees" in p for p in comparable(base, dirty))

    # Two runs from the SAME dirty tree are comparable to each other even though
    # neither is reproducible from the commit. Collapsing both cases into one
    # "dirty" verdict would hide that distinction.
    problems = comparable(dirty, _provenance(**{"source.dirty": True, "source.diff_hash": "d1"}))
    assert any("same dirty tree" in p for p in problems)
    assert not any("different working trees" in p for p in problems)


def test_comparison_fails_closed_on_incomplete_provenance():
    """Two runs that both recorded nothing are not thereby equivalent.

    The previous version compared field to field, so an empty tool hash on both
    sides compared equal and the capability check silently did nothing.
    """
    from mimir.eval.provenance import comparable

    base = _provenance()
    blank = _provenance(**{"evaluation.enabled_tools_hash": ""})

    problems = comparable(blank, _provenance(**{"evaluation.enabled_tools_hash": ""}))
    assert problems, "two runs with no tool fingerprint must not compare as equal"
    assert any("incomplete provenance" in p for p in problems)
    assert any("enabled_tools_hash" in p for p in problems)

    assert any("incomplete provenance" in p for p in comparable(base, blank))


def test_comparison_refuses_older_schema_versions():
    from mimir.eval.provenance import comparable

    legacy = {"corpus_hash": "aaa", "prompts_hash": "bbb", "mimir_commit": "abc123"}
    problems = comparable(_provenance(), legacy)
    assert any("schema" in p for p in problems)


def test_source_changing_mid_run_invalidates_the_comparison():
    """Provenance was collected at persistence time, so edits made while a run
    was in flight were recorded as the state that produced it."""
    from mimir.eval.provenance import comparable

    moved = _provenance(**{"source.changed_during_run": True})
    assert any("change while it was running" in p for p in comparable(_provenance(), moved))


def test_provenance_marks_unresolved_models():
    """An unqueryable runtime must be recorded as unknown, not silently omitted."""
    from mimir.eval.provenance import resolve_model

    identity = resolve_model("deep")
    assert identity.name
    # With no runtime listening the digest cannot be known, and that must be
    # visible rather than implied.
    if not identity.resolved:
        assert identity.digest == ""


def test_offline_selection_is_an_allowlist_not_a_denylist():
    """A new capability must be unsafe until classified.

    The first implementation used a denylist naming kubernetes, sdm, and
    database. It omitted web, and a supposedly offline benchmark sent evaluation
    prompts to Google, Yandex, and Brave. An allowlist fails closed instead.
    """
    from mimir.eval.harness import EvalHarness
    from mimir.tools.base import Capability, load_all_tools

    full = load_all_tools()
    offline = EvalHarness().offline_registry()
    enabled = set(offline.names())

    for spec in full.all():
        if spec.capability in (
            Capability.WEB,
            Capability.KUBERNETES,
            Capability.SDM,
            Capability.DATABASE,
            Capability.SHELL,
        ):
            assert spec.name not in enabled, f"{spec.name} must not be offline-safe"

    assert "search_repository" in enabled
    assert "ingest_logs" in enabled


def test_network_containment_blocks_external_and_permits_loopback():
    """Containment is the second layer, in case a tool is misclassified."""
    import socket

    from mimir.eval.offline import OfflineViolation, network_containment

    with network_containment() as report:
        try:
            socket.getaddrinfo("www.google.com", 443)
            raise AssertionError("external resolution should have been blocked")
        except OfflineViolation:
            pass
        socket.getaddrinfo("127.0.0.1", 11434)

    assert report.external_calls == 1
    assert "www.google.com" in report.blocked_hosts
    assert report.allowed_loopback >= 1
    assert not report.clean


def test_contaminated_run_is_refused_for_comparison():
    from mimir.eval.provenance import comparable

    clean = _provenance()
    assert comparable(clean, _provenance()) == []

    contaminated = _provenance(
        **{"evaluation.contaminated": True,
           "evaluation.contaminated_reason": "web tools remained enabled"}
    )
    problems = comparable(clean, contaminated)
    assert any("CONTAMINATED" in p for p in problems)


def test_external_calls_fail_the_gate():
    """An offline run that reached the network is not acceptable, whatever it scored."""
    from mimir.eval.harness import EvalReport

    report = EvalReport()
    assert report.acceptable

    report.external_calls = 1
    assert not report.acceptable


def test_dispatching_tools_cannot_resolve_past_a_filtered_registry():
    """parallel_search fans out by name and previously used the global registry,
    which let it reach tools deliberately excluded from a filtered one."""
    from mimir.eval.harness import EvalHarness
    from mimir.tools.base import ToolContext
    from mimir.tools.search import _resolve_tool

    offline = EvalHarness().offline_registry()
    ctx = ToolContext(registry=offline)
    assert _resolve_tool(ctx, "web", None) is None
    assert _resolve_tool(ctx, "web", "web_search") is None
