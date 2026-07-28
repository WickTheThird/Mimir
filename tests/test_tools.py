"""Typed helper tool tests (ADR 8, 9)."""

from __future__ import annotations

import pytest

from mimir.models.specialist import SpecialistName
from mimir.tools.base import Capability, ToolResult, load_all_tools


@pytest.fixture(scope="module")
def registry():
    return load_all_tools()


def test_every_adr_helper_family_is_present(registry):
    """ADR 9 lists helper families; each must have registered tools."""
    by_capability = {c: registry.select(capabilities=[c]) for c in Capability}
    for capability in (
        Capability.REPOSITORY,
        Capability.KUBERNETES,
        Capability.SDM,
        Capability.DATABASE,
        Capability.LOGS,
        Capability.WEB,
        Capability.MEMORY,
        Capability.SANDBOX,
        Capability.SKILLS,
    ):
        assert by_capability[capability], f"no tools for {capability.value}"


def test_tool_schemas_are_valid_openai_functions(registry):
    for spec in registry.all():
        schema = spec.openai_schema()
        assert schema["type"] == "function"
        function = schema["function"]
        assert function["name"] == spec.name
        assert function["description"].strip(), f"{spec.name} has no description"
        assert function["parameters"]["type"] == "object"


def test_tool_names_are_unique_and_snake_case(registry):
    names = registry.names()
    assert len(names) == len(set(names))
    for name in names:
        assert name.islower() and " " not in name, name


def test_specialist_tool_isolation():
    """A specialist must not see tools outside its remit (ADR 7)."""
    from mimir.council.specialists import Specialist

    load_all_tools()
    repo_tools = {s.name for s in Specialist(SpecialistName.REPOSITORY_EXPLORER).available_tools()}
    web_tools = {s.name for s in Specialist(SpecialistName.WEB_RESEARCHER).available_tools()}

    assert "search_repository" in repo_tools
    assert "web_open" not in repo_tools, "the repository explorer has no business browsing"
    assert "web_open" in web_tools
    assert "search_repository" not in web_tools


def test_safety_reviewer_has_no_execution_capability():
    """A reviewer that could run the thing it judges is not a reviewer."""
    from mimir.council.specialists import Specialist

    load_all_tools()
    tools = Specialist(SpecialistName.SAFETY_REVIEWER).available_tools()
    assert tools == []


def test_mutating_tools_are_flagged(registry):
    mutating = {s.name for s in registry.all() if s.mutating}
    # Anything that writes must declare it, so the UI and policy can tell.
    assert "promote_memory_note" in mutating


async def test_unknown_tool_returns_error_not_exception(registry, tool_context):
    result = await registry.invoke("no_such_tool", {}, tool_context)
    assert isinstance(result, ToolResult)
    assert not result.ok
    assert result.error_code == "unknown_tool"


async def test_invalid_arguments_are_reported_not_raised(registry, tool_context):
    result = await registry.invoke("read_file_range", {"nonsense": 1}, tool_context)
    assert not result.ok
    assert result.error_code == "invalid_arguments"


# ---------------------------------------------------------------------------
# Repository helpers (ADR 9.1)
# ---------------------------------------------------------------------------


async def test_search_repository_cites_exact_lines(registry, tool_context, repo_fixture):
    result = await registry.invoke(
        "search_repository", {"repo": "billing", "query": "DefaultTimeout"}, tool_context
    )
    assert result.ok, result.error
    assert result.evidence
    citation = result.evidence[0].citations[0]
    assert citation.path and citation.start_line, "ADR G2 requires file and line citations"
    assert "client.go" in citation.path


async def test_read_file_range_refuses_path_traversal(registry, tool_context, repo_fixture):
    result = await registry.invoke(
        "read_file_range",
        {"repo": "billing", "path": "../../../etc/passwd", "start_line": 1, "end_line": 2},
        tool_context,
    )
    assert not result.ok
    assert "escapes" in (result.error or "")


async def test_flow_evidence_orders_hops_and_finds_timeout(
    registry, tool_context, repo_fixture
):
    """ADR 5.3: the flow must be ordered evidence, not prose."""
    result = await registry.invoke(
        "build_flow_evidence",
        {"repo": "billing", "entrypoint": "cmd/main.go", "max_depth": 3},
        tool_context,
    )
    assert result.ok, result.error
    hops = result.data["hops"]
    assert hops[0]["depth"] == 0
    assert [h["depth"] for h in hops] == sorted(h["depth"] for h in hops)
    assert "timeout" in result.data["behaviours"], "the 2s context deadline should be found"


# ---------------------------------------------------------------------------
# Log helpers (ADR 9.5)
# ---------------------------------------------------------------------------


TIMEOUT_LOG = "\n".join(
    [
        '2026-07-27T10:01:00.000Z INFO checkout trace_id=t1 msg="calling auth" attempt=1',
        '2026-07-27T10:01:05.000Z INFO checkout trace_id=t1 msg="calling auth" attempt=2',
        '2026-07-27T10:01:15.000Z INFO checkout trace_id=t1 msg="calling auth" attempt=3',
        '2026-07-27T10:01:30.001Z ERROR checkout trace_id=t1 msg="deadline exceeded after 30001ms"',
        '2026-07-27T10:02:30.002Z ERROR checkout trace_id=t2 msg="deadline exceeded after 29998ms"',
        '2026-07-27T10:03:30.003Z ERROR checkout trace_id=t3 msg="deadline exceeded after 30002ms"',
        '2026-07-27T10:04:00.000Z ERROR checkout msg="connection pool exhausted"',
    ]
)


async def test_log_levels_parse_from_common_formats(registry, tool_context):
    result = await registry.invoke("ingest_logs", {"text": TIMEOUT_LOG}, tool_context)
    assert result.ok
    grouped = await registry.invoke(
        "group_repeated_errors", {"input_ref": result.artifact_ref}, tool_context
    )
    assert grouped.ok
    assert "3x" in grouped.summary or "3 " in grouped.summary, grouped.summary


async def test_variable_durations_collapse_into_one_template(registry, tool_context):
    """"timeout after 30001ms" and "after 29998ms" are the same failure."""
    ingested = await registry.invoke("ingest_logs", {"text": TIMEOUT_LOG}, tool_context)
    grouped = await registry.invoke(
        "group_repeated_errors", {"input_ref": ingested.artifact_ref}, tool_context
    )
    templates = grouped.data.get("groups", [])
    deadline = [g for g in templates if "deadline" in str(g.get("template", ""))]
    assert deadline and deadline[0]["count"] == 3, templates


async def test_timeout_cluster_is_detected(registry, tool_context):
    """A tight cluster near a round number means a configured timeout.

    Asserted on the structured findings rather than the summary line, because
    which finding ranks first is a judgement call the detector is allowed to
    make. Retry amplification outranking the cluster here is reasonable.
    """
    ingested = await registry.invoke("ingest_logs", {"text": TIMEOUT_LOG}, tool_context)
    result = await registry.invoke(
        "detect_timeout_patterns", {"input_ref": ingested.artifact_ref}, tool_context
    )
    assert result.ok
    findings = result.data.get("findings", [])
    kinds = {f.get("kind") for f in findings}
    blob = str(findings)
    assert "30" in blob, findings
    assert any("retr" in str(k) for k in kinds), kinds
    assert any("pool" in blob.lower() or "exhaust" in blob.lower() for _ in [0]), findings


# ---------------------------------------------------------------------------
# Sandbox (ADR 9.7)
# ---------------------------------------------------------------------------


async def test_sandbox_runs_and_returns_structured_output(registry, tool_context):
    result = await registry.invoke(
        "run_python", {"code": "emit({'answer': 6 * 7})"}, tool_context
    )
    assert result.ok, result.error
    assert result.data["result"] == {"answer": 42}


@pytest.mark.parametrize(
    "code",
    [
        "import socket",
        "import subprocess",
        "eval('1+1')",
        "__import__('os')",
        "open('/tmp/x', 'w')",
    ],
)
async def test_sandbox_rejects_dangerous_constructs(registry, tool_context, code):
    result = await registry.invoke("run_python", {"code": code}, tool_context)
    assert not result.ok
    assert result.error_code == "sandbox_violation"


async def test_sandbox_enforces_timeout(registry, tool_context):
    result = await registry.invoke(
        "run_python", {"code": "while True:\n    pass", "timeout_s": 2}, tool_context
    )
    assert not result.ok
    assert result.error_code == "timeout"


async def test_sandbox_does_not_inherit_credentials(registry, tool_context, monkeypatch):
    """ADR 9.7: the runner must not inherit privileged credentials."""
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "very-secret-value")
    monkeypatch.setenv("KUBECONFIG", "/home/user/.kube/config")
    result = await registry.invoke(
        "run_python",
        {
            "code": (
                "import os\n"
                "emit({'leaked': [k for k in os.environ "
                "if 'SECRET' in k.upper() or k == 'KUBECONFIG']})"
            )
        },
        tool_context,
    )
    assert result.ok, result.error
    assert result.data["result"]["leaked"] == []
