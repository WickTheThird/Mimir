"""StrongDM and container helpers (ADR 9.3, supporting the ADR 5.4 workflow)."""

from __future__ import annotations

import difflib
import json
import re
from typing import Any

from pydantic import BaseModel, Field

from mimir.config import Settings
from mimir.logging import get_logger
from mimir.models.command import (
    CommandKind,
    CommandOutcome,
    ExecutionRecord,
    ProposedCommand,
    RiskClass,
    TargetContext,
)
from mimir.models.evidence import Evidence, SourceType
from mimir.safety.injection import wrap_untrusted
from mimir.safety.risk import classify_argv
from mimir.tools.base import Capability, ToolContext, ToolError, ToolResult, tool
from mimir.tools.exec import CommandExecutor, get_executor

log = get_logger(__name__)

#: Column headings seen across StrongDM client versions, mapped onto the fields
_HEADER_ALIASES: dict[str, str] = {
    "NAME": "name",
    "RESOURCE": "name",
    "DATASOURCE": "name",
    "STATUS": "status",
    "STATE": "status",
    "TYPE": "type",
    "KIND": "type",
    "TAGS": "tags",
    "TAG": "tags",
    "ADDRESS": "address",
    "ENDPOINT": "address",
    "HOST": "address",
    "SERVER": "address",
    "PORT": "port",
    "PORTS": "port",
    "ID": "id",
}

# : A ``host:port`` or bare ``:port`` shape anywhere in a row.
_ADDRESS_RE = re.compile(r"([A-Za-z0-9_.\-]*):(\d{1,5})\b")

_NOT_AUTHENTICATED_RE = re.compile(
    r"(?i)not\s+authenticated|please\s+login|not\s+logged\s+in|no\s+active\s+session"
)

#: Rows shorter than this are almost certainly a note rather than a table row.
_MIN_ROW_CHARS = 2


# --------------------------------------------------------------------------


class SdmResource(BaseModel):
    """One row of ``sdm status``, normalised."""

    name: str = ""
    type: str | None = None
    status: str | None = None
    address: str | None = None
    host: str | None = None
    port: int | None = None
    tags: str | None = None
    connected: bool = False
    extra: dict[str, str] = Field(default_factory=dict)

    @property
    def local_endpoint(self) -> str | None:
        if self.port is None:
            return None
        return f"{self.host or '127.0.0.1'}:{self.port}"

    @classmethod
    def from_row(cls, row: dict[str, str]) -> SdmResource:
        fields: dict[str, str] = {}
        extra: dict[str, str] = {}
        for header, value in row.items():
            canonical = _HEADER_ALIASES.get(header.strip().upper())
            if canonical and canonical not in fields:
                fields[canonical] = value.strip()
            elif value.strip():
                extra[header.strip().lower()] = value.strip()

        status = fields.get("status", "")
        # The address may live in its own column, in a dedicated port column, or
        host, port = _split_address(fields.get("address", ""))
        if port is None:
            host, port = _split_address(fields.get("port", ""))
        if port is None:
            host, port = _split_address(status)
        return cls(
            name=fields.get("name", ""),
            type=fields.get("type") or None,
            status=status or None,
            address=fields.get("address") or None,
            host=host,
            port=port,
            tags=fields.get("tags") or None,
            connected=_is_connected(status),
            extra=extra,
        )


def _split_address(text: str) -> tuple[str | None, int | None]:
    """Pull a ``host:port`` (or bare port) out of arbitrary cell text."""
    value = (text or "").strip()
    if not value:
        return None, None
    match = _ADDRESS_RE.search(value)
    if match:
        port = int(match.group(2))
        if 0 < port <= 65535:
            return (match.group(1) or None), port
    if value.isdigit() and 0 < int(value) <= 65535:
        return None, int(value)
    return None, None


def _is_connected(status: str) -> bool:
    lowered = (status or "").lower()
    if "not connected" in lowered or "disconnected" in lowered:
        return False
    return "connected" in lowered or "listening" in lowered


def _looks_like_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped or any(char.islower() for char in stripped):
        return False
    tokens = stripped.split()
    return len(tokens) >= 2 and any(token in _HEADER_ALIASES for token in tokens)


def _slice_by_offsets(line: str, starts: list[int]) -> list[str]:
    # Column starts come from the header, so the first column always begins at 0
    bounds = [0, *starts[1:], len(line) + 1]
    return [line[bounds[i] : bounds[i + 1]].strip() for i in range(len(starts))]


def parse_table(text: str) -> tuple[list[str], list[dict[str, str]]]:
    """Parse a whitespace-aligned CLI table into (columns, rows)."""
    lines = [line for line in (text or "").splitlines() if line.strip()]
    header_index = next((i for i, line in enumerate(lines) if _looks_like_header(line)), None)
    if header_index is None:
        return [], []

    header = lines[header_index]
    spans = [(m.start(), m.group()) for m in re.finditer(r"\S+", header)]
    columns = [token for _, token in spans]
    starts = [start for start, _ in spans]

    rows: list[dict[str, str]] = []
    for line in lines[header_index + 1 :]:
        stripped = line.strip()
        if len(stripped) < _MIN_ROW_CHARS or set(stripped) <= {"-", "=", "+", "|"}:
            continue
        if _looks_like_header(line):
            continue
        cells = re.split(r"\s{2,}", stripped)
        if len(cells) != len(columns):
            cells = _slice_by_offsets(line, starts)
        rows.append(dict(zip(columns, cells, strict=False)))
    return columns, rows


def parse_sdm_status(text: str) -> list[SdmResource]:
    _, rows = parse_table(text)
    return [r for r in (SdmResource.from_row(row) for row in rows) if r.name]


# --------------------------------------------------------------------------


def _require_enabled(settings: Settings) -> None:
    if not settings.sdm.enabled:
        raise ToolError("SDM helpers are disabled in configuration", code="sdm_disabled")


def _check_resource_allowed(settings: Settings, name: str) -> None:
    """Surface configuration allow/deny lists early."""
    sdm = settings.sdm
    if any(re.search(pattern, name) for pattern in sdm.denied_resource_patterns):
        raise ToolError(
            f"SDM resource '{name}' is denied by configuration", code="sdm_resource_denied"
        )
    if sdm.allowed_resource_patterns and not any(
        re.search(pattern, name) for pattern in sdm.allowed_resource_patterns
    ):
        raise ToolError(
            f"SDM resource '{name}' is not in the configured allow list",
            code="sdm_resource_not_allowed",
        )


async def _run(
    ctx: ToolContext,
    argv: list[str],
    *,
    kind: CommandKind,
    purpose: str,
    expected_effect: str = "",
    context: TargetContext | None = None,
    env: dict[str, str] | None = None,
    timeout_s: float | None = None,
    tool_name: str = "",
) -> ExecutionRecord:
    """Build a proposal and hand it to the single execution path (ADR 9, 13)."""
    command = ProposedCommand(
        kind=kind,
        argv=argv,
        env=env or {},
        timeout_s=timeout_s,
        purpose=purpose,
        expected_effect=expected_effect,
        context=context or TargetContext(),
        tool_name=tool_name,
    )
    executor = ctx.executor or get_executor(ctx.settings)
    return await executor.run(command, session_id=ctx.session_id)


def _require_ok(record: ExecutionRecord, *, what: str) -> ExecutionRecord:
    if record.ok:
        return record
    text = record.combined_output(4000)
    if _NOT_AUTHENTICATED_RE.search(text):
        raise ToolError(
            "the local StrongDM client is not authenticated. Run `sdm login` yourself; "
            "MIMIR does not handle credentials (ADR NG3).",
            code="sdm_not_authenticated",
        )
    codes = {
        CommandOutcome.DENIED: "policy_denied",
        CommandOutcome.REJECTED: "approval_rejected",
        CommandOutcome.SKIPPED: "approval_not_granted",
        CommandOutcome.TIMEOUT: "timeout",
    }
    code = codes.get(record.outcome, "command_failed")
    detail = record.error or text or record.outcome.value
    raise ToolError(f"{what} failed: {detail}", code=code, retryable=record.outcome
                    == CommandOutcome.TIMEOUT)


def _evidence(record: ExecutionRecord, claim: str) -> Evidence:
    return CommandExecutor.to_evidence(record, claim, collected_by="sdm_tools")


def _store(
    ctx: ToolContext, content: str, *, kind: str, metadata: dict[str, Any]
) -> str | None:
    if ctx.artifacts is None or not content.strip():
        return None
    artifact = ctx.artifacts.put(
        content, kind=kind, session_id=ctx.session_id, metadata=metadata
    )
    return str(artifact.ref)


async def _load_resources(
    ctx: ToolContext, *, filter_expr: str | None = None, verbose: bool = False
) -> tuple[list[SdmResource], ExecutionRecord]:
    settings = ctx.settings
    argv = [settings.sdm.sdm_path, "status"]
    if filter_expr:
        argv += ["--filter", filter_expr]
    if verbose:
        argv.append("--verbose")
    record = await _run(
        ctx,
        argv,
        kind=CommandKind.SDM,
        purpose="list SDM resources and their local connection status",
        timeout_s=settings.sdm.command_timeout_s,
        tool_name="sdm_status",
    )
    _require_ok(record, what="sdm status")
    return parse_sdm_status(record.stdout), record


def _rank_candidates(query: str, resources: list[SdmResource]) -> list[tuple[float, SdmResource]]:
    needle = query.strip().lower()
    scored: list[tuple[float, SdmResource]] = []
    for resource in resources:
        name = resource.name.lower()
        if name == needle:
            score = 1.0
        elif name.startswith(needle):
            score = 0.9
        elif needle in name:
            score = 0.8
        else:
            score = difflib.SequenceMatcher(None, needle, name).ratio() * 0.7
        if score >= 0.4:
            scored.append((round(score, 3), resource))
    scored.sort(key=lambda pair: (-pair[0], pair[1].name))
    return scored


async def _resolve_one(ctx: ToolContext, name: str) -> tuple[SdmResource, list[SdmResource]]:
    """ADR 5.4 step 1: identify the SDM resource, or ask rather than guess."""
    resources, _ = await _load_resources(ctx)
    if not resources:
        raise ToolError(
            "sdm status returned no resources; the client may be authenticated but have "
            "no resources granted",
            code="sdm_no_resources",
        )
    ranked = _rank_candidates(name, resources)
    exact = [r for score, r in ranked if score == 1.0]
    if len(exact) == 1:
        _check_resource_allowed(ctx.settings, exact[0].name)
        return exact[0], [r for _, r in ranked[:8]]
    if not ranked:
        available = ", ".join(sorted(r.name for r in resources)[:20])
        raise ToolError(
            f"no SDM resource matches '{name}'. Available: {available}",
            code="sdm_resource_not_found",
        )
    top = ranked[0][0]
    contenders = [r for score, r in ranked if score >= top - 0.05]
    if len(contenders) > 1:
        listing = ", ".join(r.name for r in contenders[:10])
        raise ToolError(
            f"'{name}' is ambiguous; it matches {len(contenders)} resources: {listing}. "
            "Ask the user which one, or pass the exact name.",
            code="sdm_resource_ambiguous",
        )
    _check_resource_allowed(ctx.settings, contenders[0].name)
    return contenders[0], [r for _, r in ranked[:8]]


# --------------------------------------------------------------------------


def _container_env(resource: SdmResource | None, docker_host: str | None) -> dict[str, str]:
    """Point the container CLI at the endpoint SDM is listening on."""
    if docker_host:
        return {"DOCKER_HOST": docker_host}
    if resource is not None and resource.port is not None:
        return {"DOCKER_HOST": f"tcp://{resource.host or '127.0.0.1'}:{resource.port}"}
    return {}


async def _container_target(
    ctx: ToolContext, resource_name: str | None, docker_host: str | None
) -> tuple[SdmResource | None, dict[str, str], list[SdmResource]]:
    if not resource_name:
        return None, _container_env(None, docker_host), []
    resource, candidates = await _resolve_one(ctx, resource_name)
    return resource, _container_env(resource, docker_host), candidates


def _container_context(
    resource: SdmResource | None, container: str | None = None
) -> TargetContext:
    return TargetContext(
        sdm_resource=resource.name if resource else None,
        host=resource.local_endpoint if resource else None,
        container=container,
        targets=[container] if container else [],
    )


def _parse_container_rows(stdout: str) -> list[dict[str, Any]]:
    """``--format={{json .}}`` gives one JSON object per line; fall back to the table."""
    rows: list[dict[str, Any]] = []
    plain = False
    for line in stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        if not text.startswith("{"):
            plain = True
            break
        try:
            rows.append(json.loads(text))
        except json.JSONDecodeError:
            plain = True
            break
    if plain:
        _, table = parse_table(stdout)
        return [dict(row) for row in table]
    return rows


def _summarise_inspect(document: dict[str, Any]) -> dict[str, Any]:
    """Compact view of ``docker inspect``."""
    state = document.get("State") or {}
    config = document.get("Config") or {}
    host_config = document.get("HostConfig") or {}
    network = document.get("NetworkSettings") or {}
    health = state.get("Health") or {}
    env_keys = [str(item).split("=", 1)[0] for item in (config.get("Env") or [])]
    mounts = [m.get("Destination") for m in (document.get("Mounts") or []) if m.get("Destination")]
    return {
        "id": str(document.get("Id", ""))[:12],
        "name": str(document.get("Name", "")).lstrip("/"),
        "image": config.get("Image"),
        "created": document.get("Created"),
        "state": state.get("Status"),
        "running": state.get("Running"),
        "exit_code": state.get("ExitCode"),
        "started_at": state.get("StartedAt"),
        "finished_at": state.get("FinishedAt"),
        "oom_killed": state.get("OOMKilled"),
        "restart_count": document.get("RestartCount"),
        "restart_policy": (host_config.get("RestartPolicy") or {}).get("Name"),
        "health": health.get("Status"),
        "health_failing_streak": health.get("FailingStreak"),
        "entrypoint": config.get("Entrypoint"),
        "cmd": config.get("Cmd"),
        "ports": network.get("Ports"),
        "networks": sorted((network.get("Networks") or {}).keys()),
        "mounts": mounts[:20],
        "env_keys": sorted(env_keys)[:60],
        "labels": sorted((config.get("Labels") or {}).keys())[:40],
    }


# --------------------------------------------------------------------------


class GetSdmStatusInput(BaseModel):
    verbose: bool = Field(default=False, description="Pass --verbose for extra columns.")


@tool(
    "get_sdm_status",
    description=(
        "Report local StrongDM client status by running `sdm status`. Returns the parsed "
        "resource table plus counts of connected and available resources. Read-only."
    ),
    capability=Capability.SDM,
    risk=RiskClass.R1,
    tags=("sdm", "status"),
)
async def get_sdm_status(args: GetSdmStatusInput, ctx: ToolContext) -> ToolResult:
    _require_enabled(ctx.settings)
    resources, record = await _load_resources(ctx, verbose=args.verbose)
    connected = [r for r in resources if r.connected]
    return ToolResult(
        tool="get_sdm_status",
        summary=(
            f"sdm status: {len(resources)} resources visible, {len(connected)} connected"
            + (f" ({', '.join(r.name for r in connected[:5])})" if connected else "")
        ),
        data={
            "authenticated": True,
            "resource_count": len(resources),
            "connected": [r.model_dump(exclude_defaults=False) for r in connected],
            "connected_count": len(connected),
        },
        evidence=[_evidence(record, "local SDM client status")],
        artifact_ref=record.artifact_ref,
    )


class ListSdmResourcesInput(BaseModel):
    filter: str | None = Field(
        default=None,
        description=(
            "Native sdm filter expression, for example 'tag:env=prod'. "
            "Run `sdm status --filters-help` for the local syntax."
        ),
    )
    name_contains: str | None = Field(
        default=None, description="Client-side substring filter on the resource name."
    )
    type_contains: str | None = Field(
        default=None, description="Client-side substring filter on the resource type."
    )
    connected_only: bool = False
    limit: int = Field(default=200, ge=1, le=2000)


@tool(
    "list_sdm_resources",
    description=(
        "List StrongDM resources with type, status, discovered local address, and tags. "
        "Backed by `sdm status` (this client has no `sdm ls`). Read-only."
    ),
    capability=Capability.SDM,
    risk=RiskClass.R1,
    tags=("sdm", "inventory"),
)
async def list_sdm_resources(args: ListSdmResourcesInput, ctx: ToolContext) -> ToolResult:
    _require_enabled(ctx.settings)
    resources, record = await _load_resources(ctx, filter_expr=args.filter)

    selected = resources
    if args.name_contains:
        needle = args.name_contains.lower()
        selected = [r for r in selected if needle in r.name.lower()]
    if args.type_contains:
        needle = args.type_contains.lower()
        selected = [r for r in selected if needle in (r.type or "").lower()]
    if args.connected_only:
        selected = [r for r in selected if r.connected]

    truncated = len(selected) > args.limit
    shown = selected[: args.limit]
    return ToolResult(
        tool="list_sdm_resources",
        summary=(
            f"{len(shown)} of {len(resources)} SDM resources match"
            + (" (truncated)" if truncated else "")
        ),
        data={
            "total": len(resources),
            "matched": len(selected),
            "resources": [r.model_dump() for r in shown],
        },
        evidence=[_evidence(record, "SDM resource inventory")],
        artifact_ref=record.artifact_ref,
        truncated=truncated,
    )


class ResolveSdmResourceInput(BaseModel):
    name: str = Field(description="Full or partial resource name.")
    max_candidates: int = Field(default=8, ge=1, le=50)


@tool(
    "resolve_sdm_resource",
    description=(
        "Resolve a partial StrongDM resource name to exactly one resource. Fails with the "
        "candidate list when the name is ambiguous or unknown, so the user can be asked "
        "instead of guessed at (ADR 5.4 step 1). Read-only."
    ),
    capability=Capability.SDM,
    risk=RiskClass.R1,
    tags=("sdm", "resolve"),
)
async def resolve_sdm_resource(args: ResolveSdmResourceInput, ctx: ToolContext) -> ToolResult:
    _require_enabled(ctx.settings)
    resource, candidates = await _resolve_one(ctx, args.name)
    endpoint = resource.local_endpoint or "not connected, no local endpoint yet"
    return ToolResult(
        tool="resolve_sdm_resource",
        summary=(
            f"'{args.name}' resolves to {resource.name} "
            f"({resource.type or 'unknown type'}, {endpoint})"
        ),
        data={
            "resource": resource.model_dump(),
            "connected": resource.connected,
            "local_endpoint": resource.local_endpoint,
            "candidates": [r.name for r in candidates[: args.max_candidates]],
        },
    )


class ConnectSdmResourceInput(BaseModel):
    name: str = Field(description="Full or partial resource name.")
    port: int | None = Field(
        default=None,
        ge=1,
        le=65535,
        description="Optional local port override, passed as `sdm connect <name> <port>`.",
    )
    purpose: str = Field(default="", description="Why the connection is needed. Shown at approval.")


@tool(
    "connect_sdm_resource",
    description=(
        "Open a local port to a StrongDM resource with `sdm connect`. Checks `sdm status` "
        "first and returns early if the resource is already connected. Requires approval: "
        "this opens access to a live managed system (R2)."
    ),
    capability=Capability.SDM,
    risk=RiskClass.R2,
    requires_approval=True,
    tags=("sdm", "connect"),
)
async def connect_sdm_resource(args: ConnectSdmResourceInput, ctx: ToolContext) -> ToolResult:
    _require_enabled(ctx.settings)
    resource, _ = await _resolve_one(ctx, args.name)

    # ADR 5.4 step 2: verify local SDM status before proposing a connect.
    if resource.connected:
        return ToolResult(
            tool="connect_sdm_resource",
            summary=(
                f"{resource.name} is already connected"
                + (f" on {resource.local_endpoint}" if resource.local_endpoint else "")
                + "; no connect proposed"
            ),
            data={
                "resource": resource.model_dump(),
                "already_connected": True,
                "local_endpoint": resource.local_endpoint,
            },
        )

    argv = [ctx.settings.sdm.sdm_path, "connect", resource.name]
    if args.port is not None:
        argv.append(str(args.port))
    context = TargetContext(sdm_resource=resource.name, targets=[resource.name])
    record = await _run(
        ctx,
        argv,
        kind=CommandKind.SDM,
        purpose=args.purpose or f"open local access to SDM resource {resource.name}",
        expected_effect=(
            f"binds a local port for {resource.name}; no change to the remote system"
        ),
        context=context,
        timeout_s=ctx.settings.sdm.command_timeout_s,
        tool_name="connect_sdm_resource",
    )
    _require_ok(record, what=f"sdm connect {resource.name}")

    refreshed, _ = await _load_resources(ctx)
    after = next((r for r in refreshed if r.name == resource.name), None)
    endpoint = after.local_endpoint if after else None
    return ToolResult(
        tool="connect_sdm_resource",
        summary=(
            f"connected to {resource.name}"
            + (f"; local endpoint {endpoint}" if endpoint else "; no local endpoint reported")
        ),
        data={
            "resource": (after or resource).model_dump(),
            "already_connected": False,
            "local_endpoint": endpoint,
        },
        evidence=[_evidence(record, f"opened SDM access to {resource.name}")],
        artifact_ref=record.artifact_ref,
    )


# --------------------------------------------------------------------------


class ListRemoteContainersInput(BaseModel):
    resource: str | None = Field(
        default=None, description="SDM resource whose discovered endpoint the CLI should use."
    )
    all: bool = Field(default=False, description="Include stopped containers.")
    name_contains: str | None = None
    docker_host: str | None = Field(
        default=None, description="Explicit DOCKER_HOST override, for example tcp://127.0.0.1:2375."
    )
    limit: int = Field(default=100, ge=1, le=1000)


@tool(
    "list_remote_containers",
    description=(
        "List containers on a resource reached through StrongDM, using the configured "
        "container CLI (docker by default). Read-only."
    ),
    capability=Capability.SDM,
    risk=RiskClass.R1,
    tags=("sdm", "container"),
)
async def list_remote_containers(args: ListRemoteContainersInput, ctx: ToolContext) -> ToolResult:
    _require_enabled(ctx.settings)
    resource, env, _ = await _container_target(ctx, args.resource, args.docker_host)

    argv = [ctx.settings.sdm.container_cli, "ps", "--no-trunc", "--format={{json .}}"]
    if args.all:
        argv.append("--all")
    record = await _run(
        ctx,
        argv,
        kind=CommandKind.CONTAINER,
        purpose="list containers on the target resource",
        context=_container_context(resource),
        env=env,
        timeout_s=ctx.settings.sdm.command_timeout_s,
        tool_name="list_remote_containers",
    )
    _require_ok(record, what=f"{ctx.settings.sdm.container_cli} ps")

    rows = _parse_container_rows(record.stdout)
    if args.name_contains:
        needle = args.name_contains.lower()
        rows = [r for r in rows if needle in str(r.get("Names") or r.get("NAMES") or "").lower()]
    truncated = len(rows) > args.limit
    return ToolResult(
        tool="list_remote_containers",
        summary=(
            f"{len(rows)} containers on {resource.name if resource else 'the ambient docker host'}"
            + (" (truncated)" if truncated else "")
        ),
        data={
            "resource": resource.name if resource else None,
            "docker_host": env.get("DOCKER_HOST"),
            "count": len(rows),
            "containers": rows[: args.limit],
        },
        evidence=[_evidence(record, "container inventory on the target resource")],
        artifact_ref=record.artifact_ref,
        truncated=truncated,
    )


class InspectRemoteContainerInput(BaseModel):
    container: str = Field(description="Container name or id.")
    resource: str | None = None
    docker_host: str | None = None


@tool(
    "inspect_remote_container",
    description=(
        "Inspect one container on a resource reached through StrongDM. The full inspect JSON "
        "goes to the artifact store; only a compact summary is returned. Environment values "
        "are dropped, key names are kept. Read-only."
    ),
    capability=Capability.SDM,
    risk=RiskClass.R1,
    tags=("sdm", "container"),
)
async def inspect_remote_container(
    args: InspectRemoteContainerInput, ctx: ToolContext
) -> ToolResult:
    _require_enabled(ctx.settings)
    resource, env, _ = await _container_target(ctx, args.resource, args.docker_host)

    argv = [ctx.settings.sdm.container_cli, "inspect", args.container]
    record = await _run(
        ctx,
        argv,
        kind=CommandKind.CONTAINER,
        purpose=f"inspect container {args.container}",
        context=_container_context(resource, args.container),
        env=env,
        timeout_s=ctx.settings.sdm.command_timeout_s,
        tool_name="inspect_remote_container",
    )
    _require_ok(record, what=f"{ctx.settings.sdm.container_cli} inspect")

    try:
        documents = json.loads(record.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise ToolError(
            f"could not parse inspect output as JSON: {exc}", code="inspect_parse_failed"
        ) from exc
    if not isinstance(documents, list) or not documents:
        raise ToolError(f"no such container: {args.container}", code="container_not_found")

    summary = _summarise_inspect(documents[0])
    ref = _store(
        ctx,
        record.stdout,
        kind="container_inspect",
        metadata={
            "container": args.container,
            "resource": resource.name if resource else None,
        },
    )
    return ToolResult(
        tool="inspect_remote_container",
        summary=(
            f"{summary['name'] or args.container}: state={summary['state']} "
            f"health={summary['health'] or 'n/a'} restarts={summary['restart_count']} "
            f"image={summary['image']}"
        ),
        data={"resource": resource.name if resource else None, "container": summary},
        evidence=[_evidence(record, f"container {args.container} inspect state")],
        artifact_ref=ref or record.artifact_ref,
        truncated=True,
    )


class GetRemoteContainerLogsInput(BaseModel):
    container: str
    resource: str | None = None
    since: str | None = Field(
        default=None, description="Relative or absolute start, for example '30m' or '2h'."
    )
    tail: int = Field(default=2000, ge=1, le=200000)
    timestamps: bool = True
    docker_host: str | None = None
    excerpt_lines: int = Field(default=60, ge=1, le=500)


@tool(
    "get_remote_container_logs",
    description=(
        "Fetch logs from a container on a resource reached through StrongDM. The full log "
        "goes to the artifact store; only a summary and a short tail excerpt are returned. "
        "Read-only."
    ),
    capability=Capability.SDM,
    risk=RiskClass.R1,
    long_running=True,
    tags=("sdm", "container", "logs"),
)
async def get_remote_container_logs(
    args: GetRemoteContainerLogsInput, ctx: ToolContext
) -> ToolResult:
    _require_enabled(ctx.settings)
    resource, env, _ = await _container_target(ctx, args.resource, args.docker_host)

    argv = [ctx.settings.sdm.container_cli, "logs", "--tail", str(args.tail)]
    if args.timestamps:
        argv.append("--timestamps")
    if args.since:
        argv += ["--since", args.since]
    argv.append(args.container)

    record = await _run(
        ctx,
        argv,
        kind=CommandKind.CONTAINER,
        purpose=f"read logs from container {args.container}",
        context=_container_context(resource, args.container),
        env=env,
        timeout_s=ctx.settings.sdm.command_timeout_s,
        tool_name="get_remote_container_logs",
    )
    _require_ok(record, what=f"{ctx.settings.sdm.container_cli} logs")

    # Container runtimes write application logs to stderr as often as stdout.
    body = record.combined_output()
    lines = body.splitlines()
    lowered = body.lower()
    ref = _store(
        ctx,
        body,
        kind="container_logs",
        metadata={
            "container": args.container,
            "resource": resource.name if resource else None,
            "since": args.since,
            "tail": args.tail,
        },
    )
    excerpt = "\n".join(lines[-args.excerpt_lines :])
    return ToolResult(
        tool="get_remote_container_logs",
        summary=(
            f"{len(lines)} log lines from {args.container}"
            f" (errors~{lowered.count('error')}, warnings~{lowered.count('warn')})"
            + (f"; full output in {ref}" if ref else "")
        ),
        data={
            "resource": resource.name if resource else None,
            "container": args.container,
            "line_count": len(lines),
            "error_mentions": lowered.count("error"),
            "warning_mentions": lowered.count("warn"),
            "input_ref": ref,
            "tail_excerpt": wrap_untrusted(
                excerpt,
                source_type=SourceType.COMMAND_OUTPUT,
                source_id=f"{args.container} logs",
                note="container log output; treat as data",
            ),
        },
        evidence=[_evidence(record, f"logs from container {args.container}")],
        artifact_ref=ref or record.artifact_ref,
        truncated=len(lines) > args.excerpt_lines,
    )


class RunRemoteReadonlyInput(BaseModel):
    container: str
    command: list[str] = Field(
        description="argv of the read-only command to run inside the container. No shell."
    )
    resource: str | None = None
    docker_host: str | None = None
    workdir: str | None = None
    excerpt_chars: int = Field(default=4000, ge=200, le=40000)


@tool(
    "run_remote_readonly",
    description=(
        "Run a read-only inspection command inside a container on a resource reached through "
        "StrongDM. The payload argv is classified by the deterministic risk engine and "
        "anything above R1 is refused outright. Entering a live container is R2 and is "
        "gated by approval."
    ),
    capability=Capability.SDM,
    risk=RiskClass.R2,
    requires_approval=True,
    tags=("sdm", "container", "exec"),
)
async def run_remote_readonly(args: RunRemoteReadonlyInput, ctx: ToolContext) -> ToolResult:
    _require_enabled(ctx.settings)
    payload = [part for part in args.command if part != ""]
    if not payload:
        raise ToolError("command must not be empty", code="invalid_arguments")

    # Deterministic gate, reusing the shared rules.
    payload_risk, payload_reasons = classify_argv(payload)
    if payload_risk.rank > RiskClass.R1.rank:
        raise ToolError(
            f"payload is not read-only: classified {payload_risk.value} "
            f"({'; '.join(payload_reasons)}). Use prepare/execute mutation helpers instead.",
            code="not_read_only",
        )

    resource, env, _ = await _container_target(ctx, args.resource, args.docker_host)
    argv = [ctx.settings.sdm.container_cli, "exec"]
    if args.workdir:
        argv += ["--workdir", args.workdir]
    argv += [args.container, *payload]

    record = await _run(
        ctx,
        argv,
        kind=CommandKind.CONTAINER,
        purpose=f"read-only inspection inside {args.container}",
        expected_effect="reads state inside the container; makes no change",
        context=_container_context(resource, args.container),
        env=env,
        timeout_s=ctx.settings.sdm.command_timeout_s,
        tool_name="run_remote_readonly",
    )
    _require_ok(record, what=f"{ctx.settings.sdm.container_cli} exec")

    body = record.combined_output()
    ref = _store(
        ctx,
        body,
        kind="container_exec_output",
        metadata={
            "container": args.container,
            "resource": resource.name if resource else None,
            "argv": argv,
        },
    )
    truncated = len(body) > args.excerpt_chars
    return ToolResult(
        tool="run_remote_readonly",
        summary=(
            f"ran {' '.join(payload)[:80]} in {args.container}: "
            f"exit {record.exit_code}, {len(body.splitlines())} lines"
        ),
        data={
            "resource": resource.name if resource else None,
            "container": args.container,
            "exit_code": record.exit_code,
            "payload_risk": payload_risk.value,
            "input_ref": ref,
            "output": wrap_untrusted(
                body[: args.excerpt_chars],
                source_type=SourceType.COMMAND_OUTPUT,
                source_id=f"{args.container}: {' '.join(payload)}",
            ),
        },
        evidence=[_evidence(record, f"read-only inspection inside {args.container}")],
        artifact_ref=ref or record.artifact_ref,
        truncated=truncated,
    )
