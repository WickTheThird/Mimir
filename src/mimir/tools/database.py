"""Database helpers (ADR 9.4, safety model ADR 13).

Five helpers, matching ADR 9.4: schema inspection, read-only query, explain,
mutation preparation, and execution of a prepared mutation. ADR 9.4 requires that
"Database mutation should use explicit transactions and preview where possible",
which is what :func:`prepare_database_mutation` enforces before anything runs.

How a query is made safe
------------------------
Three independent layers, in order of how much they can be fooled:

1. :func:`mimir.safety.risk.classify_sql` classifies the statement. Anything
   above R1 is refused by :func:`run_readonly_query`. This is a parser and a
   parser can be tricked.
2. The session runs with ``default_transaction_read_only=on``, delivered through
   the libpq ``options`` connection parameter. PostgreSQL then refuses any write
   inside that transaction regardless of how the statement was spelled.
3. ``--single-transaction`` wraps the whole input in one BEGIN/COMMIT, so with
   layer 2 in force the transaction is genuinely a read-only transaction and
   ``ON_ERROR_STOP`` rolls it back on the first error.

The BEGIN/COMMIT is supplied by psql rather than spliced into the SQL text on
purpose. Prepending ``BEGIN READ ONLY;`` to the statement would change what the
classifier sees (it would read a multi-statement script whose first statement is
unrecognised) and would misreport the risk of an ordinary SELECT.

Credentials
-----------
No helper here accepts, stores, or logs a password. ADR NG3: "MIMIR is not a
credential vault. It should use existing local authentication systems." Provide
credentials through the usual libpq channels instead: ``PGPASSFILE`` (``~/.pgpass``),
``PGSERVICE`` (``~/.pg_service.conf``), ``PGPASSWORD`` in the ambient environment,
a peer/trust socket, or a local port opened by StrongDM. A ``password=`` value or
a URI carrying user info is rejected outright rather than passed through.

Connection resolution order: explicit ``host``/``port``, then the local endpoint
discovered from ``sdm status`` when ``resource`` is given, then whatever libpq
picks up from the environment.

Every invocation is argv-only, never a shell string, and is spawned through
:class:`mimir.tools.exec.CommandExecutor` so policy, approval, redaction, and
audit apply. The statement itself travels on stdin, which keeps SQL punctuation
out of argv and lets the classifier see the statement rather than a wrapper.
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

from mimir.config import Settings
from mimir.logging import get_logger
from mimir.models.command import (
    CommandKind,
    CommandOutcome,
    ExecutionRecord,
    ProposedCommand,
    RiskAssessment,
    RiskClass,
    TargetContext,
)
from mimir.models.evidence import Evidence, SourceType
from mimir.safety.injection import wrap_untrusted
from mimir.safety.risk import SQL_STATEMENT_SPLIT, classify, classify_sql
from mimir.tools.base import Capability, ToolContext, ToolError, ToolResult, tool
from mimir.tools.exec import CommandExecutor, get_executor

log = get_logger(__name__)

#: ASCII unit and record separators. Using control characters rather than a
#: printable delimiter means a value containing commas, pipes, or newlines still
#: round-trips through the unaligned output format.
FIELD_SEP = "\x1f"
RECORD_SEP = "\x1e"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]{0,62}$")
_PATTERN_RE = re.compile(r"^[A-Za-z0-9_%.\-]{1,128}$")

#: Anything that looks like an inline credential. Refused, never forwarded.
_CREDENTIAL_RE = re.compile(r"(?i)password\s*=|://[^/\s]*:[^/@\s]*@")

_COMMAND_TAG_RE = re.compile(
    r"^(SELECT|INSERT|UPDATE|DELETE|MERGE|COPY|TRUNCATE|CREATE|DROP|ALTER)\s+(\d+)"
    r"(?:\s+(\d+))?\s*$",
    re.MULTILINE,
)

#: Simple single-table forms whose WHERE clause can be reused for a row preview.
#: Deliberately narrow: anything with a subquery, a join, or RETURNING is left
#: without a preview rather than previewed wrongly.
_SIMPLE_DELETE_RE = re.compile(
    r"^\s*delete\s+from\s+(?P<table>[A-Za-z_][\w$]*(?:\.[A-Za-z_][\w$]*)?)"
    r"\s+where\s+(?P<where>.+?)\s*;?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_SIMPLE_UPDATE_RE = re.compile(
    r"^\s*update\s+(?P<table>[A-Za-z_][\w$]*(?:\.[A-Za-z_][\w$]*)?)"
    r"\s+set\s+(?P<assignments>.+?)\s+where\s+(?P<where>.+?)\s*;?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_PREVIEW_BLOCKERS = re.compile(r"(?i)\b(select|join|returning|using|from)\b")


# --------------------------------------------------------------------------
# Connection target
# --------------------------------------------------------------------------


class DatabaseTarget(BaseModel):
    """Where to connect. No password field exists here by design (ADR NG3)."""

    resource: str | None = Field(
        default=None,
        description=(
            "StrongDM resource fronting this database. When host is omitted the local "
            "listen address is discovered from `sdm status`."
        ),
    )
    database: str | None = Field(default=None, description="Database name.")
    host: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    user: str | None = Field(default=None, description="Role name only. Never a password.")
    service: str | None = Field(
        default=None, description="Entry in ~/.pg_service.conf to take connection details from."
    )


def _require_enabled(settings: Settings) -> None:
    if not settings.database_tool.enabled:
        raise ToolError("database helpers are disabled in configuration", code="database_disabled")


def _reject_credentials(target: DatabaseTarget) -> None:
    for label, value in (
        ("host", target.host),
        ("database", target.database),
        ("user", target.user),
        ("service", target.service),
        ("resource", target.resource),
    ):
        if value and _CREDENTIAL_RE.search(value):
            raise ToolError(
                f"the {label} value looks like it carries a credential. MIMIR does not accept "
                "passwords; use PGPASSFILE, PGSERVICE, or an SDM-provided local port "
                "(ADR NG3).",
                code="credential_rejected",
            )


def _quote_conninfo(value: str) -> str:
    """Quote a libpq keyword/value parameter. No shell is involved anywhere."""
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _conninfo(settings: Settings, target: DatabaseTarget, *, read_only: bool) -> str:
    """Build the libpq conninfo string carried in argv.

    The statement timeout travels as a backend option on the connection itself,
    so it applies to every statement of every invocation and is visible in the
    argv shown at approval time.
    """
    options = [f"-c statement_timeout={settings.database_tool.statement_timeout_ms}"]
    if read_only:
        options.append("-c default_transaction_read_only=on")
    parts = [f"options={_quote_conninfo(' '.join(options))}"]
    if target.service:
        parts.append(f"service={_quote_conninfo(target.service)}")
    if target.host:
        parts.append(f"host={_quote_conninfo(target.host)}")
    if target.port:
        parts.append(f"port={target.port}")
    if target.user:
        parts.append(f"user={_quote_conninfo(target.user)}")
    if target.database:
        parts.append(f"dbname={_quote_conninfo(target.database)}")
    return " ".join(parts)


async def _resolve_target(ctx: ToolContext, target: DatabaseTarget) -> DatabaseTarget:
    """Fill in host/port from the SDM listing when only a resource was given.

    This is the ADR 5.4 handoff: the resource is resolved and its local endpoint
    discovered, not assumed.
    """
    _reject_credentials(target)
    if target.host or not target.resource:
        return target

    from mimir.tools.sdm import _resolve_one  # local import: keeps the modules decoupled

    resource, _ = await _resolve_one(ctx, target.resource)
    if resource.port is None:
        raise ToolError(
            f"SDM resource '{resource.name}' has no local endpoint yet. Call "
            "connect_sdm_resource first, or pass host and port explicitly.",
            code="sdm_resource_not_connected",
        )
    return target.model_copy(
        update={"host": resource.host or "127.0.0.1", "port": resource.port}
    )


def _target_context(target: DatabaseTarget) -> TargetContext:
    return TargetContext(
        sdm_resource=target.resource,
        database=target.database,
        host=f"{target.host}:{target.port}" if target.host and target.port else target.host,
        targets=[target.database] if target.database else [],
    )


# --------------------------------------------------------------------------
# psql invocation
# --------------------------------------------------------------------------


def _psql_argv(
    settings: Settings, target: DatabaseTarget, *, read_only: bool, structured: bool
) -> list[str]:
    """Build the psql argv.

    Long option names are used throughout. ``-A`` in particular must be avoided:
    the shared policy rules treat a bare ``-A`` as a wildcard-all flag and would
    escalate an otherwise ordinary statement.
    """
    argv = [
        settings.database_tool.psql_path,
        "--no-psqlrc",
        "--single-transaction",
        "--set=ON_ERROR_STOP=1",
        "--pset=footer=off",
        "--pset=pager=off",
    ]
    if structured:
        argv += [
            "--no-align",
            f"--field-separator={FIELD_SEP}",
            f"--record-separator={RECORD_SEP}",
        ]
    argv.append(f"--dbname={_conninfo(settings, target, read_only=read_only)}")
    return argv


async def _run_sql(
    ctx: ToolContext,
    target: DatabaseTarget,
    statement: str,
    *,
    read_only: bool,
    structured: bool,
    purpose: str,
    expected_effect: str = "",
    tool_name: str = "",
) -> ExecutionRecord:
    settings = ctx.settings
    command = ProposedCommand(
        kind=CommandKind.SQL,
        argv=_psql_argv(settings, target, read_only=read_only, structured=structured),
        stdin=_terminated(statement),
        timeout_s=settings.database_tool.command_timeout_s,
        purpose=purpose,
        expected_effect=expected_effect,
        context=_target_context(target),
        tool_name=tool_name,
        metadata={"read_only_session": read_only},
    )
    executor = ctx.executor or get_executor(settings)
    return await executor.run(command, session_id=ctx.session_id)


def _terminated(statement: str) -> str:
    text = statement.strip()
    if not text.endswith(";"):
        text += ";"
    return text + "\n"


def _require_ok(record: ExecutionRecord, *, what: str) -> ExecutionRecord:
    if record.ok:
        return record
    codes = {
        CommandOutcome.DENIED: "policy_denied",
        CommandOutcome.REJECTED: "approval_rejected",
        CommandOutcome.SKIPPED: "approval_not_granted",
        CommandOutcome.TIMEOUT: "statement_timeout",
    }
    code = codes.get(record.outcome, "query_failed")
    detail = record.stderr.strip() or record.error or record.stdout.strip() or record.outcome.value
    raise ToolError(f"{what} failed: {detail[:1500]}", code=code)


def _statement_count(statement: str) -> int:
    return len([s for s in SQL_STATEMENT_SPLIT.split(statement.strip()) if s.strip()])


def _strip_leading_comments(statement: str) -> str:
    return re.sub(r"^(\s*--[^\n]*\n|\s*/\*.*?\*/)+", "", statement, flags=re.S).strip()


def _parse_rows(stdout: str, max_rows: int) -> tuple[list[str], list[list[str]], int, bool]:
    """Parse the unaligned psql output into columns and rows."""
    body = stdout.strip("\n")
    if not body.strip():
        return [], [], 0, False
    records = [r.strip("\n") for r in body.split(RECORD_SEP)]
    records = [r for r in records if r.strip()]
    if not records:
        return [], [], 0, False
    columns = records[0].split(FIELD_SEP)
    rows = [record.split(FIELD_SEP) for record in records[1:]]
    total = len(rows)
    truncated = total > max_rows
    return columns, rows[:max_rows], total, truncated


def _rows_as_dicts(columns: list[str], rows: list[list[str]]) -> list[dict[str, str]]:
    return [dict(zip(columns, row, strict=False)) for row in rows]


def _affected_rows(stdout: str) -> int | None:
    matches = _COMMAND_TAG_RE.findall(stdout or "")
    if not matches:
        return None
    # INSERT reports "INSERT <oid> <count>"; everything else reports one number.
    verb, first, second = matches[-1]
    return int(second) if verb.upper() == "INSERT" and second else int(first)


def _evidence(record: ExecutionRecord, claim: str) -> Evidence:
    return CommandExecutor.to_evidence(record, claim, collected_by="database_tools")


def _store(ctx: ToolContext, content: str, *, kind: str, metadata: dict[str, Any]) -> str | None:
    if ctx.artifacts is None or not content.strip():
        return None
    artifact = ctx.artifacts.put(content, kind=kind, session_id=ctx.session_id, metadata=metadata)
    return str(artifact.ref)


def _identifier(value: str, *, label: str) -> str:
    if not _IDENTIFIER_RE.match(value):
        raise ToolError(
            f"{label} '{value}' is not a plain SQL identifier; quote-heavy or dotted names "
            "must be inspected with an explicit query",
            code="invalid_identifier",
        )
    return value


def _pattern(value: str, *, label: str) -> str:
    if not _PATTERN_RE.match(value):
        raise ToolError(
            f"{label} '{value}' contains characters that are not allowed in a match pattern",
            code="invalid_pattern",
        )
    return value


# --------------------------------------------------------------------------
# Tools (ADR 9.4)
# --------------------------------------------------------------------------


class InspectDatabaseSchemaInput(DatabaseTarget):
    schema_name: str | None = Field(
        default=None, description="Restrict to one schema. Defaults to all non-system schemas."
    )
    table: str | None = Field(
        default=None, description="When set, returns columns and indexes for this table."
    )
    name_contains: str | None = Field(
        default=None, description="Substring filter on the table name."
    )
    max_rows: int | None = Field(default=None, ge=1, le=5000)


@tool(
    "inspect_database_schema",
    description=(
        "Inspect database schema: list tables, or list the columns and indexes of one table. "
        "Runs read-only catalogue queries inside a read-only transaction with a statement "
        "timeout."
    ),
    capability=Capability.DATABASE,
    risk=RiskClass.R2,
    tags=("database", "schema"),
)
async def inspect_database_schema(
    args: InspectDatabaseSchemaInput, ctx: ToolContext
) -> ToolResult:
    _require_enabled(ctx.settings)
    target = await _resolve_target(ctx, args)
    max_rows = args.max_rows or ctx.settings.database_tool.max_rows

    conditions = ["n.nspname NOT IN ('pg_catalog', 'information_schema')"]
    if args.schema_name:
        conditions.append(f"n.nspname = '{_identifier(args.schema_name, label='schema')}'")
    if args.table:
        conditions.append(f"c.relname = '{_identifier(args.table, label='table')}'")
    if args.name_contains:
        conditions.append(
            f"c.relname LIKE '%{_pattern(args.name_contains, label='name_contains')}%'"
        )
    where = " AND ".join(conditions)

    if args.table:
        statement = (
            "SELECT n.nspname AS schema, c.relname AS table, a.attnum AS position, "
            "a.attname AS column, format_type(a.atttypid, a.atttypmod) AS type, "
            "CASE WHEN a.attnotnull THEN 'no' ELSE 'yes' END AS nullable, "
            "COALESCE(pg_get_expr(d.adbin, d.adrelid), '') AS default "
            "FROM pg_attribute a "
            "JOIN pg_class c ON c.oid = a.attrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "LEFT JOIN pg_attrdef d ON d.adrelid = c.oid AND d.adnum = a.attnum "
            f"WHERE {where} AND a.attnum > 0 AND NOT a.attisdropped "
            "ORDER BY n.nspname, c.relname, a.attnum"
        )
        claim = f"columns of table {args.table}"
    else:
        statement = (
            "SELECT n.nspname AS schema, c.relname AS table, "
            "CASE c.relkind WHEN 'r' THEN 'table' WHEN 'v' THEN 'view' "
            "WHEN 'm' THEN 'materialized view' WHEN 'p' THEN 'partitioned table' "
            "WHEN 'f' THEN 'foreign table' ELSE c.relkind::text END AS kind, "
            "c.reltuples::bigint AS estimated_rows, "
            "pg_size_pretty(pg_total_relation_size(c.oid)) AS total_size "
            "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            f"WHERE {where} AND c.relkind IN ('r', 'v', 'm', 'p', 'f') "
            "ORDER BY n.nspname, c.relname"
        )
        claim = "database table inventory"

    record = await _run_sql(
        ctx,
        target,
        statement,
        read_only=True,
        structured=True,
        purpose=claim,
        tool_name="inspect_database_schema",
    )
    _require_ok(record, what="schema inspection")
    columns, rows, total, truncated = _parse_rows(record.stdout, max_rows)

    data: dict[str, Any] = {
        "database": target.database,
        "resource": target.resource,
        "columns": columns,
        "row_count": total,
        "rows": _rows_as_dicts(columns, rows),
        "truncated": truncated,
    }

    if args.table:
        index_statement = (
            "SELECT schemaname AS schema, tablename AS table, indexname AS index, "
            "indexdef AS definition FROM pg_indexes "
            f"WHERE tablename = '{_identifier(args.table, label='table')}'"
            + (
                f" AND schemaname = '{_identifier(args.schema_name, label='schema')}'"
                if args.schema_name
                else ""
            )
            + " ORDER BY indexname"
        )
        index_record = await _run_sql(
            ctx,
            target,
            index_statement,
            read_only=True,
            structured=True,
            purpose=f"indexes on table {args.table}",
            tool_name="inspect_database_schema",
        )
        _require_ok(index_record, what="index inspection")
        index_columns, index_rows, _, _ = _parse_rows(index_record.stdout, max_rows)
        data["indexes"] = _rows_as_dicts(index_columns, index_rows)

    return ToolResult(
        tool="inspect_database_schema",
        summary=(
            f"{claim}: {total} rows"
            + (f" (showing {len(rows)})" if truncated else "")
            + (f", {len(data.get('indexes', []))} indexes" if args.table else "")
        ),
        data=data,
        evidence=[_evidence(record, claim)],
        artifact_ref=record.artifact_ref,
        truncated=truncated,
    )


class RunReadonlyQueryInput(DatabaseTarget):
    query: str = Field(description="A single SELECT/WITH/SHOW/EXPLAIN statement.")
    max_rows: int | None = Field(default=None, ge=1, le=5000)


@tool(
    "run_readonly_query",
    description=(
        "Run one read-only SQL query and return structured rows. The statement is classified "
        "first and anything above R1 is refused, and the session additionally runs with "
        "default_transaction_read_only=on inside a single transaction, so a write cannot "
        "succeed even if the classifier is fooled."
    ),
    capability=Capability.DATABASE,
    risk=RiskClass.R2,
    tags=("database", "query"),
)
async def run_readonly_query(args: RunReadonlyQueryInput, ctx: ToolContext) -> ToolResult:
    _require_enabled(ctx.settings)
    query = args.query.strip()
    if not query:
        raise ToolError("query must not be empty", code="invalid_arguments")

    risk, reasons = classify_sql(query)
    if risk.rank > RiskClass.R1.rank:
        raise ToolError(
            f"refused: statement classified {risk.value} ({'; '.join(reasons)}). "
            "Use prepare_database_mutation for anything that writes.",
            code="not_read_only",
        )
    if _statement_count(query) > 1:
        raise ToolError(
            "submit one statement at a time so results can be attributed to it",
            code="multiple_statements",
        )

    target = await _resolve_target(ctx, args)
    max_rows = args.max_rows or ctx.settings.database_tool.max_rows
    record = await _run_sql(
        ctx,
        target,
        query,
        read_only=True,
        structured=True,
        purpose="read-only query",
        expected_effect="reads data; the session cannot write",
        tool_name="run_readonly_query",
    )
    _require_ok(record, what="query")

    columns, rows, total, truncated = _parse_rows(record.stdout, max_rows)
    ref = _store(
        ctx,
        record.stdout,
        kind="query_result",
        metadata={"database": target.database, "query": query[:500], "row_count": total},
    )
    return ToolResult(
        tool="run_readonly_query",
        summary=(
            f"{total} rows, {len(columns)} columns"
            + (f"; truncated to {max_rows}" if truncated else "")
        ),
        data={
            "database": target.database,
            "resource": target.resource,
            "statement_timeout_ms": ctx.settings.database_tool.statement_timeout_ms,
            "columns": columns,
            "row_count": total,
            "returned_rows": len(rows),
            "truncated": truncated,
            "truncation_note": (
                f"result set has {total} rows; only the first {max_rows} are returned. "
                "Add a LIMIT or a narrower WHERE clause."
                if truncated
                else None
            ),
            "rows": _rows_as_dicts(columns, rows),
            "input_ref": ref,
        },
        evidence=[_evidence(record, f"result of: {query[:160]}")],
        artifact_ref=ref or record.artifact_ref,
        truncated=truncated,
    )


class ExplainQueryInput(DatabaseTarget):
    query: str
    analyze: bool = Field(
        default=False,
        description=(
            "Run EXPLAIN ANALYZE, which executes the query. Only permitted for SELECT/WITH."
        ),
    )
    verbose: bool = False
    buffers: bool = Field(default=False, description="Requires analyze on older PostgreSQL.")
    max_rows: int | None = Field(default=None, ge=1, le=5000)


@tool(
    "explain_query",
    description=(
        "Return the query plan. EXPLAIN alone is planning only; EXPLAIN ANALYZE actually "
        "executes the query and is therefore permitted only for SELECT and WITH statements. "
        "Always runs in a read-only transaction."
    ),
    capability=Capability.DATABASE,
    risk=RiskClass.R2,
    tags=("database", "explain"),
)
async def explain_query(args: ExplainQueryInput, ctx: ToolContext) -> ToolResult:
    _require_enabled(ctx.settings)
    query = _strip_leading_comments(args.query)
    if not query:
        raise ToolError("query must not be empty", code="invalid_arguments")
    if _statement_count(query) > 1:
        raise ToolError("explain one statement at a time", code="multiple_statements")

    lowered = query.lower()
    if args.analyze and not lowered.startswith(("select", "with", "table", "values")):
        # ANALYZE executes the statement. The read-only transaction would block a
        # write anyway, but failing here gives a clear reason instead of a
        # PostgreSQL error.
        raise ToolError(
            "EXPLAIN ANALYZE executes the statement, so it is only allowed for "
            "SELECT/WITH queries. Use analyze=false, or prepare_database_mutation.",
            code="analyze_requires_select",
        )
    if lowered.startswith("explain"):
        raise ToolError("pass the bare query; EXPLAIN is added here", code="invalid_arguments")

    options = ["FORMAT TEXT"]
    if args.analyze:
        options.append("ANALYZE")
    if args.verbose:
        options.append("VERBOSE")
    if args.buffers and args.analyze:
        options.append("BUFFERS")
    statement = f"EXPLAIN ({', '.join(options)}) {query}"

    target = await _resolve_target(ctx, args)
    max_rows = args.max_rows or ctx.settings.database_tool.max_rows
    record = await _run_sql(
        ctx,
        target,
        statement,
        read_only=True,
        structured=True,
        purpose="explain query plan",
        expected_effect=(
            "executes the query to collect real timings" if args.analyze else "plans only"
        ),
        tool_name="explain_query",
    )
    _require_ok(record, what="explain")

    columns, rows, total, truncated = _parse_rows(record.stdout, max_rows)
    plan = "\n".join(FIELD_SEP.join(row) for row in rows)
    ref = _store(
        ctx,
        record.stdout,
        kind="query_plan",
        metadata={"database": target.database, "query": query[:500], "analyze": args.analyze},
    )
    return ToolResult(
        tool="explain_query",
        summary=(
            f"{'EXPLAIN ANALYZE' if args.analyze else 'EXPLAIN'} returned "
            f"{total} plan lines for: {query[:80]}"
        ),
        data={
            "database": target.database,
            "analyze": args.analyze,
            "columns": columns,
            "plan_lines": total,
            "plan": plan,
            "input_ref": ref,
        },
        evidence=[_evidence(record, f"query plan for: {query[:160]}")],
        artifact_ref=ref or record.artifact_ref,
        truncated=truncated,
    )


class PreparedMutation(BaseModel):
    """A classified, previewed, not-yet-executed mutation."""

    plan_id: str
    statement: str
    target: DatabaseTarget
    risk: RiskClass
    reasons: list[str] = Field(default_factory=list)
    preview_rows: int | None = None
    preview_query: str | None = None
    preview_note: str = ""
    created_at: float = Field(default_factory=time.time)
    executed_at: float | None = None


#: Prepared plans awaiting execution. Local-first and process-scoped: a plan does
#: not survive a restart, which is the safe default for a pending mutation.
_PLANS: dict[str, PreparedMutation] = {}


def _preview_count_query(statement: str) -> tuple[str, str] | tuple[None, str]:
    """Derive a `SELECT count(*)` with the same WHERE clause, when that is sound.

    Returns (query, note) or (None, reason). Anything with a join, subquery, or
    RETURNING gets no preview rather than a misleading one.
    """
    text = statement.strip()
    for pattern, verb in ((_SIMPLE_DELETE_RE, "DELETE"), (_SIMPLE_UPDATE_RE, "UPDATE")):
        match = pattern.match(text)
        if not match:
            continue
        where = match.group("where").strip()
        if _PREVIEW_BLOCKERS.search(where):
            return None, f"{verb} WHERE clause contains a subquery or join; no safe row preview"
        table = match.group("table")
        return (
            f"SELECT count(*) AS affected FROM {table} WHERE {where}",
            f"counts the rows this {verb} would match",
        )
    return None, "statement is not a simple single-table UPDATE or DELETE; no row preview"


class PrepareDatabaseMutationInput(DatabaseTarget):
    statement: str = Field(description="The mutating SQL. It is never executed by this tool.")
    purpose: str = Field(default="", description="Why the change is needed. Shown at approval.")


@tool(
    "prepare_database_mutation",
    description=(
        "Classify and preview a database mutation without executing it. Returns the risk "
        "class, the exact command that would run, and a count of the rows a simple "
        "UPDATE/DELETE would affect. The statement will be executed inside an explicit "
        "single transaction by execute_approved_database_mutation."
    ),
    capability=Capability.DATABASE,
    risk=RiskClass.R0,
    tags=("database", "mutation", "preview"),
)
async def prepare_database_mutation(
    args: PrepareDatabaseMutationInput, ctx: ToolContext
) -> ToolResult:
    _require_enabled(ctx.settings)
    statement = args.statement.strip()
    if not statement:
        raise ToolError("statement must not be empty", code="invalid_arguments")

    risk, reasons = classify_sql(statement)
    if risk.rank <= RiskClass.R1.rank:
        raise ToolError(
            "this statement is read-only; run it with run_readonly_query",
            code="not_a_mutation",
        )

    target = await _resolve_target(ctx, args)
    if not target.database:
        # A mutation with no resolved database is an unclear target (ADR 13.2 R4).
        raise ToolError(
            "name the database explicitly before preparing a mutation",
            code="unclear_target",
        )

    command = ProposedCommand(
        kind=CommandKind.SQL,
        argv=_psql_argv(ctx.settings, target, read_only=False, structured=False),
        stdin=_terminated(statement),
        timeout_s=ctx.settings.database_tool.command_timeout_s,
        purpose=args.purpose or "database mutation",
        expected_effect=(
            "runs inside a single explicit transaction; ON_ERROR_STOP rolls it back on error"
        ),
        context=_target_context(target),
        tool_name="execute_approved_database_mutation",
    )
    assessment: RiskAssessment = classify(command)

    preview_query, note = _preview_count_query(statement)
    preview_rows: int | None = None
    if preview_query:
        preview_record = await _run_sql(
            ctx,
            target,
            preview_query,
            read_only=True,
            structured=True,
            purpose="count rows the pending mutation would affect",
            tool_name="prepare_database_mutation",
        )
        if preview_record.ok:
            _, rows, _, _ = _parse_rows(preview_record.stdout, 1)
            if rows and rows[0] and rows[0][0].strip().isdigit():
                preview_rows = int(rows[0][0].strip())
        else:
            note = f"row preview failed: {preview_record.stderr.strip()[:200] or 'unknown error'}"

    plan = PreparedMutation(
        plan_id=f"dbplan_{uuid.uuid4().hex[:12]}",
        statement=statement,
        target=target,
        risk=assessment.risk,
        reasons=[*reasons, *assessment.reasons],
        preview_rows=preview_rows,
        preview_query=preview_query,
        preview_note=note,
    )
    _PLANS[plan.plan_id] = plan
    log.info(
        "database_mutation_prepared",
        plan_id=plan.plan_id,
        risk=assessment.risk.value,
        database=target.database,
        preview_rows=preview_rows,
    )

    return ToolResult(
        tool="prepare_database_mutation",
        summary=(
            f"prepared {plan.plan_id}: risk {assessment.risk.value}, "
            + (
                f"{preview_rows} rows would be affected"
                if preview_rows is not None
                else "no row preview available"
            )
            + ". Nothing has run. Call execute_approved_database_mutation to proceed."
        ),
        data={
            "plan_id": plan.plan_id,
            "database": target.database,
            "resource": target.resource,
            "risk": assessment.risk.value,
            "statement_risk": risk.value,
            "reasons": plan.reasons,
            "reversible": assessment.reversible,
            "rollback_hint": assessment.rollback_hint,
            "production_target": assessment.production_target,
            "requires_approval": assessment.requires_approval,
            "affected_rows_preview": preview_rows,
            "preview_query": preview_query,
            "preview_note": note,
            "statement": statement,
            "preview": command.render_preview(),
        },
        proposed_commands=[command],
    )


class ExecuteApprovedDatabaseMutationInput(BaseModel):
    plan_id: str = Field(
        description=(
            "Identifier returned by prepare_database_mutation. The approval itself is raised "
            "and recorded by the executor when this runs."
        )
    )
    confirm_statement: str | None = Field(
        default=None,
        description="Optional: the exact statement text, checked against the stored plan.",
    )


@tool(
    "execute_approved_database_mutation",
    description=(
        "Execute a previously prepared database mutation. The statement runs inside a single "
        "explicit transaction through the command executor, so the approval gate, redaction, "
        "and audit trail apply. A plan can only be executed once."
    ),
    capability=Capability.DATABASE,
    risk=RiskClass.R3,
    mutating=True,
    requires_approval=True,
    tags=("database", "mutation"),
)
async def execute_approved_database_mutation(
    args: ExecuteApprovedDatabaseMutationInput, ctx: ToolContext
) -> ToolResult:
    _require_enabled(ctx.settings)
    plan = _PLANS.get(args.plan_id)
    if plan is None:
        raise ToolError(
            f"unknown plan '{args.plan_id}'. Prepare the mutation again.", code="unknown_plan"
        )
    if plan.executed_at is not None:
        raise ToolError(
            f"plan '{args.plan_id}' has already been executed; prepare a new one",
            code="plan_already_executed",
        )
    if args.confirm_statement and args.confirm_statement.strip() != plan.statement:
        raise ToolError(
            "confirm_statement does not match the prepared statement", code="statement_mismatch"
        )

    record = await _run_sql(
        ctx,
        plan.target,
        plan.statement,
        read_only=False,
        structured=False,
        purpose=f"execute prepared mutation {plan.plan_id}",
        expected_effect=(
            f"affects approximately {plan.preview_rows} rows"
            if plan.preview_rows is not None
            else "row count was not previewable"
        ),
        tool_name="execute_approved_database_mutation",
    )
    _require_ok(record, what=f"mutation {plan.plan_id}")
    plan.executed_at = time.time()

    affected = _affected_rows(record.stdout)
    ref = _store(
        ctx,
        record.combined_output(),
        kind="mutation_output",
        metadata={"plan_id": plan.plan_id, "database": plan.target.database},
    )
    log.info(
        "database_mutation_executed",
        plan_id=plan.plan_id,
        database=plan.target.database,
        affected_rows=affected,
    )
    return ToolResult(
        tool="execute_approved_database_mutation",
        summary=(
            f"{plan.plan_id} committed; {affected if affected is not None else 'unknown'} rows "
            f"affected (preview said {plan.preview_rows})"
        ),
        data={
            "plan_id": plan.plan_id,
            "database": plan.target.database,
            "risk": plan.risk.value,
            "affected_rows": affected,
            "affected_rows_preview": plan.preview_rows,
            "matches_preview": (
                None if affected is None or plan.preview_rows is None
                else affected == plan.preview_rows
            ),
            "output": wrap_untrusted(
                record.combined_output(4000),
                source_type=SourceType.COMMAND_OUTPUT,
                source_id=f"psql {plan.plan_id}",
            ),
            "input_ref": ref,
        },
        evidence=[_evidence(record, f"executed mutation {plan.plan_id}")],
        artifact_ref=ref or record.artifact_ref,
    )


def get_prepared_mutation(plan_id: str) -> PreparedMutation | None:
    """Read a pending plan, for the approval UI (ADR 13.3)."""
    return _PLANS.get(plan_id)


def clear_prepared_mutations() -> None:
    _PLANS.clear()
