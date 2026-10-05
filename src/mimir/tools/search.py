"""K1 parallel search helper (ADR 8 K1, 5)."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from mimir.logging import get_logger
from mimir.models.command import RiskClass
from mimir.models.evidence import Evidence, SourceType
from mimir.tools.base import REGISTRY, Capability, ToolContext, ToolResult, ToolSpec, tool

log = get_logger(__name__)

SourceName = Literal["repos", "memory", "web"]

# : Helpers to try per logical source, best first.
SOURCE_TOOL_CANDIDATES: dict[str, tuple[str, ...]] = {
    "repos": (
        "search_repository",
        "repo_search",
        "repo_grep",
        "search_repositories",
        "code_search",
        "grep_repositories",
    ),
    "memory": (
        "memory_search",
        "knowledge_search",
        "search_memory",
        "search_knowledge",
        "search_notes",
        "recall_incidents",
    ),
    "web": ("web_search",),
}

# : Baseline weight per source, mirroring the ADR 11.4 trust ladder.
SOURCE_WEIGHT: dict[str, float] = {"repos": 1.0, "memory": 0.8, "web": 0.5}

_QUERY_FIELDS = ("query", "pattern", "q", "text", "term", "search", "question", "keywords")
_LIMIT_FIELDS = ("max_results", "limit", "top_k", "k", "max_matches", "n", "count")
_REPO_FIELDS = ("repos", "repositories", "repo", "repository", "repo_names", "names")
_PATH_FIELDS = ("paths", "path_globs", "globs", "include", "path")
#: Helpers that take a regex by default are switched to literal matching, since
_LITERAL_FIELDS = ("fixed_string", "literal", "fixed", "plain_text")

_LOCATOR_KEYS = (
    "path",
    "file",
    "file_path",
    "url",
    "locator",
    "ref",
    "document_id",
    "id",
    "symbol",
    "title",
    "name",
)
_SNIPPET_KEYS = (
    "snippet",
    "excerpt",
    "line",
    "text",
    "content",
    "summary",
    "preview",
    "body",
    "match",
    "description",
)
_SCORE_KEYS = ("score", "relevance", "similarity", "rank")
_LIST_KEYS = ("results", "matches", "hits", "items", "files", "documents", "entries", "rows")

_WORD_RE = re.compile(r"[a-z0-9_]{2,}")
_SNIPPET_CHARS = 220


@dataclass(slots=True)
class SearchResult:
    """One normalised hit, deliberately small enough to print in bulk."""

    source: str
    tool: str
    locator: str
    snippet: str
    score: float = 0.0
    query: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        row = {
            "source": self.source,
            "locator": self.locator,
            "snippet": self.snippet,
            "score": round(self.score, 3),
        }
        if self.extra:
            row.update(self.extra)
        return row

    def render(self) -> str:
        return f"[{self.source}] {self.locator}\n    {self.snippet}"


def _one_line(text: str, limit: int = _SNIPPET_CHARS) -> str:
    collapsed = " ".join(str(text).split())
    return collapsed[:limit] + ("..." if len(collapsed) > limit else "")


def _terms(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _relevance(query: str, *parts: str) -> float:
    """Fraction of query terms present in the hit."""
    wanted = _terms(query)
    if not wanted:
        return 0.0
    haystack = _terms(" ".join(parts))
    return len(wanted & haystack) / len(wanted)


def _first_key(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _registry(ctx: ToolContext) -> Any:
    """The registry this fan-out must resolve through."""
    return getattr(ctx, "registry", None) or REGISTRY


def _resolve_tool(
    ctx: ToolContext, source: str, override: str | None
) -> ToolSpec[Any] | None:
    registry = _registry(ctx)
    if override:
        return registry.get(override)
    for name in SOURCE_TOOL_CANDIDATES.get(source, ()):
        spec = registry.get(name)
        if spec is not None:
            return spec
    return None


def _build_args(
    spec: ToolSpec[Any],
    *,
    query: str,
    limit: int,
    repos: list[str] | None,
    paths: list[str] | None,
    literal: bool,
) -> dict[str, Any]:
    """Map generic parameters onto the target helper's own field names."""
    fields = spec.input_model.model_fields
    args: dict[str, Any] = {}
    filled: set[str] = set()

    def assign(names: tuple[str, ...], value: Any, *, as_list: bool = False) -> None:
        for name in names:
            info = fields.get(name)
            if info is None:
                continue
            annotation = str(info.annotation)
            wants_list = "list" in annotation.lower()
            if as_list and not wants_list and isinstance(value, list):
                if not value:
                    return
                args[name] = value[0]
            elif wants_list and not isinstance(value, list):
                args[name] = [value]
            else:
                args[name] = value
            filled.add(name)
            return

    assign(_QUERY_FIELDS, query)
    assign(_LIMIT_FIELDS, limit)
    if literal:
        assign(_LITERAL_FIELDS, True)
    if repos:
        assign(_REPO_FIELDS, list(repos), as_list=True)
    if paths:
        assign(_PATH_FIELDS, list(paths), as_list=True)

    missing = [
        name
        for name, info in fields.items()
        if info.is_required() and name not in filled
    ]
    if missing:
        raise ValueError(
            f"cannot delegate to {spec.name}: it requires {', '.join(sorted(missing))} "
            "which parallel_search does not know how to supply"
        )
    return args


def _from_evidence(item: Evidence, source: str, tool_name: str, query: str) -> SearchResult:
    locator = item.citations[0].render() if item.citations else item.source_id
    return SearchResult(
        source=source,
        tool=tool_name,
        locator=locator or item.source_id,
        snippet=_one_line(item.excerpt or item.claim),
        score=item.confidence,
        query=query,
        extra=(
            {"untrusted": True}
            if item.source_type == SourceType.WEB
            else {}
        ),
    )


def _normalise(result: ToolResult, source: str, tool_name: str, query: str) -> list[SearchResult]:
    """Turn any helper's output into comparable hits."""
    hits: list[SearchResult] = []
    for item in result.evidence:
        hits.append(_from_evidence(item, source, tool_name, query))
    if hits:
        return hits

    for key in _LIST_KEYS:
        rows = result.data.get(key)
        if not isinstance(rows, list) or not rows:
            continue
        for row in rows:
            if isinstance(row, str):
                hits.append(
                    SearchResult(
                        source=source,
                        tool=tool_name,
                        locator=_one_line(row, 160),
                        snippet=_one_line(row),
                        query=query,
                    )
                )
                continue
            if not isinstance(row, dict):
                continue
            locator = _first_key(row, _LOCATOR_KEYS)
            snippet = _first_key(row, _SNIPPET_KEYS)
            if locator is None and snippet is None:
                continue
            raw_score = _first_key(row, _SCORE_KEYS)
            try:
                score = float(raw_score) if raw_score is not None else 0.0
            except (TypeError, ValueError):
                score = 0.0
            line = row.get("line_number") or row.get("start_line")
            hits.append(
                SearchResult(
                    source=source,
                    tool=tool_name,
                    locator=f"{locator}:{line}" if locator and line else _one_line(
                        str(locator or snippet), 200
                    ),
                    snippet=_one_line(str(snippet or locator)),
                    score=score,
                    query=query,
                    extra={"artifact_ref": result.artifact_ref} if result.artifact_ref else {},
                )
            )
        if hits:
            return hits

    if result.summary:
        hits.append(
            SearchResult(
                source=source,
                tool=tool_name,
                locator=result.artifact_ref or tool_name,
                snippet=_one_line(result.summary),
                query=query,
            )
        )
    return hits


def _dedup_key(hit: SearchResult) -> tuple[str, str]:
    locator = re.sub(r"\s+", " ", hit.locator.strip().lower())
    locator = locator.rstrip("/")
    return hit.source, locator


def rank_and_dedup(hits: list[SearchResult], limit: int) -> list[SearchResult]:
    """Merge scores per unique locator and return the strongest hits first."""
    merged: dict[tuple[str, str], SearchResult] = {}
    for hit in hits:
        key = _dedup_key(hit)
        existing = merged.get(key)
        if existing is None:
            merged[key] = hit
            continue
        existing.score = max(existing.score, hit.score) + 0.1
        if len(hit.snippet) > len(existing.snippet):
            existing.snippet = hit.snippet
    ordered = sorted(merged.values(), key=lambda h: (-h.score, h.source, h.locator))
    return ordered[:limit]


class ParallelSearchInput(BaseModel):
    queries: list[str] = Field(
        min_length=1,
        description="One or more focused queries. They are run concurrently against every source.",
    )
    sources: list[SourceName] = Field(
        default_factory=lambda: ["repos", "memory"],
        description="Which sources to fan out to: repos, memory, web.",
    )
    max_results: int = Field(
        default=20, ge=1, le=100, description="Cap on the merged, ranked result set."
    )
    per_source_limit: int = Field(
        default=10, ge=1, le=50, description="Cap requested from each delegated helper."
    )
    repos: list[str] = Field(
        default_factory=list, description="Restrict repository search to these repositories."
    )
    paths: list[str] = Field(
        default_factory=list, description="Restrict repository search to these path globs."
    )
    tool_overrides: dict[str, str] = Field(
        default_factory=dict,
        description="Force a specific helper per source, for example {'repos': 'repo_grep'}.",
    )
    include_evidence: bool = Field(
        default=False,
        description="Also return the evidence items the delegated helpers produced.",
    )
    literal: bool = Field(
        default=True,
        description="Treat queries as literal text. Set false to pass them through as regex.",
    )


@tool(
    "parallel_search",
    description=(
        "Run several queries across repositories, incident/runbook memory, and the web at the "
        "same time and get back one compact ranked list of source, locator, and a one-line "
        "snippet. Use this instead of issuing separate search calls per source or per query; "
        "it exists to cut round trips. Open the promising locators afterwards with read_document "
        "or web_open."
    ),
    capability=Capability.INTERNAL,
    risk=RiskClass.R1,
    tags=("search", "k1", "parallel"),
)
async def parallel_search(args: ParallelSearchInput, ctx: ToolContext) -> ToolResult:
    started = time.time()
    plan: list[tuple[str, ToolSpec[Any], str, dict[str, Any]]] = []
    errors: list[dict[str, str]] = []
    searched: dict[str, str] = {}

    for source in dict.fromkeys(args.sources):
        spec = _resolve_tool(ctx, source, args.tool_overrides.get(source))
        if spec is None:
            candidates = ", ".join(SOURCE_TOOL_CANDIDATES.get(source, ())) or "(none)"
            errors.append(
                {
                    "source": source,
                    "tool": "",
                    "error": f"no helper registered for {source}; tried: {candidates}",
                }
            )
            continue
        if source == "web" and not ctx.settings.web.enabled:
            errors.append({"source": source, "tool": spec.name, "error": "web access is disabled"})
            continue
        searched[source] = spec.name
        for query in args.queries:
            try:
                call_args = _build_args(
                    spec,
                    query=query,
                    limit=args.per_source_limit,
                    repos=args.repos or None,
                    paths=args.paths or None,
                    literal=args.literal,
                )
            except ValueError as exc:
                errors.append({"source": source, "tool": spec.name, "error": str(exc)})
                break
            plan.append((source, spec, query, call_args))

    if not plan:
        return ToolResult(
            ok=False,
            tool="parallel_search",
            summary="no source could be searched",
            error="; ".join(e["error"] for e in errors) or "no sources selected",
            error_code="no_sources",
            data={"errors": errors, "sources_searched": searched},
        )

    async def run(spec: ToolSpec[Any], call_args: dict[str, Any]) -> ToolResult:
        return await spec.invoke(call_args, ctx)

    outcomes = await asyncio.gather(
        *(run(spec, call_args) for _, spec, _, call_args in plan),
        return_exceptions=True,
    )

    hits: list[SearchResult] = []
    evidence: list[Evidence] = []
    for (source, spec, query, _), outcome in zip(plan, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            log.warning("parallel_search_source_crashed", source=source, tool=spec.name)
            errors.append(
                {
                    "source": source,
                    "tool": spec.name,
                    "query": query,
                    "error": f"{type(outcome).__name__}: {outcome}",
                }
            )
            continue
        if not outcome.ok:
            errors.append(
                {
                    "source": source,
                    "tool": spec.name,
                    "query": query,
                    "error": outcome.error or "unknown error",
                }
            )
            continue
        weight = SOURCE_WEIGHT.get(source, 0.5)
        for hit in _normalise(outcome, source, spec.name, query):
            hit.score = weight + hit.score * 0.5 + _relevance(query, hit.locator, hit.snippet)
            hits.append(hit)
        if args.include_evidence:
            evidence.extend(outcome.evidence)

    ranked = rank_and_dedup(hits, args.max_results)
    by_source: dict[str, int] = {}
    for hit in ranked:
        by_source[hit.source] = by_source.get(hit.source, 0) + 1

    summary_bits = [f"{len(ranked)} result(s) from {len(plan)} concurrent search(es)"]
    if by_source:
        summary_bits.append(", ".join(f"{k}={v}" for k, v in sorted(by_source.items())))
    if errors:
        summary_bits.append(f"{len(errors)} source error(s)")

    return ToolResult(
        tool="parallel_search",
        summary="; ".join(summary_bits),
        data={
            "queries": args.queries,
            "sources_searched": searched,
            "result_count": len(ranked),
            "results_by_source": by_source,
            "results": [hit.to_dict() for hit in ranked],
            "errors": errors,
            "duration_s": round(time.time() - started, 3),
        },
        evidence=evidence,
        truncated=len(hits) > len(ranked),
    )


__all__ = [
    "SOURCE_TOOL_CANDIDATES",
    "SOURCE_WEIGHT",
    "ParallelSearchInput",
    "SearchResult",
    "parallel_search",
    "rank_and_dedup",
]
