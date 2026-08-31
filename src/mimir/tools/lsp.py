"""Language server tools: exact answers where regex was guessing.

``find_symbol`` and ``find_references`` are ripgrep plus a heuristic that
decides whether a matching line looks like a definition. That works until a
name is shadowed, re-exported, imported under an alias, defined in a string, or
simply common. A language server resolves the same questions from a parsed,
type-aware model of the project and is right by construction.

Every tool here follows the same chain, and no step in it is a guess:

    document symbols  ->  exact position of the name
                      ->  server query
                      ->  location, signature or diagnostic

ADR-003 section 4.7: where a problem has a reliable algorithmic solution, use
it rather than asking a language model to imitate one. Section 4.2 makes the
consequence explicit - a server answer outranks anything the model believes
about the same symbol.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel, Field

from mimir.lsp.client import LspClient, LspUnavailable, get_client
from mimir.lsp.servers import SERVERS, available_servers, server_for
from mimir.models.evidence import Citation, Evidence, EvidenceKind, SourceType
from mimir.safety.risk import RiskClass
from mimir.tools.base import Capability, ToolContext, ToolError, ToolResult, tool
from mimir.tools.repo import _target_repos

MAX_RESULTS = 60


class LspStatusInput(BaseModel):
    pass


class SymbolQuery(BaseModel):
    path: str = Field(
        description="Repository-relative path to the file, for example src/mimir/verify/claims.py"
    )
    symbol: str = Field(
        description="Exact name of the function, class, method or variable to resolve."
    )
    repo: str | None = None


class FileQuery(BaseModel):
    path: str = Field(description="Repository-relative path to the file.")
    repo: str | None = None


async def _resolve(ctx: ToolContext, repo: str | None, rel: str) -> tuple[Path, Path]:
    """Return (repository root, absolute file path), refusing to leave the repo."""
    repos = await _target_repos(ctx, repo)
    for candidate in repos:
        root = Path(candidate.root).resolve()
        target = (root / rel).resolve()
        if not target.is_relative_to(root):
            raise ToolError(
                f"{rel} resolves outside the repository", code="invalid_arguments"
            )
        if target.is_file():
            return root, target
    names = ", ".join(r.name for r in repos)
    raise ToolError(f"{rel} not found in {names}", code="not_found")


def _client(target: Path, root: Path) -> LspClient:
    try:
        return get_client(target, root)
    except LspUnavailable as exc:
        # Distinguished from an empty result on purpose. "No server installed"
        # must never read as "this symbol does not exist".
        raise ToolError(str(exc), code="unavailable") from exc


def _position(client: LspClient, target: Path, symbol: str) -> tuple[int, int]:
    """Exact position of ``symbol``, from the server's own symbol table.

    The line comes from the server. The column is the offset of the name within
    that line, because a server returning SymbolInformation reports the range
    of the whole definition - column 0, the ``def`` or ``class`` keyword - and
    a position query there resolves nothing. Locating the name inside a line
    the server already identified keeps the answer anchored to the parse rather
    than to a text search over the file.
    """
    symbols = client.document_symbols(target)
    matches = [e for e in symbols if e.name == symbol]
    if not matches:
        known = sorted({e.name for e in symbols})[:12]
        raise ToolError(
            f"{symbol!r} is not a symbol in {target.name}. "
            f"Symbols found: {', '.join(known) or 'none'}",
            code="not_found",
        )

    entry = matches[0]
    line = entry.location.line
    try:
        text = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return line, entry.location.column
    if 0 < line <= len(text):
        column = text[line - 1].find(symbol)
        if column >= 0:
            return line, column
    return line, entry.location.column


def _evidence(claim: str, path: str, line: int, excerpt: str, root: Path) -> Evidence:
    rel = str(Path(path).resolve()).replace(str(root) + "/", "")
    return Evidence(
        claim=claim,
        kind=EvidenceKind.OBSERVED,
        source_type=SourceType.REPOSITORY,
        source_id=rel,
        excerpt=excerpt[:600],
        confidence=0.95,
        collected_by="language_server",
        citations=[
            Citation(source_type=SourceType.REPOSITORY, locator=rel, path=rel, start_line=line)
        ],
    )


@tool(
    "lsp_status",
    description=(
        "List which language servers are installed and what they cover. Call this when a "
        "language-server tool reports 'unavailable', to see what is missing and how to "
        "install it."
    ),
    capability=Capability.REPOSITORY,
    risk=RiskClass.R0,
    tags=("repository", "lsp", "discovery"),
)
async def lsp_status(args: LspStatusInput, ctx: ToolContext) -> ToolResult:
    installed = {s.binary for s in available_servers()}
    rows = [
        {
            "language": spec.language,
            "server": spec.binary,
            "installed": spec.binary in installed,
            "extensions": list(spec.extensions),
            "install": spec.install_hint,
        }
        for spec in SERVERS
    ]
    ready = [r["language"] for r in rows if r["installed"]]
    return ToolResult(
        tool="lsp_status",
        summary=(
            f"{len(ready)} language server(s) available: {', '.join(sorted(set(ready)))}"
            if ready
            else "no language servers installed; symbol queries fall back to text search"
        ),
        data={"servers": rows},
    )


@tool(
    "lsp_definition",
    description=(
        "Resolve where a symbol is defined, exactly, using the language server rather than "
        "a text search. Prefer this over find_symbol when the language has a server "
        "available: it is correct for shadowed names, aliased imports and re-exports."
    ),
    capability=Capability.REPOSITORY,
    risk=RiskClass.R1,
    tags=("repository", "lsp", "symbols"),
)
async def lsp_definition(args: SymbolQuery, ctx: ToolContext) -> ToolResult:
    root, target = await _resolve(ctx, args.repo, args.path)

    def work() -> ToolResult:
        client = _client(target, root)
        line, column = _position(client, target, args.symbol)
        found = client.definition(target, line, column)
        evidence = [
            _evidence(
                f"{args.symbol} is defined at {loc.render()}",
                loc.path, loc.line, f"{args.symbol} defined here", root,
            )
            for loc in found[:MAX_RESULTS]
        ]
        return ToolResult(
            tool="lsp_definition",
            summary=(
                f"{args.symbol} defined at " + ", ".join(x.render() for x in found[:3])
                if found
                else f"the server resolved no definition for {args.symbol}"
            ),
            data={
                "symbol": args.symbol,
                "definitions": [
                    {"path": x.path, "line": x.line, "column": x.column} for x in found
                ],
                "server": client.spec.binary,
            },
            evidence=evidence,
        )

    return await asyncio.to_thread(work)


@tool(
    "lsp_references",
    description=(
        "Find every use of a symbol, exactly, using the language server. Unlike a text "
        "search this excludes same-named symbols in other scopes and includes uses through "
        "aliases. Use it to judge blast radius before proposing a change."
    ),
    capability=Capability.REPOSITORY,
    risk=RiskClass.R1,
    tags=("repository", "lsp", "symbols", "impact"),
)
async def lsp_references(args: SymbolQuery, ctx: ToolContext) -> ToolResult:
    root, target = await _resolve(ctx, args.repo, args.path)

    def work() -> ToolResult:
        client = _client(target, root)
        line, column = _position(client, target, args.symbol)
        found = client.references(target, line, column)
        by_file: dict[str, int] = {}
        for loc in found:
            rel = loc.path.replace(str(root) + "/", "")
            by_file[rel] = by_file.get(rel, 0) + 1
        evidence = [
            _evidence(
                f"{args.symbol} is referenced {count} time(s) in {rel}",
                str(root / rel), 0, f"{count} reference(s)", root,
            )
            for rel, count in sorted(by_file.items(), key=lambda kv: -kv[1])[:12]
        ]
        return ToolResult(
            tool="lsp_references",
            summary=(
                f"{len(found)} reference(s) to {args.symbol} across {len(by_file)} file(s)"
            ),
            data={
                "symbol": args.symbol,
                "total": len(found),
                "by_file": by_file,
                "references": [
                    {"path": x.path, "line": x.line} for x in found[:MAX_RESULTS]
                ],
                "server": client.spec.binary,
            },
            evidence=evidence,
            truncated=len(found) > MAX_RESULTS,
        )

    return await asyncio.to_thread(work)


@tool(
    "lsp_hover",
    description=(
        "Get the resolved signature, type and documentation for a symbol. This is the "
        "language server's own type inference, not a guess from the surrounding text."
    ),
    capability=Capability.REPOSITORY,
    risk=RiskClass.R1,
    tags=("repository", "lsp", "types"),
)
async def lsp_hover(args: SymbolQuery, ctx: ToolContext) -> ToolResult:
    root, target = await _resolve(ctx, args.repo, args.path)

    def work() -> ToolResult:
        client = _client(target, root)
        line, column = _position(client, target, args.symbol)
        text = client.hover(target, line, column)
        return ToolResult(
            tool="lsp_hover",
            summary=(text.splitlines()[0][:200] if text else f"no hover for {args.symbol}"),
            data={"symbol": args.symbol, "hover": text, "server": client.spec.binary},
            evidence=(
                [_evidence(f"{args.symbol} has signature: {text.splitlines()[0][:200]}",
                           str(target), line, text, root)]
                if text
                else []
            ),
        )

    return await asyncio.to_thread(work)


@tool(
    "lsp_symbols",
    description=(
        "List every symbol a file defines, with kind and line, from the language server's "
        "parse. Use it to understand a file's structure without reading the whole thing."
    ),
    capability=Capability.REPOSITORY,
    risk=RiskClass.R1,
    tags=("repository", "lsp", "structure"),
)
async def lsp_symbols(args: FileQuery, ctx: ToolContext) -> ToolResult:
    root, target = await _resolve(ctx, args.repo, args.path)

    def work() -> ToolResult:
        client = _client(target, root)
        found = client.document_symbols(target)
        interesting = [s for s in found if s.kind in ("class", "function", "method")]
        return ToolResult(
            tool="lsp_symbols",
            summary=(
                f"{len(interesting)} definition(s) in {args.path} "
                f"({len(found)} symbols total)"
            ),
            data={
                "path": args.path,
                "symbols": [
                    {
                        "name": s.name,
                        "kind": s.kind,
                        "line": s.location.line,
                        "container": s.container,
                    }
                    for s in interesting[:MAX_RESULTS]
                ],
                "server": client.spec.binary,
            },
            truncated=len(interesting) > MAX_RESULTS,
        )

    return await asyncio.to_thread(work)


@tool(
    "lsp_diagnostics",
    description=(
        "Compiler, type and lint errors the language server reports for a file. This is "
        "ground truth about whether the code is valid, not an opinion about whether it "
        "looks right."
    ),
    capability=Capability.REPOSITORY,
    risk=RiskClass.R1,
    tags=("repository", "lsp", "verification"),
)
async def lsp_diagnostics(args: FileQuery, ctx: ToolContext) -> ToolResult:
    root, target = await _resolve(ctx, args.repo, args.path)

    def work() -> ToolResult:
        client = _client(target, root)
        items = client.diagnostics(target)
        errors = [d for d in items if d["severity"] == "error"]
        return ToolResult(
            tool="lsp_diagnostics",
            summary=(
                f"{len(errors)} error(s), {len(items) - len(errors)} other diagnostic(s) "
                f"in {args.path}"
                if items
                else f"no diagnostics reported for {args.path}"
            ),
            data={"path": args.path, "diagnostics": items[:MAX_RESULTS],
                  "server": client.spec.binary},
            evidence=[
                _evidence(
                    f"{args.path}:{d['line']} reports {d['severity']}: {d['message'][:160]}",
                    str(target), d["line"], d["message"], root,
                )
                for d in errors[:10]
            ],
        )

    return await asyncio.to_thread(work)


__all__ = [
    "lsp_definition",
    "lsp_diagnostics",
    "lsp_hover",
    "lsp_references",
    "lsp_status",
    "lsp_symbols",
    "server_for",
]
