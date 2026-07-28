"""Memory tools (ADR 11, G5).

The agent-facing surface over :mod:`mimir.knowledge`. Two properties matter here
more than convenience:

* Retrieved notes carry their freshness and verification status into the model
  context. A stale runbook is returned and labelled stale, not hidden, because
  ADR R2's mitigation is visible metadata rather than suppression.
* Nothing is promoted into trusted memory by a tool call. ``propose_memory_note``
  creates a proposal; ``promote_memory_note`` refuses to write to ``stable/`` or
  ``runbooks/`` without an explicit approval flag (ADR 11.6, NG4).

Memory documents are locally authored, but they are still content that ends up
in a prompt, so their bodies are wrapped as untrusted data (ADR 13.5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mimir.knowledge.importers import ConversationImporter
from mimir.knowledge.index import get_knowledge_index
from mimir.knowledge.promotion import MemoryPromoter, render_incident_body
from mimir.knowledge.retrieval import MemoryRetriever, RetrievalResult, build_filters
from mimir.knowledge.store import get_knowledge_store
from mimir.logging import get_logger
from mimir.models.command import RiskClass
from mimir.models.evidence import SourceType
from mimir.models.state import MemoryProposal
from mimir.safety.injection import wrap_untrusted
from mimir.tools.base import Capability, ToolContext, ToolError, ToolResult, tool

log = get_logger(__name__)


def _retriever(ctx: ToolContext) -> MemoryRetriever:
    return MemoryRetriever(get_knowledge_index(ctx.settings), settings=ctx.settings)


def _promoter(ctx: ToolContext) -> MemoryPromoter:
    return MemoryPromoter(get_knowledge_store(ctx.settings), settings=ctx.settings,
                          hooks=ctx.hooks)


def _result_payload(result: RetrievalResult, max_chars: int) -> dict[str, Any]:
    return {
        "query": result.query,
        "results": [chunk.to_payload(max_chars) for chunk in result.chunks],
        "conflicts": [conflict.render() for conflict in result.conflicts],
        "candidates_considered": result.candidates_considered,
        "semantic_used": result.semantic_used,
        "filters": result.filters_applied,
    }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class SearchMemoryInput(BaseModel):
    query: str = Field(description="What you want to know. Natural language works.")
    limit: int = Field(default=6, ge=1, le=25, description="Maximum chunks to return.")
    category: str | None = Field(
        default=None,
        description="Restrict to a category, for example 'runbooks/kubernetes'.",
    )
    service: str | None = Field(default=None, description="Restrict to one service.")
    environment: str | None = Field(default=None, description="Restrict to one environment.")
    tags: list[str] = Field(default_factory=list, description="Restrict to documents with tags.")
    layers: list[str] = Field(
        default_factory=list,
        description="Restrict to memory layers: stable, runbooks, history, imports.",
    )
    include_stale: bool = Field(
        default=True,
        description="Keep documents past their freshness window. They are marked stale.",
    )


@tool(
    "search_memory",
    description=(
        "Search curated operational memory: runbooks, service notes, environment "
        "conventions, past incidents, and imported knowledge. Returns ranked excerpts "
        "with their freshness and verification status. Prefer live evidence over "
        "anything returned here."
    ),
    capability=Capability.MEMORY,
    risk=RiskClass.R1,
)
async def search_memory(args: SearchMemoryInput, ctx: ToolContext) -> ToolResult:
    filters = build_filters(
        category=args.category,
        service=args.service,
        environment=args.environment,
        tags=args.tags or None,
        layers=args.layers or None,
    )
    result = _retriever(ctx).search(
        args.query,
        top_k=args.limit,
        filters=filters,
        include_stale=args.include_stale,
    )
    max_chars = ctx.settings.knowledge.max_snippet_chars
    evidence = result.to_evidence()

    if not result.chunks:
        return ToolResult(
            ok=True,
            tool="search_memory",
            summary=(
                f"no curated memory matched '{args.query}'. "
                "Nothing has been written on this yet, so rely on live evidence."
            ),
            data=_result_payload(result, max_chars),
        )

    lines = [result.summary()]
    for chunk in result.chunks:
        markers = f" [{', '.join(chunk.markers())}]" if chunk.markers() else ""
        lines.append(f"  {chunk.citation.render()}{markers} {chunk.heading_path or chunk.title}")
    if result.conflicts:
        lines.append(f"  {len(result.conflicts)} conflict(s) between notes; both sides kept")

    return ToolResult(
        ok=True,
        tool="search_memory",
        summary="\n".join(lines),
        data=_result_payload(result, max_chars),
        evidence=evidence,
    )


class ReadMemoryInput(BaseModel):
    document_id: str = Field(description="Document id as returned by search_memory.")
    max_chars: int = Field(default=6000, ge=200, le=40000)


@tool(
    "read_memory_document",
    description=(
        "Read a full curated memory document by id. Use search_memory first to find "
        "the id; read the whole document only when the excerpt was not enough."
    ),
    capability=Capability.MEMORY,
    risk=RiskClass.R1,
)
async def read_memory_document(args: ReadMemoryInput, ctx: ToolContext) -> ToolResult:
    store = get_knowledge_store(ctx.settings)
    document = store.get(args.document_id)
    if document is None:
        raise ToolError(
            f"no memory document with id '{args.document_id}'", code="not_found"
        )
    freshness = document.freshness(ctx.settings.knowledge.stale_after_days)
    body = wrap_untrusted(
        document.body[: args.max_chars],
        source_type=SourceType.RUNBOOK,
        source_id=args.document_id,
        note=(
            f"freshness={freshness.value} "
            f"verification={document.metadata.verification_status} "
            f"last_verified={document.metadata.last_verified or 'never'}"
        ),
    )
    return ToolResult(
        ok=True,
        tool="read_memory_document",
        summary=(
            f"{document.metadata.title} ({args.document_id}) "
            f"freshness={freshness.value} verification={document.metadata.verification_status}"
        ),
        data={
            "document_id": args.document_id,
            "title": document.metadata.title,
            "category": document.metadata.category,
            "freshness": freshness.value,
            "verification_status": str(document.metadata.verification_status),
            "last_verified": str(document.metadata.last_verified or ""),
            "tags": list(document.metadata.tags),
            "content": body,
        },
        evidence=[_document_evidence(document, freshness)],
    )


def _document_evidence(document: Any, freshness: Any) -> Any:
    from mimir.models.evidence import Evidence, EvidenceKind

    return Evidence(
        claim=f"memory document: {document.metadata.title}",
        kind=EvidenceKind.INFERRED,
        source_type=SourceType.RUNBOOK,
        source_id=document.metadata.title,
        excerpt=document.body[:600],
        citations=[document.citation()],
        freshness=freshness,
        confidence=0.5,
        collected_by="memory_curator",
    )


class ListMemoryInput(BaseModel):
    layer: str | None = Field(
        default=None,
        description=(
            "stable, runbooks, history/incidents, history/investigations, or imports."
        ),
    )
    category: str | None = None
    service: str | None = None
    environment: str | None = None
    tag: str | None = None
    limit: int = Field(default=50, ge=1, le=500)


@tool(
    "list_memory",
    description="List curated memory documents with their freshness, to see what exists.",
    capability=Capability.MEMORY,
    risk=RiskClass.R1,
)
async def list_memory(args: ListMemoryInput, ctx: ToolContext) -> ToolResult:
    from mimir.knowledge.store import MemoryLayer

    store = get_knowledge_store(ctx.settings)
    layer = None
    if args.layer:
        try:
            layer = MemoryLayer(args.layer)
        except ValueError:
            raise ToolError(
                f"unknown layer '{args.layer}'; expected one of: "
                + ", ".join(m.value for m in MemoryLayer),
                code="invalid_arguments",
            ) from None

    documents = store.list(
        layer=layer,
        category=args.category,
        service=args.service,
        environment=args.environment,
        tag=args.tag,
    )[: args.limit]
    stale_days = ctx.settings.knowledge.stale_after_days
    rows = [
        {
            "document_id": store.doc_id_for(d.path),
            "title": d.metadata.title,
            "category": d.metadata.category,
            "freshness": d.freshness(stale_days).value,
            "verification_status": str(d.metadata.verification_status),
            "tags": list(d.metadata.tags),
        }
        for d in documents
    ]
    return ToolResult(
        ok=True,
        tool="list_memory",
        summary=f"{len(rows)} memory document(s)",
        data={"documents": rows},
    )


# ---------------------------------------------------------------------------
# Promotion (ADR 11.6)
# ---------------------------------------------------------------------------


class ProposeNoteInput(BaseModel):
    title: str
    body: str = Field(description="Markdown body. Include the evidence, not a transcript.")
    category: str = Field(
        default="history/investigations",
        description=(
            "Target category. Only history/investigations accepts unverified notes; "
            "stable/ and runbooks/ require verification and explicit approval."
        ),
    )
    sources: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    verification_status: str = Field(default="unverified")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    supersedes: str | None = None


@tool(
    "propose_memory_note",
    description=(
        "Propose a note for curated memory. This does NOT write anything. It returns a "
        "proposal that a human reviews. Use it at the end of an investigation to capture "
        "what is worth keeping."
    ),
    capability=Capability.MEMORY,
    risk=RiskClass.R1,
)
async def propose_memory_note(args: ProposeNoteInput, ctx: ToolContext) -> ToolResult:
    promoter = _promoter(ctx)
    proposal = promoter.propose(
        title=args.title,
        body=args.body,
        category=args.category,
        sources=args.sources,
        tags=args.tags,
        verification_status=args.verification_status,
        confidence=args.confidence,
        supersedes=args.supersedes,
    )
    review = promoter.review(proposal, approved=False)
    return ToolResult(
        ok=True,
        tool="propose_memory_note",
        summary=(
            f"proposal '{proposal.title}' for {review.category}. {review.summary()}"
        ),
        data={
            "proposal": proposal.model_dump(mode="json"),
            "target_layer": review.target_layer.value,
            "target_document_id": review.target_doc_id,
            "requires_approval": review.requires_approval,
            "blocking_reasons": review.blocking_reasons,
            "warnings": review.warnings,
            "downgrade_available": review.downgrade_available,
        },
    )


class PromoteNoteInput(BaseModel):
    proposal: dict[str, Any] = Field(description="The proposal object from propose_memory_note.")
    approved: bool = Field(
        default=False,
        description=(
            "Must be true, and must reflect a real human decision, before a note can "
            "reach stable/ or runbooks/."
        ),
    )
    approved_by: str = ""
    allow_downgrade: bool = Field(
        default=False,
        description="Store in history/investigations instead when approval is missing.",
    )
    owner: str | None = None
    service: str | None = None
    environment: str | None = None
    expires_after: str | None = None


@tool(
    "promote_memory_note",
    description=(
        "Write a reviewed proposal into curated memory. Refuses to write to stable/ or "
        "runbooks/ without explicit approval (ADR 11.6). Unverified notes may only land "
        "in history/investigations."
    ),
    capability=Capability.MEMORY,
    risk=RiskClass.R1,
    mutating=True,
)
async def promote_memory_note(args: PromoteNoteInput, ctx: ToolContext) -> ToolResult:
    try:
        proposal = MemoryProposal.model_validate(args.proposal)
    except Exception as exc:
        raise ToolError(f"invalid proposal payload: {exc}", code="invalid_arguments") from exc

    outcome = _promoter(ctx).promote(
        proposal,
        approved=args.approved,
        approved_by=args.approved_by,
        allow_downgrade=args.allow_downgrade,
        owner=args.owner,
        service=args.service,
        environment=args.environment,
        expires_after=args.expires_after,
    )
    if not outcome.stored:
        return ToolResult(
            ok=False,
            tool="promote_memory_note",
            summary=outcome.summary(),
            error=outcome.reason or "promotion refused",
            error_code="promotion_refused",
            data={
                "blocking_reasons": outcome.review.blocking_reasons,
                "downgrade_available": outcome.review.downgrade_available,
            },
        )
    # A new document changes what retrieval should see, so refresh the index now
    # rather than leaving the next search to miss it.
    get_knowledge_index(ctx.settings).reindex()
    return ToolResult(
        ok=True,
        tool="promote_memory_note",
        summary=outcome.summary(),
        data={
            "document_id": outcome.doc_id,
            "path": outcome.path,
            "layer": outcome.layer.value if outcome.layer else None,
            "downgraded": outcome.downgraded,
        },
    )


class RecordIncidentInput(BaseModel):
    title: str
    occurred_on: str = Field(description="ISO date, YYYY-MM-DD.")
    symptoms: str = ""
    timeline: str = ""
    evidence: str = ""
    root_cause: str = ""
    resolution: str = ""
    lessons: str = ""
    applicability_limits: str = Field(
        default="",
        description="Where this incident's conclusions do NOT apply. Prevents overreach later.",
    )
    services: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    verified: bool = False


@tool(
    "record_incident",
    description=(
        "Propose a structured historical incident record (ADR 11.1 M3). Returns a "
        "proposal; it is not stored until promoted."
    ),
    capability=Capability.MEMORY,
    risk=RiskClass.R1,
)
async def record_incident(args: RecordIncidentInput, ctx: ToolContext) -> ToolResult:
    body = render_incident_body(
        title=args.title,
        occurred_on=args.occurred_on,
        symptoms=args.symptoms,
        timeline=args.timeline,
        evidence=args.evidence,
        root_cause=args.root_cause,
        resolution=args.resolution,
        lessons=args.lessons,
        applicability=args.applicability_limits,
    )
    promoter = _promoter(ctx)
    proposal = promoter.propose(
        title=args.title,
        body=body,
        category="history/incidents",
        tags=[*args.tags, *args.services],
        verification_status="verified" if args.verified else "unverified",
        confidence=0.7 if args.verified else 0.4,
    )
    review = promoter.review(proposal, approved=False)
    return ToolResult(
        ok=True,
        tool="record_incident",
        summary=f"incident record proposed: {args.title}. {review.summary()}",
        data={"proposal": proposal.model_dump(mode="json")},
    )


class SimilarIncidentInput(BaseModel):
    symptoms: str = Field(description="Describe the current symptoms in plain language.")
    limit: int = Field(default=5, ge=1, le=20)


@tool(
    "find_similar_incidents",
    description=(
        "Find past incidents whose symptoms resemble the current one. Read the "
        "applicability limits before reusing a conclusion; a matching symptom is not a "
        "matching cause."
    ),
    capability=Capability.MEMORY,
    risk=RiskClass.R1,
)
async def find_similar_incidents(args: SimilarIncidentInput, ctx: ToolContext) -> ToolResult:
    result = _retriever(ctx).search(
        args.symptoms,
        top_k=args.limit,
        filters=build_filters(layers=["history/incidents", "history/investigations"]),
    )
    payload = _result_payload(result, ctx.settings.knowledge.max_snippet_chars)
    if not result.chunks:
        return ToolResult(
            ok=True,
            tool="find_similar_incidents",
            summary="no past incident resembles these symptoms",
            data=payload,
        )
    lines = ["past incidents with similar symptoms (similar symptom is not similar cause):"]
    lines.extend(
        f"  {c.citation.render()} [{c.freshness.value}] {c.title}" for c in result.chunks
    )
    return ToolResult(
        ok=True,
        tool="find_similar_incidents",
        summary="\n".join(lines),
        data=payload,
        evidence=result.to_evidence(),
    )


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------


class ReindexInput(BaseModel):
    force: bool = Field(default=False, description="Rebuild every chunk, not just changed files.")
    embed: bool = Field(default=True, description="Recompute embeddings as well.")


@tool(
    "reindex_memory",
    description="Rebuild the memory search index after files changed on disk.",
    capability=Capability.MEMORY,
    risk=RiskClass.R1,
)
async def reindex_memory(args: ReindexInput, ctx: ToolContext) -> ToolResult:
    stats = get_knowledge_index(ctx.settings).reindex(force=args.force, embed=args.embed)
    return ToolResult(
        ok=True,
        tool="reindex_memory",
        summary=stats.summary(),
        data={"stats": stats.__dict__},
    )


class ImportKnowledgeInput(BaseModel):
    path: str = Field(description="File or directory to import.")
    tool_name: str = Field(
        default="",
        description="claude, codex, chatgpt, or other. Inferred from the path when empty.",
    )
    limit: int | None = Field(default=None, ge=1, le=2000)


@tool(
    "import_knowledge",
    description=(
        "Import prior Claude, Codex, or ChatGPT material into imports/ as untrusted "
        "candidate memory (ADR 11.5). Content is summarised, secrets are stripped, and "
        "everything lands unverified. It never reaches stable memory this way."
    ),
    capability=Capability.MEMORY,
    risk=RiskClass.R1,
    mutating=True,
)
async def import_knowledge(args: ImportKnowledgeInput, ctx: ToolContext) -> ToolResult:
    path = Path(args.path).expanduser()
    if not path.exists():
        raise ToolError(f"path does not exist: {path}", code="not_found")
    importer = ConversationImporter(get_knowledge_store(ctx.settings), settings=ctx.settings)
    result = importer.import_path(path, tool=args.tool_name, limit=args.limit)
    if result.notes_written:
        get_knowledge_index(ctx.settings).reindex()
    return ToolResult(
        ok=True,
        tool="import_knowledge",
        summary=result.summary(),
        data={
            "notes_written": result.notes_written,
            "conversations_seen": result.conversations_seen,
            "skipped": result.skipped,
            "document_ids": result.doc_ids[:50],
            "errors": result.errors[:20],
        },
    )
