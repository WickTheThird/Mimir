"""Hybrid memory retrieval with trust ordering (ADR 11.4, 11.5, G5, R2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from mimir.config import Settings, get_settings
from mimir.knowledge.index import ChunkHit, KnowledgeIndex, MemoryFilters
from mimir.knowledge.store import (
    Confidence,
    KnowledgeStore,
    MemoryDocument,
    MemoryLayer,
    VerificationStatus,
)
from mimir.logging import get_logger
from mimir.models.evidence import (
    TRUST_ORDER,
    Citation,
    Evidence,
    EvidenceKind,
    Freshness,
    SourceType,
)

log = get_logger(__name__)

DEFAULT_KEYWORD_WEIGHT = 0.6
DEFAULT_SEMANTIC_WEIGHT = 0.4
# : How far down the ladder each trust rung costs.
TRUST_STEP = 0.07
TRUST_FLOOR = 0.4

FRESHNESS_MULTIPLIER: dict[Freshness, float] = {
    Freshness.LIVE: 1.0,
    Freshness.RECENT: 1.0,
    Freshness.UNKNOWN: 0.85,
    Freshness.STALE: 0.55,
}

VERIFICATION_MULTIPLIER: dict[VerificationStatus, float] = {
    VerificationStatus.VERIFIED: 1.0,
    VerificationStatus.USER_CONFIRMED: 0.95,
    VerificationStatus.UNVERIFIED: 0.85,
    VerificationStatus.CONTRADICTED: 0.5,
}

_TITLE_STOPWORDS = frozenset({"the", "a", "an", "for", "and", "of", "in", "to", "with", "on"})


def _title_tokens(title: str) -> frozenset[str]:
    parts = [p.strip("-_.,:") for p in title.lower().split()]
    return frozenset(p for p in parts if p and p not in _TITLE_STOPWORDS and len(p) > 2)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass(slots=True)
class MemoryConflict:
    """Two documents that disagree, or may disagree, and both stay visible."""

    doc_a: str
    doc_b: str
    reason: str
    """One of ``supersedes``, ``declared_contradiction``, ``same_topic_divergent``."""

    detail: str
    resolved: bool = False

    def render(self) -> str:
        return f"[conflict:{self.reason}] {self.doc_a} vs {self.doc_b}: {self.detail}"


@dataclass(slots=True)
class RetrievedChunk:
    """One ranked chunk plus everything needed to cite and judge it."""

    chunk_id: str
    doc_id: str
    title: str
    path: str
    heading_path: str
    text: str
    layer: MemoryLayer
    source_type: SourceType
    freshness: Freshness
    verification_status: VerificationStatus
    confidence: Confidence
    category: str
    service: str | None
    environment: str | None
    tags: tuple[str, ...]
    last_verified: date | None
    citation: Citation
    keyword_score: float = 0.0
    semantic_score: float = 0.0
    base_score: float = 0.0
    trust_multiplier: float = 1.0
    freshness_multiplier: float = 1.0
    score: float = 0.0
    start_line: int = 1
    end_line: int = 1
    conflicts: list[MemoryConflict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def stale(self) -> bool:
        return self.freshness is Freshness.STALE

    @property
    def trust_rank(self) -> int:
        return TRUST_ORDER.get(self.source_type, 99)

    def markers(self) -> list[str]:
        out = [f"layer={self.layer.value}", f"freshness={self.freshness.value}"]
        if self.stale:
            verified = self.last_verified.isoformat() if self.last_verified else "never"
            out.append(f"STALE (last_verified={verified}; treat as a lead, verify live)")
        if self.verification_status is not VerificationStatus.VERIFIED:
            out.append(f"verification={self.verification_status.value}")
        if self.conflicts:
            out.append(f"CONFLICT x{len(self.conflicts)}")
        return out

    def render(self, max_chars: int = 1200) -> str:
        body = self.text if len(self.text) <= max_chars else self.text[:max_chars] + " ...[cut]"
        head = f"{self.title}" + (f" > {self.heading_path}" if self.heading_path else "")
        lines = [
            f"### {head}",
            f"id: {self.doc_id}  score: {self.score:.3f}  " + "  ".join(self.markers()),
        ]
        lines.extend(conflict.render() for conflict in self.conflicts)
        lines.extend(f"note: {note}" for note in self.notes)
        lines.append(body)
        return "\n".join(lines)

    def to_evidence(self, claim: str = "") -> Evidence:
        return Evidence(
            claim=claim or f"memory: {self.title}",
            kind=EvidenceKind.INFERRED,
            source_type=self.source_type,
            source_id=self.doc_id,
            excerpt=self.text[:600],
            citations=[self.citation],
            freshness=self.freshness,
            confidence=min(0.95, max(0.05, self.confidence.weight * self.freshness_multiplier)),
            collected_by="memory_curator",
            tags=list(self.tags),
        )

    def to_payload(self, max_chars: int = 1200) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "title": self.title,
            "path": self.path,
            "heading_path": self.heading_path,
            "layer": self.layer.value,
            "source_type": self.source_type.value,
            "trust_rank": self.trust_rank,
            "freshness": self.freshness.value,
            "stale": self.stale,
            "verification_status": self.verification_status.value,
            "confidence": self.confidence.value,
            "service": self.service,
            "environment": self.environment,
            "tags": list(self.tags),
            "last_verified": self.last_verified.isoformat() if self.last_verified else None,
            "score": round(self.score, 4),
            "keyword_score": round(self.keyword_score, 4),
            "semantic_score": round(self.semantic_score, 4),
            "citation": self.citation.render(),
            "conflicts": [
                {"with": c.doc_b if c.doc_a == self.doc_id else c.doc_a,
                 "reason": c.reason,
                 "detail": c.detail}
                for c in self.conflicts
            ],
            "notes": list(self.notes),
            "text": self.text[:max_chars],
        }


@dataclass(slots=True)
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    conflicts: list[MemoryConflict] = field(default_factory=list)
    candidates_considered: int = 0
    semantic_used: bool = False
    keyword_used: bool = True
    filters_applied: dict[str, object] = field(default_factory=dict)

    @property
    def stale_count(self) -> int:
        return sum(1 for c in self.chunks if c.stale)

    def summary(self) -> str:
        parts = [f"{len(self.chunks)} memory chunks for {self.query!r}"]
        if self.stale_count:
            parts.append(f"{self.stale_count} stale (returned with a marker)")
        if self.conflicts:
            parts.append(f"{len(self.conflicts)} unresolved conflict(s)")
        if not self.semantic_used:
            parts.append("keyword only")
        return "; ".join(parts)

    def render(self, max_chars_per_chunk: int = 1200) -> str:
        blocks = [chunk.render(max_chars_per_chunk) for chunk in self.chunks]
        if self.conflicts:
            blocks.append(
                "## Unresolved memory conflicts (ADR 11.5: kept as conflicts, not merged)\n"
                + "\n".join(c.render() for c in self.conflicts)
            )
        return "\n\n".join(blocks)

    def to_evidence(self) -> list[Evidence]:
        return [chunk.to_evidence() for chunk in self.chunks]


class MemoryRetriever:
    """Hybrid retrieval over :class:`~mimir.knowledge.index.KnowledgeIndex`."""

    def __init__(
        self,
        index: KnowledgeIndex | None = None,
        *,
        store: KnowledgeStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.index = index or KnowledgeIndex(store=store, settings=self.settings)
        self.store = store or self.index.store

    # -- scoring helpers -------------------------------------------------

    @staticmethod
    def _normalise(hits: list[ChunkHit]) -> dict[str, float]:
        if not hits:
            return {}
        scores = [h.score for h in hits]
        low, high = min(scores), max(scores)
        if high <= low:
            return {h.chunk_id: 1.0 for h in hits}
        return {h.chunk_id: (h.score - low) / (high - low) for h in hits}

    @staticmethod
    def trust_multiplier(source_type: SourceType, layer_rank: int) -> float:
        rung = TRUST_ORDER.get(source_type, 99)
        base = max(TRUST_FLOOR, 1.0 - TRUST_STEP * min(rung, 8))
        # Small tie-break so stable knowledge edges out a runbook on the same rung.
        return base - 0.005 * layer_rank

    # -- main entry point ------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: MemoryFilters | None = None,
        keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
        include_stale: bool = True,
        today: date | None = None,
    ) -> RetrievalResult:
        limit = top_k or self.settings.knowledge.top_k
        filters = filters or MemoryFilters()
        pool = max(limit * 6, 40)

        keyword_hits = self.index.keyword_search(query, limit=pool, filters=filters)
        semantic_hits = (
            self.index.vector_search(query, limit=pool, filters=filters)
            if self.index.embedder is not None
            else []
        )
        kw_norm = self._normalise(keyword_hits)
        sem_norm = {h.chunk_id: max(0.0, min(1.0, h.score)) for h in semantic_hits}

        if not semantic_hits:
            keyword_weight, semantic_weight = 1.0, 0.0
        elif not keyword_hits:
            keyword_weight, semantic_weight = 0.0, 1.0

        merged: dict[str, ChunkHit] = {h.chunk_id: h for h in semantic_hits}
        merged.update({h.chunk_id: h for h in keyword_hits})
        if not merged:
            return RetrievalResult(
                query=query,
                semantic_used=bool(semantic_hits),
                filters_applied=_filter_payload(filters),
            )

        scored: list[tuple[float, ChunkHit, float, float]] = []
        for chunk_id, hit in merged.items():
            kw = kw_norm.get(chunk_id, 0.0)
            sem = sem_norm.get(chunk_id, 0.0)
            base = keyword_weight * kw + semantic_weight * sem
            scored.append((base, hit, kw, sem))
        scored.sort(key=lambda item: -item[0])
        shortlist = scored[: max(limit * 4, 24)]

        stale_after = self.settings.knowledge.stale_after_days
        documents: dict[str, MemoryDocument] = {}
        for _, hit, _, _ in shortlist:
            if hit.doc_id not in documents:
                doc = self.store.get(hit.doc_id)
                if doc is not None:
                    documents[hit.doc_id] = doc

        results: list[RetrievedChunk] = []
        for base, hit, kw, sem in shortlist:
            doc = documents.get(hit.doc_id)
            if doc is None:
                # Indexed but deleted on disk since the last reindex.
                continue
            meta = doc.metadata
            freshness = doc.freshness(stale_after, today)
            if freshness is Freshness.STALE and not include_stale:
                continue
            trust = self.trust_multiplier(doc.source_type, doc.layer_rank)
            fresh_mult = FRESHNESS_MULTIPLIER.get(freshness, 0.85)
            verification = VERIFICATION_MULTIPLIER.get(meta.verification_status, 0.85)
            confidence_mult = 0.8 + 0.2 * meta.confidence.weight
            final = base * trust * fresh_mult * verification * confidence_mult
            results.append(
                RetrievedChunk(
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    title=doc.title,
                    path=doc.relative_path,
                    heading_path=hit.heading_path,
                    text=hit.text,
                    layer=doc.layer,
                    source_type=doc.source_type,
                    freshness=freshness,
                    verification_status=meta.verification_status,
                    confidence=meta.confidence,
                    category=meta.category,
                    service=meta.service,
                    environment=meta.environment,
                    tags=tuple(meta.tags),
                    last_verified=meta.last_verified,
                    citation=doc.citation(hit.heading_path, hit.start_line),
                    keyword_score=kw,
                    semantic_score=sem,
                    base_score=base,
                    trust_multiplier=trust,
                    freshness_multiplier=fresh_mult,
                    score=final,
                    start_line=hit.start_line,
                    end_line=hit.end_line,
                )
            )

        results.sort(key=lambda c: (-c.score, c.trust_rank, c.doc_id, c.chunk_id))
        top = results[:limit]
        conflicts = self.detect_conflicts(
            [documents[c.doc_id] for c in top if c.doc_id in documents]
        )
        by_doc: dict[str, list[RetrievedChunk]] = {}
        for chunk in top:
            by_doc.setdefault(chunk.doc_id, []).append(chunk)
        for conflict in conflicts:
            for doc_id in (conflict.doc_a, conflict.doc_b):
                for chunk in by_doc.get(doc_id, []):
                    chunk.conflicts.append(conflict)

        return RetrievalResult(
            query=query,
            chunks=top,
            conflicts=conflicts,
            candidates_considered=len(merged),
            semantic_used=bool(semantic_hits),
            keyword_used=bool(keyword_hits),
            filters_applied=_filter_payload(filters),
        )

    # -- ADR 11.5 conflict retention -------------------------------------

    def detect_conflicts(self, documents: list[MemoryDocument]) -> list[MemoryConflict]:
        """Find disagreements among the retrieved set and keep both sides."""
        conflicts: list[MemoryConflict] = []
        seen: set[tuple[str, str]] = set()

        def add(a: str, b: str, reason: str, detail: str) -> None:
            key = (a, b) if a <= b else (b, a)
            if key in seen:
                return
            seen.add(key)
            conflicts.append(MemoryConflict(doc_a=a, doc_b=b, reason=reason, detail=detail))

        index_map = {doc.doc_id: doc for doc in documents}
        superseded = self.index.superseded_by()

        for doc in documents:
            for target in doc.metadata.supersedes:
                key = target.strip().removesuffix(".md")
                if key and key != doc.doc_id:
                    add(
                        doc.doc_id,
                        key,
                        "supersedes",
                        f"{doc.doc_id} declares it supersedes {key}; both are shown until "
                        "the superseded note is retired or verified",
                    )
            for target in doc.metadata.contradicts:
                key = target.strip().removesuffix(".md")
                if key and key != doc.doc_id:
                    add(
                        doc.doc_id,
                        key,
                        "declared_contradiction",
                        f"{doc.doc_id} marks {key} as contradictory; resolve before relying "
                        "on either",
                    )
            # A newer note may supersede this one without being retrieved.
            for replacement in superseded.get(doc.doc_id, []):
                if replacement != doc.doc_id:
                    add(
                        replacement,
                        doc.doc_id,
                        "supersedes",
                        f"{doc.doc_id} is superseded by {replacement}, which this query did "
                        "not retrieve",
                    )

        ordered = list(index_map.values())
        for i, first in enumerate(ordered):
            for second in ordered[i + 1 :]:
                if (first.doc_id, second.doc_id) in seen or (
                    second.doc_id,
                    first.doc_id,
                ) in seen:
                    continue
                service_a = (first.metadata.service or "").lower()
                service_b = (second.metadata.service or "").lower()
                if not service_a or service_a != service_b:
                    continue
                overlap = _jaccard(_title_tokens(first.title), _title_tokens(second.title))
                if overlap < 0.6:
                    continue
                add(
                    first.doc_id,
                    second.doc_id,
                    "same_topic_divergent",
                    f"both cover service {service_a!r} with near-identical titles "
                    f"(title overlap {overlap:.2f}); they may disagree, neither was picked",
                )
        return conflicts


def _filter_payload(filters: MemoryFilters) -> dict[str, object]:
    return {
        "layers": [layer.value for layer in filters.layers],
        "category": filters.category,
        "service": filters.service,
        "environment": filters.environment,
        "tags": list(filters.tags),
    }


def build_filters(
    *,
    category: str | None = None,
    service: str | None = None,
    environment: str | None = None,
    tags: list[str] | None = None,
    layers: list[str] | None = None,
    doc_ids: list[str] | None = None,
) -> MemoryFilters:
    """Turn loose tool arguments into a :class:`MemoryFilters`, ignoring junk."""
    resolved: list[MemoryLayer] = []
    for name in layers or []:
        try:
            resolved.append(MemoryLayer(str(name).strip().strip("/")))
        except ValueError:
            log.info("unknown_memory_layer", layer=name)
    return MemoryFilters(
        layers=tuple(resolved),
        category=category or None,
        service=service or None,
        environment=environment or None,
        tags=tuple(t.lower() for t in (tags or []) if t),
        doc_ids=tuple(doc_ids or ()),
    )
