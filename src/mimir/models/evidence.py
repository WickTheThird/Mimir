"""Evidence model (ADR 11.4, 12)."""

from __future__ import annotations

import hashlib
import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, computed_field


class SourceType(StrEnum):
    """Ordered by ADR 11.4 trust ranking, most trusted first."""

    COMMAND_OUTPUT = "command_output"
    REPOSITORY = "repository"
    DEPLOYMENT_STATE = "deployment_state"
    RUNBOOK = "runbook"
    HISTORICAL_INCIDENT = "historical_incident"
    IMPORTED_MEMORY = "imported_memory"
    MODEL_KNOWLEDGE = "model_knowledge"
    # Not part of the trust ladder; ranked explicitly at the bottom because it
    WEB = "web"
    USER_PROVIDED = "user_provided"


TRUST_ORDER: dict[SourceType, int] = {
    SourceType.COMMAND_OUTPUT: 0,
    SourceType.REPOSITORY: 1,
    SourceType.DEPLOYMENT_STATE: 2,
    SourceType.RUNBOOK: 3,
    SourceType.HISTORICAL_INCIDENT: 4,
    SourceType.IMPORTED_MEMORY: 5,
    SourceType.USER_PROVIDED: 6,
    SourceType.WEB: 7,
    SourceType.MODEL_KNOWLEDGE: 8,
}

#: Source types whose content must be treated as untrusted data and never as
UNTRUSTED_SOURCES: frozenset[SourceType] = frozenset(
    {
        SourceType.WEB,
        SourceType.REPOSITORY,
        SourceType.COMMAND_OUTPUT,
        SourceType.IMPORTED_MEMORY,
        SourceType.RUNBOOK,
        SourceType.HISTORICAL_INCIDENT,
    }
)


class EvidenceKind(StrEnum):
    """Whether the item is a direct observation or a derived claim."""

    OBSERVED = "observed"
    INFERRED = "inferred"
    HYPOTHESIS = "hypothesis"


class Freshness(StrEnum):
    LIVE = "live"  # collected during this session
    RECENT = "recent"  # verified within the freshness window
    STALE = "stale"  # older than the configured window
    UNKNOWN = "unknown"


class Citation(BaseModel):
    """A precise pointer back into a source."""

    source_type: SourceType
    locator: str
    """Repo-relative path, URL, command string, or memory document id."""

    repo: str | None = None
    path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    url: str | None = None
    retrieved_at: float | None = None
    title: str | None = None

    def render(self) -> str:
        if self.path and self.start_line:
            span = (
                f"{self.start_line}-{self.end_line}"
                if self.end_line and self.end_line != self.start_line
                else str(self.start_line)
            )
            prefix = f"{self.repo}:" if self.repo else ""
            return f"{prefix}{self.path}:{span}"
        if self.url:
            return (self.title and f"{self.title} <{self.url}>") or self.url
        return self.locator


class Evidence(BaseModel):
    """A single piece of support for or against a claim (ADR 12)."""

    id: str = ""
    claim: str
    """The claim this evidence supports or refutes, stated plainly."""

    kind: EvidenceKind = EvidenceKind.OBSERVED
    source_type: SourceType
    source_id: str
    """Stable identifier: command id, file path, document id, or URL."""

    excerpt: str = ""
    """Exact excerpt. Keep it short; the artifact store holds the full output."""

    structured: dict[str, Any] | None = None
    citations: list[Citation] = Field(default_factory=list)
    collected_at: float = Field(default_factory=time.time)
    freshness: Freshness = Freshness.LIVE
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    supports: bool = True
    """False when the evidence contradicts the claim."""

    collected_by: str = "mimir"
    """Specialist or helper that produced the item."""

    artifact_ref: str | None = None
    """Pointer into the artifact store for the full untruncated output."""

    tags: list[str] = Field(default_factory=list)

    def model_post_init(self, _context: Any) -> None:
        if not self.id:
            digest = hashlib.sha256(
                f"{self.source_type}|{self.source_id}|{self.claim}|{self.excerpt[:512]}".encode()
            ).hexdigest()[:16]
            object.__setattr__(self, "id", f"ev_{digest}")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def trust_rank(self) -> int:
        return TRUST_ORDER.get(self.source_type, 99)

    @property
    def is_untrusted_content(self) -> bool:
        return self.source_type in UNTRUSTED_SOURCES

    def render(self) -> str:
        marker = "+" if self.supports else "-"
        cites = "; ".join(c.render() for c in self.citations) or self.source_id
        body = self.excerpt.strip()
        if len(body) > 600:
            body = body[:600] + " ...[truncated]"
        return f"[{marker}{self.kind.value[:3]}] {self.claim}\n    source: {cites}\n    {body}"


def rank_evidence(items: list[Evidence]) -> list[Evidence]:
    """Sort by ADR 11.4 trust order, then freshness, then confidence."""

    freshness_rank = {
        Freshness.LIVE: 0,
        Freshness.RECENT: 1,
        Freshness.UNKNOWN: 2,
        Freshness.STALE: 3,
    }
    return sorted(
        items,
        key=lambda e: (
            e.trust_rank,
            freshness_rank.get(e.freshness, 9),
            -e.confidence,
        ),
    )
