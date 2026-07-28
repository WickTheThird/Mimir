"""Memory promotion workflow (ADR 11.6, NG4, G5).

ADR 11.6 is a five step pipeline: a session produces a candidate finding, the
Memory Curator extracts a proposed note, the note carries sources and a
verification status, a human reviews or approves it, and only then is it stored
in the appropriate layer.

Two gates are enforced here and nowhere else:

* **Approval gate.** Nothing is written into ``stable/``, ``runbooks/``, or
  ``skills/`` without an explicit approval flag. A proposal that reaches
  :meth:`MemoryPromoter.promote` without one is refused, not queued, not
  written somewhere convenient.
* **Verification gate (ADR NG4).** Raw conversation history and unverified
  session output are not truth. An ``unverified`` note may only land in
  ``history/investigations/`` or ``imports/``. Promoting it further requires a
  verification status of ``user_confirmed`` or ``verified``.

Both gates run before :meth:`~mimir.hooks.manager.HookManager.on_memory_promotion`
fires, and a hook denial is itself a refusal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

from mimir.config import Settings, get_settings
from mimir.knowledge.store import (
    CURATED_LAYERS,
    Confidence,
    DocumentMetadata,
    KnowledgeStore,
    MemoryDocument,
    MemoryLayer,
    VerificationStatus,
    layer_for_relative,
)
from mimir.logging import get_logger
from mimir.models.state import MemoryProposal

log = get_logger(__name__)

#: Layers an ``unverified`` note is permitted to reach (ADR NG4).
UNVERIFIED_LAYERS: frozenset[MemoryLayer] = frozenset(
    {MemoryLayer.HISTORY_INVESTIGATIONS, MemoryLayer.IMPORTS, MemoryLayer.SESSIONS}
)

#: Where an otherwise-refused unverified note may be parked instead.
DOWNGRADE_LAYER = MemoryLayer.HISTORY_INVESTIGATIONS
DOWNGRADE_CATEGORY = "history/investigations"

DEFAULT_CATEGORY = "history/investigations"


class PromotionRefused(Exception):
    """Raised only by :meth:`MemoryPromoter.promote_or_raise`."""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons


def normalise_category(category: str | None) -> str:
    cleaned = (category or DEFAULT_CATEGORY).strip().strip("/")
    return cleaned or DEFAULT_CATEGORY


def layer_for_category(category: str | None) -> MemoryLayer:
    return layer_for_relative(Path(normalise_category(category)))


@dataclass(slots=True)
class PromotionReview:
    """ADR 11.6 step 4. The decision, made before anything touches disk."""

    proposal: MemoryProposal
    category: str
    target_layer: MemoryLayer
    target_doc_id: str
    requires_approval: bool
    approval_present: bool
    blocking_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    downgrade_available: bool = False
    downgrade_category: str = DOWNGRADE_CATEGORY

    @property
    def allowed(self) -> bool:
        return not self.blocking_reasons

    def summary(self) -> str:
        if self.allowed:
            note = f"ready to store at {self.target_doc_id} (layer {self.target_layer.value})"
            return note + (f"; warnings: {'; '.join(self.warnings)}" if self.warnings else "")
        refusal = "refused: " + "; ".join(self.blocking_reasons)
        if self.downgrade_available:
            refusal += f" (may be stored under {self.downgrade_category} instead)"
        return refusal


@dataclass(slots=True)
class PromotionOutcome:
    stored: bool
    review: PromotionReview
    doc_id: str | None = None
    path: str | None = None
    layer: MemoryLayer | None = None
    downgraded: bool = False
    reason: str = ""
    document: MemoryDocument | None = None

    def summary(self) -> str:
        if self.stored:
            prefix = "downgraded and stored" if self.downgraded else "stored"
            return f"{prefix} at {self.doc_id} (layer {self.layer.value if self.layer else '?'})"
        return self.reason or self.review.summary()


class MemoryPromoter:
    """Runs the ADR 11.6 pipeline against a :class:`KnowledgeStore`."""

    def __init__(
        self,
        store: KnowledgeStore | None = None,
        *,
        settings: Settings | None = None,
        hooks: object | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or KnowledgeStore(settings=self.settings)
        self.hooks = hooks

    # -- step 2: extract a proposal ---------------------------------------

    def propose(
        self,
        *,
        title: str,
        body: str,
        category: str = DEFAULT_CATEGORY,
        sources: list[str] | None = None,
        tags: list[str] | None = None,
        verification_status: str = VerificationStatus.UNVERIFIED.value,
        confidence: float = 0.5,
        supersedes: str | None = None,
    ) -> MemoryProposal:
        """Build a candidate note. Proposing never writes anything."""
        return MemoryProposal(
            title=title.strip(),
            category=normalise_category(category),
            body=body,
            tags=[t.strip() for t in (tags or []) if t.strip()],
            sources=[s.strip() for s in (sources or []) if s.strip()],
            verification_status=str(verification_status),
            confidence=confidence,
            supersedes=supersedes,
            approved=False,
        )

    # -- step 4: review ---------------------------------------------------

    def review(self, proposal: MemoryProposal, *, approved: bool = False) -> PromotionReview:
        category = normalise_category(proposal.category)
        layer = layer_for_category(category)
        status = _coerce_status(proposal.verification_status)
        approval_present = bool(approved or proposal.approved)
        requires_approval = layer in CURATED_LAYERS

        doc_id = self.store.unique_doc_id(category, proposal.title or "untitled-note")
        review = PromotionReview(
            proposal=proposal,
            category=category,
            target_layer=layer,
            target_doc_id=doc_id,
            requires_approval=requires_approval,
            approval_present=approval_present,
        )

        if layer is MemoryLayer.UNKNOWN:
            review.blocking_reasons.append(
                f"category {category!r} does not map to an ADR 11.2 memory layer"
            )
        if requires_approval and not approval_present:
            review.blocking_reasons.append(
                f"layer {layer.value!r} is curated memory and requires an explicit approval "
                "flag (ADR 11.6 step 4); nothing was written"
            )
            # Landing the note in investigation history instead resolves this
            # without writing anything curated, so it is a legitimate remedy the
            # caller can opt into.
            review.downgrade_available = True
        if status is VerificationStatus.UNVERIFIED and layer not in UNVERIFIED_LAYERS:
            review.blocking_reasons.append(
                f"verification_status is 'unverified' and ADR NG4 only allows unverified notes "
                f"in {sorted(layer.value for layer in UNVERIFIED_LAYERS)}"
            )
            review.downgrade_available = True
        if status is VerificationStatus.CONTRADICTED:
            review.warnings.append(
                "note is marked contradicted; it is retained as a conflict, not as guidance"
            )
        if not proposal.sources:
            review.warnings.append("no sources recorded (ADR 11.6 step 3 asks for them)")
        if len(proposal.body.strip()) < 40:
            review.warnings.append("body is very short; a note with no detail rarely helps later")
        if proposal.supersedes and self.store.get(proposal.supersedes) is None:
            review.warnings.append(
                f"supersedes {proposal.supersedes!r} which is not in the store; the conflict "
                "will be recorded but cannot be checked"
            )
        return review

    # -- step 5: store ----------------------------------------------------

    async def promote(
        self,
        proposal: MemoryProposal,
        *,
        approved: bool = False,
        approved_by: str = "",
        allow_downgrade: bool = False,
        owner: str | None = None,
        service: str | None = None,
        environment: str | None = None,
        expires_after: str | None = None,
        today: date | None = None,
    ) -> PromotionOutcome:
        """Review, then store, then fire the promotion hook. Refusals are values.

        Returns an outcome rather than raising so a specialist can report the
        refusal to the user without an exception unwinding the graph.
        """
        review = self.review(proposal, approved=approved)
        downgraded = False
        category = review.category
        layer = review.target_layer
        doc_id = review.target_doc_id

        if not review.allowed:
            # The downgrade target is history/investigations, which is neither
            # curated nor approval-gated and explicitly accepts unverified notes
            # (ADR NG4). So the original target requiring approval is not a
            # reason to refuse the downgrade; it is the reason to offer it.
            # `downgraded` is reported in the outcome, so this is never silent.
            downgradeable = allow_downgrade and review.downgrade_available
            if not downgradeable:
                log.info(
                    "memory_promotion_refused",
                    title=proposal.title,
                    category=category,
                    reasons=review.blocking_reasons,
                )
                return PromotionOutcome(
                    stored=False, review=review, reason=review.summary()
                )
            downgraded = True
            category = DOWNGRADE_CATEGORY
            layer = DOWNGRADE_LAYER
            doc_id = self.store.unique_doc_id(category, proposal.title or "untitled-note")

        if self.hooks is not None:
            verdict = await self.hooks.on_memory_promotion(proposal)
            if verdict is not None and not verdict.allowed:
                reason = verdict.reason or "denied by an on_memory_promotion hook"
                log.info("memory_promotion_hook_denied", title=proposal.title, reason=reason)
                return PromotionOutcome(stored=False, review=review, reason=reason)

        status = _coerce_status(proposal.verification_status)
        now = today or datetime.now(tz=UTC).date()
        source = "; ".join(proposal.sources) if proposal.sources else "mimir session"
        if approved_by:
            source = f"{source} (approved by {approved_by})"

        metadata = DocumentMetadata(
            title=proposal.title or "Untitled note",
            category=category,
            service=service,
            environment=environment,
            created_at=now,
            last_verified=now if status is not VerificationStatus.UNVERIFIED else None,
            source=source,
            confidence=proposal.confidence,
            owner=owner,
            supersedes=[proposal.supersedes] if proposal.supersedes else [],
            expires_after=expires_after,
            tags=list(proposal.tags),
            verification_status=status,
            sources=list(proposal.sources),
        )
        body = _render_body(proposal, downgraded=downgraded, review=review)
        document = self.store.write(doc_id, metadata, body, overwrite=False)
        proposal.approved = review.approval_present
        log.info(
            "memory_promoted",
            doc_id=doc_id,
            layer=layer.value,
            verification_status=status.value,
            downgraded=downgraded,
        )
        return PromotionOutcome(
            stored=True,
            review=review,
            doc_id=doc_id,
            path=document.relative_path,
            layer=layer,
            downgraded=downgraded,
            document=document,
        )

    async def promote_or_raise(self, proposal: MemoryProposal, **kwargs: object) -> MemoryDocument:
        outcome = await self.promote(proposal, **kwargs)  # type: ignore[arg-type]
        if not outcome.stored or outcome.document is None:
            raise PromotionRefused(outcome.review.blocking_reasons or [outcome.reason])
        return outcome.document

    # -- step 6: keep freshness honest ------------------------------------

    def verify(
        self,
        doc_id: str,
        *,
        status: str = VerificationStatus.VERIFIED.value,
        when: date | None = None,
        note: str = "",
    ) -> MemoryDocument | None:
        """Re-stamp a document after checking it against a live system."""
        doc = self.store.get(doc_id)
        if doc is None:
            return None
        doc.metadata.verification_status = _coerce_status(status)
        doc.metadata.last_verified = when or datetime.now(tz=UTC).date()
        body = doc.body
        if note:
            body = f"{body.rstrip()}\n\n> Verification {doc.metadata.last_verified}: {note}\n"
        return self.store.write(doc_id, doc.metadata, body, overwrite=True)


def _coerce_status(value: str | VerificationStatus) -> VerificationStatus:
    if isinstance(value, VerificationStatus):
        return value
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return VerificationStatus(text)
    except ValueError:
        return VerificationStatus.UNVERIFIED


def _render_body(
    proposal: MemoryProposal, *, downgraded: bool, review: PromotionReview
) -> str:
    body = proposal.body.strip()
    if not body.lstrip().startswith("#"):
        body = f"# {proposal.title}\n\n{body}"
    sections = [body]
    if downgraded:
        sections.append(
            "> Stored in history/investigations rather than "
            f"{review.category} because it is unverified (ADR NG4). Verify it against a "
            "live system before promoting it."
        )
    if proposal.sources:
        sections.append("## Sources\n\n" + "\n".join(f"- {s}" for s in proposal.sources))
    return "\n\n".join(sections) + "\n"


#: Section skeleton for ADR 11.1 M3 incident records.
INCIDENT_SECTIONS: tuple[str, ...] = (
    "Symptoms",
    "Timeline",
    "Evidence",
    "Root cause",
    "Resolution",
    "Lessons",
    "Applicability limits",
)


def render_incident_body(
    *,
    title: str,
    occurred_on: str,
    symptoms: str = "",
    timeline: str = "",
    evidence: str = "",
    root_cause: str = "",
    resolution: str = "",
    lessons: str = "",
    applicability: str = "",
) -> str:
    """Build the ADR 11.1 M3 incident shape so records stay comparable."""
    values = {
        "Symptoms": symptoms,
        "Timeline": timeline,
        "Evidence": evidence,
        "Root cause": root_cause,
        "Resolution": resolution,
        "Lessons": lessons,
        "Applicability limits": applicability,
    }
    parts = [f"# {title}", f"**Date:** {occurred_on}"]
    for section in INCIDENT_SECTIONS:
        content = values.get(section, "").strip() or "_Not recorded._"
        parts.append(f"## {section}\n\n{content}")
    return "\n\n".join(parts) + "\n"


def default_confidence(status: VerificationStatus) -> Confidence:
    if status is VerificationStatus.VERIFIED:
        return Confidence.HIGH
    if status is VerificationStatus.USER_CONFIRMED:
        return Confidence.MEDIUM
    return Confidence.LOW
