"""Human-in-the-loop approval records (ADR 13, 14.2, 15).

Approvals are raised by the policy engine, surfaced by whichever interface owns
the session (CLI, web UI, API), and resolved out of band. LangGraph interrupts
carry the :class:`ApprovalRequest` payload verbatim.
"""

from __future__ import annotations

import time
import uuid
from enum import StrEnum

from pydantic import BaseModel, Field

from mimir.models.command import ProposedCommand, RiskAssessment


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ApprovalRequest(BaseModel):
    id: str = Field(default_factory=lambda: f"apr_{uuid.uuid4().hex[:12]}")
    session_id: str | None = None
    command: ProposedCommand
    assessment: RiskAssessment
    prompt: str = ""
    """Rendered human-facing text, already including the ADR 13.3 display."""

    created_at: float = Field(default_factory=time.time)
    expires_at: float | None = None
    requested_by: str = "mimir"
    status: ApprovalStatus = ApprovalStatus.PENDING

    @property
    def expired(self) -> bool:
        return self.expires_at is not None and time.time() > self.expires_at


class ApprovalDecision(BaseModel):
    approval_id: str
    status: ApprovalStatus
    decided_at: float = Field(default_factory=time.time)
    decided_by: str = "user"
    reason: str | None = None
    edited_argv: list[str] | None = None
    """Set when the operator edited the command before approving it."""

    @property
    def allows_execution(self) -> bool:
        return self.status in (ApprovalStatus.APPROVED, ApprovalStatus.EDITED)
