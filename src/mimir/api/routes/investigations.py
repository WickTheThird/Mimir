"""Investigation, session, and approval routes (ADR 6.2 C1, 15)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from mimir.api.auth import Caller, require_local
from mimir.api.deps import get_runner_dependency
from mimir.export import evidence_package_json, evidence_package_markdown
from mimir.graph.runner import InvestigationRunner
from mimir.logging import get_logger
from mimir.models.approval import ApprovalStatus
from mimir.models.state import EnvironmentContext

log = get_logger(__name__)
router = APIRouter(tags=["investigations"])


class InvestigateRequest(BaseModel):
    question: str
    cluster_context: str | None = None
    namespace: str | None = None
    repositories: list[str] = Field(default_factory=list)
    sdm_resource: str | None = None
    time_range: str | None = None
    session_id: str | None = None

    def environment(self) -> EnvironmentContext:
        return EnvironmentContext(
            cluster_context=self.cluster_context,
            namespace=self.namespace,
            repositories=self.repositories,
            sdm_resource=self.sdm_resource,
            time_range=self.time_range,
        )


@router.post("/investigations")
async def start_investigation(
    payload: InvestigateRequest,
    caller: Caller = Depends(require_local),
    runner: InvestigationRunner = Depends(get_runner_dependency),
) -> dict[str, Any]:
    """Run an investigation to completion and return the full state."""
    state = await runner.run(
        payload.question,
        environment=payload.environment(),
        interface="web",
        session_id=payload.session_id,
    )
    return json.loads(state.model_dump_json())


@router.post("/investigations/stream")
async def stream_investigation(
    payload: InvestigateRequest,
    caller: Caller = Depends(require_local),
    runner: InvestigationRunner = Depends(get_runner_dependency),
) -> EventSourceResponse:
    """Stream an investigation as server-sent events (ADR 15 streaming)."""
    state = runner.new_session(
        payload.question,
        environment=payload.environment(),
        interface="web",
        session_id=payload.session_id,
    )

    async def events() -> Any:
        try:
            async for event in runner.stream(payload.question, state=state):
                yield {"event": event.type.value, "data": json.dumps(event.to_dict(), default=str)}
        except asyncio.CancelledError:
            # The browser navigated away.
            log.info("investigation_stream_cancelled", session_id=state.session_id)
            raise
        except Exception as exc:
            log.exception("investigation_stream_failed", session_id=state.session_id)
            yield {"event": "error", "data": json.dumps({"error": str(exc)})}

    return EventSourceResponse(events(), ping=15)


@router.get("/sessions")
async def list_sessions(
    limit: int = 50,
    caller: Caller = Depends(require_local),
) -> dict[str, Any]:
    from mimir.persistence.repositories import SessionRepository

    sessions = SessionRepository().list(limit=limit)
    return {"sessions": [json.loads(s.model_dump_json()) for s in sessions]}


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str, caller: Caller = Depends(require_local)
) -> dict[str, Any]:
    from mimir.persistence.repositories import load_state

    state = load_state(session_id)
    if state is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown session {session_id}")
    return json.loads(state.model_dump_json())


@router.post("/sessions/{session_id}/resume")
async def resume_session(
    session_id: str,
    caller: Caller = Depends(require_local),
    runner: InvestigationRunner = Depends(get_runner_dependency),
) -> dict[str, Any]:
    state = await runner.resume(session_id)
    if state is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no checkpoint for {session_id}")
    return json.loads(state.model_dump_json())


@router.get("/sessions/{session_id}/export")
async def export_session(
    session_id: str,
    fmt: str = "md",
    caller: Caller = Depends(require_local),
) -> dict[str, Any]:
    """Evidence package handoff (ADR 5.8)."""
    from mimir.persistence.repositories import load_state

    state = load_state(session_id)
    if state is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown session {session_id}")
    body = evidence_package_json(state) if fmt == "json" else evidence_package_markdown(state)
    return {"format": fmt, "content": body}


# ---------------------------------------------------------------------------


class ApprovalDecisionRequest(BaseModel):
    decision: str = Field(description="approve, reject, or edit.")
    reason: str | None = None
    edited_argv: list[str] | None = None
    decided_by: str = "web"


@router.get("/approvals")
async def list_approvals(
    session_id: str | None = None,
    caller: Caller = Depends(require_local),
    runner: InvestigationRunner = Depends(get_runner_dependency),
) -> dict[str, Any]:
    pending = runner.approvals.pending(session_id)
    return {
        "approvals": [
            {
                "id": request.id,
                "session_id": request.session_id,
                "command": request.command.display,
                "risk": request.assessment.risk.value,
                "production_target": request.assessment.production_target,
                "reversible": request.assessment.reversible,
                "rollback_hint": request.assessment.rollback_hint,
                "reasons": request.assessment.reasons,
                "context": request.command.context.model_dump(exclude_none=True),
                "purpose": request.command.purpose,
                "expected_effect": request.command.expected_effect,
                "prompt": request.prompt,
                "created_at": request.created_at,
                "expires_at": request.expires_at,
            }
            for request in pending
        ]
    }


@router.post("/approvals/{approval_id}")
async def decide_approval(
    approval_id: str,
    payload: ApprovalDecisionRequest,
    caller: Caller = Depends(require_local),
    runner: InvestigationRunner = Depends(get_runner_dependency),
) -> dict[str, Any]:
    mapping = {
        "approve": ApprovalStatus.APPROVED,
        "reject": ApprovalStatus.REJECTED,
        "edit": ApprovalStatus.EDITED,
    }
    resolved = mapping.get(payload.decision.lower())
    if resolved is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"decision must be one of: {', '.join(mapping)}",
        )
    if resolved == ApprovalStatus.EDITED and not payload.edited_argv:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "edit requires edited_argv"
        )
    if runner.approvals.get(approval_id) is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"approval {approval_id} is not pending; it may have expired or been decided",
        )

    decision = await runner.approvals.resolve(
        approval_id,
        resolved,
        decided_by=payload.decided_by,
        reason=payload.reason,
        edited_argv=payload.edited_argv,
    )
    return json.loads(decision.model_dump_json())


# ---------------------------------------------------------------------------


@router.get("/artifacts/{ref}")
async def read_artifact(
    ref: str,
    limit: int = 200_000,
    caller: Caller = Depends(require_local),
) -> dict[str, Any]:
    """Read stored command output or a fetched document."""
    from mimir.tools.artifacts import get_artifact_store

    artifact = get_artifact_store().get(ref)
    if artifact is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown artifact {ref}")
    return {
        "ref": artifact.ref,
        "kind": artifact.kind,
        "size_bytes": artifact.size_bytes,
        "created_at": artifact.created_at,
        "metadata": artifact.metadata,
        "content": artifact.read(limit),
        "truncated": artifact.size_bytes > limit,
    }
