"""Session and chat message models (ADR 19.3)."""

from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SPECIALIST = "specialist"
    APPROVAL = "approval"


class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    role: MessageRole
    content: str = ""
    name: str | None = None
    created_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionStatus(StrEnum):
    ACTIVE = "active"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_INPUT = "waiting_input"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Session(BaseModel):
    id: str = Field(default_factory=lambda: f"ses_{uuid.uuid4().hex[:12]}")
    title: str = ""
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    interface: str = "cli"
    """cli, web, api, or warp."""

    user_request: str = ""
    task_type: str | None = None
    model_alias: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = time.time()
