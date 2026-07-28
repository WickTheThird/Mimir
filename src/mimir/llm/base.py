"""Model runtime abstraction (ADR 6.2 C3, 18).

The ADR deliberately leaves the model and the runtime unresolved (section 25),
so nothing above this layer may assume Ollama, MLX, llama.cpp, vLLM, or LiteLLM.
Everything talks to :class:`ChatModel`.

The interface is intentionally small: chat, stream, and structured output. Tool
calling is expressed in the OpenAI shape because every candidate runtime in ADR
6.2 C3 either speaks it natively or can be adapted to it.
"""

from __future__ import annotations

import json
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:10]}")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str = ""

    @classmethod
    def from_openai(cls, payload: dict[str, Any]) -> ToolCall:
        function = payload.get("function", {})
        raw = function.get("arguments", "") or ""
        try:
            parsed = json.loads(raw) if raw.strip() else {}
            if not isinstance(parsed, dict):
                parsed = {"value": parsed}
        except json.JSONDecodeError:
            parsed = {}
        return cls(
            id=payload.get("id") or f"call_{uuid.uuid4().hex[:10]}",
            name=function.get("name", ""),
            arguments=parsed,
            raw_arguments=raw,
        )

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.raw_arguments or json.dumps(self.arguments),
            },
        }


class LLMMessage(BaseModel):
    role: Role
    content: str = ""
    name: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None

    def to_openai(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role.value}
        # An assistant turn that only calls tools must send content: null, not "",
        # or several runtimes reject the follow-up request.
        payload["content"] = self.content if self.content or not self.tool_calls else None
        if self.name:
            payload["name"] = self.name
        if self.tool_calls:
            payload["tool_calls"] = [c.to_openai() for c in self.tool_calls]
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        return payload

    @classmethod
    def system(cls, content: str) -> LLMMessage:
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> LLMMessage:
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: list[ToolCall] | None = None) -> LLMMessage:
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool_result(cls, tool_call_id: str, content: str, name: str | None = None) -> LLMMessage:
        return cls(role=Role.TOOL, content=content, tool_call_id=tool_call_id, name=name)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class ChatResponse(BaseModel):
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str = "stop"
    model: str = ""
    alias: str = ""
    usage: Usage = Field(default_factory=Usage)
    latency_s: float = 0.0
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def as_message(self) -> LLMMessage:
        return LLMMessage.assistant(self.content, self.tool_calls)


class ChunkType(StrEnum):
    CONTENT = "content"
    TOOL_CALL = "tool_call"
    DONE = "done"
    ERROR = "error"


@dataclass(slots=True)
class StreamChunk:
    type: ChunkType
    text: str = ""
    tool_call: ToolCall | None = None
    finish_reason: str | None = None
    error: str | None = None


@dataclass(slots=True)
class GenerationOptions:
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: Sequence[str] | None = None
    seed: int | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    timeout_s: float | None = None


class ModelError(Exception):
    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable
        self.status = status


class ChatModel(ABC):
    """What every runtime adapter implements."""

    alias: str
    model: str
    context_window: int
    supports_tools: bool
    supports_json_schema: bool

    @abstractmethod
    async def chat(
        self, messages: Sequence[LLMMessage], options: GenerationOptions | None = None
    ) -> ChatResponse: ...

    @abstractmethod
    def stream(
        self, messages: Sequence[LLMMessage], options: GenerationOptions | None = None
    ) -> AsyncIterator[StreamChunk]: ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise ModelError(f"{type(self).__name__} does not implement embeddings")

    async def health(self) -> tuple[bool, str]:
        """Cheap reachability probe used by ``mimir doctor``."""
        try:
            response = await self.chat(
                [LLMMessage.user("ping")],
                GenerationOptions(max_tokens=4, temperature=0.0, timeout_s=20.0),
            )
        except Exception as exc:  # noqa: BLE001 - the probe reports, never raises
            return False, f"{type(exc).__name__}: {exc}"
        return True, f"{self.model} responded in {response.latency_s:.2f}s"

    async def close(self) -> None:
        return None


def schema_stub(schema: dict[str, Any], depth: int = 0) -> Any:
    """Build a minimal instance that satisfies a JSON Schema.

    Used by :class:`EchoModel` so the structured-output path can be exercised
    without a model runtime. It emits required fields only, with type-appropriate
    empty values, which is exactly the shape a validator accepts and a caller
    must still cope with.
    """
    if depth > 6:
        return None
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            options = schema[key]
            chosen = next((o for o in options if o.get("type") != "null"), options[0])
            return schema_stub(chosen, depth + 1)
    if schema.get("enum"):
        return schema["enum"][0]
    if "const" in schema:
        return schema["const"]

    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "null")
    if kind == "object" or "properties" in schema:
        properties = schema.get("properties", {})
        required = schema.get("required", list(properties))
        return {
            name: schema_stub(properties.get(name, {}), depth + 1)
            for name in required
            if name in properties
        }
    if kind == "array":
        return []
    if kind == "string":
        return ""
    if kind == "integer":
        return 0
    if kind == "number":
        return 0.0
    if kind == "boolean":
        return False
    return None


def _resolve_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline ``$ref`` pointers so :func:`schema_stub` can walk a pydantic schema."""
    defs = schema.get("$defs", {}) or schema.get("definitions", {})

    def walk(node: Any, depth: int = 0) -> Any:
        if depth > 8:
            return {}
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/"):
                target = defs.get(ref.rsplit("/", 1)[-1], {})
                return walk(target, depth + 1)
            return {k: walk(v, depth + 1) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(item, depth + 1) for item in node]
        return node

    return walk(schema)


class EchoModel(ChatModel):
    """Deterministic stand-in used by tests and by ``--no-model`` runs.

    It never fabricates an investigative answer; it echoes what it was asked so a
    graph or CLI path can be exercised without a runtime attached. When a JSON
    schema is requested it returns a minimal valid instance rather than prose, so
    structured-output call sites are exercised rather than always falling back.
    """

    def __init__(self, alias: str = "echo", scripted: list[ChatResponse] | None = None) -> None:
        self.alias = alias
        self.model = "echo"
        self.context_window = 8192
        self.supports_tools = True
        self.supports_json_schema = True
        self.scripted = list(scripted or [])
        self.calls: list[list[LLMMessage]] = []

    async def chat(
        self, messages: Sequence[LLMMessage], options: GenerationOptions | None = None
    ) -> ChatResponse:
        self.calls.append(list(messages))
        if self.scripted:
            return self.scripted.pop(0)
        last = next(
            (m.content for m in reversed(messages) if m.role == Role.USER),
            "",
        )
        content = f"[echo] {last}"
        response_format = (options.response_format if options else None) or {}
        if response_format.get("type") == "json_schema":
            schema = response_format.get("json_schema", {}).get("schema", {})
            content = json.dumps(schema_stub(_resolve_refs(schema)))
        elif response_format.get("type") == "json_object":
            content = "{}"
        return ChatResponse(
            content=content,
            model="echo",
            alias=self.alias,
            latency_s=0.0,
        )

    async def stream(  # type: ignore[override]
        self, messages: Sequence[LLMMessage], options: GenerationOptions | None = None
    ) -> AsyncIterator[StreamChunk]:
        response = await self.chat(messages, options)
        for token in response.content.split(" "):
            yield StreamChunk(type=ChunkType.CONTENT, text=token + " ")
        yield StreamChunk(type=ChunkType.DONE, finish_reason="stop")

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        from mimir.knowledge.embeddings import hash_embedding

        return [hash_embedding(t) for t in texts]


def estimate_tokens(text: str) -> int:
    """Rough token estimate used for context budgeting (ADR 20, R7).

    Deliberately crude. Runtimes disagree on tokenisers, and the ADR only needs
    this for budgeting decisions and telemetry, not for billing.
    """
    return max(1, len(text) // 4)


def messages_token_estimate(messages: Sequence[LLMMessage]) -> int:
    total = 0
    for message in messages:
        total += estimate_tokens(message.content) + 4
        for call in message.tool_calls:
            total += estimate_tokens(call.raw_arguments or json.dumps(call.arguments)) + 8
    return total


def trim_to_context(
    messages: list[LLMMessage], budget_tokens: int, *, keep_first: int = 1, keep_last: int = 6
) -> list[LLMMessage]:
    """Drop middle turns when the conversation outgrows the window (ADR R7).

    The system prompt and the most recent exchanges are preserved; a marker
    replaces what was dropped so the model is not silently misled about what it
    has seen.
    """
    if messages_token_estimate(messages) <= budget_tokens:
        return messages
    head = messages[:keep_first]
    tail = messages[-keep_last:] if keep_last else []
    middle = messages[keep_first : len(messages) - len(tail)]
    dropped = 0
    while middle and messages_token_estimate([*head, *middle, *tail]) > budget_tokens:
        middle.pop(0)
        dropped += 1
    if dropped:
        notice = LLMMessage.system(
            f"[{dropped} earlier turns were dropped to fit the context window. "
            "Evidence and command history remain available in the investigation state.]"
        )
        return [*head, notice, *middle, *tail]
    return [*head, *middle, *tail]


@dataclass(slots=True)
class ModelCallRecord:
    """Telemetry row (ADR 20)."""

    alias: str
    model: str
    started_at: float
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    tool_calls: int
    finish_reason: str
    session_id: str | None = None
    purpose: str = ""
    error: str | None = None

    @classmethod
    def from_response(
        cls,
        response: ChatResponse,
        *,
        session_id: str | None = None,
        purpose: str = "",
        started_at: float | None = None,
    ) -> ModelCallRecord:
        return cls(
            alias=response.alias,
            model=response.model,
            started_at=started_at or time.time(),
            latency_s=response.latency_s,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            tool_calls=len(response.tool_calls),
            finish_reason=response.finish_reason,
            session_id=session_id,
            purpose=purpose,
        )
