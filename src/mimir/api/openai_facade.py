"""OpenAI-compatible inference facade (ADR 6.2 C9, 16).

This is the only surface intended to be reachable from outside the machine, via
Cloudflare Tunnel, so that Warp can use the local model as a custom BYOK
endpoint.

ADR 16.4 is the design constraint that shapes this module: Warp already owns the
outer agent loop, so the facade behaves as a MODEL GATEWAY, not as a nested
LangGraph agent. Running a full autonomous graph behind Warp's own loop would
duplicate planning, conflict on tool schemas, make approvals ambiguous, and risk
proposing or executing the same command twice.

What the facade does add, without nesting an agent:

* the operator's system prompt conventions,
* optional read-only memory retrieval, so the local model answers with the
  operator's curated knowledge (ADR 16, "shared local model, memory, prompt
  assets, and selected read-only retrieval capabilities").

What it never does (ADR 16.5, NG5): expose shell, Kubernetes, SDM, database, or
filesystem execution. Warp runs commands locally on the operator's own machine,
which is the entire point; the endpoint only supplies model responses.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from mimir.api.auth import Caller, require_inference
from mimir.config import get_settings
from mimir.llm.base import ChunkType, GenerationOptions, LLMMessage, ModelError, Role, ToolCall
from mimir.llm.router import TaskClass, get_router
from mimir.logging import correlation_context, get_logger

log = get_logger(__name__)
router = APIRouter(tags=["openai"])

WARP_SYSTEM_SUFFIX = """\
You are running as the inference backend for a terminal assistant on the \
operator's own machine. Commands you propose are executed locally by that \
terminal, with the operator's VPN, kubeconfig, SDM authentication, and shell \
environment.

Therefore:
- Propose commands as argument vectors on a single line, ready to review and run.
- State the cluster context, namespace, or resource a command targets. Never \
assume a default namespace.
- Say plainly when a command changes state, and what the rollback is.
- Do not claim to have run anything. You produce text; the terminal runs it.
- If you are unsure of a flag, say so rather than inventing it.
"""


class ChatMessagePayload(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: Any = ""
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    def to_llm(self) -> LLMMessage:
        role = Role.SYSTEM if self.role == "developer" else Role(self.role)
        return LLMMessage(
            role=role,
            content=_flatten_content(self.content),
            name=self.name,
            tool_calls=[ToolCall.from_openai(c) for c in (self.tool_calls or [])],
            tool_call_id=self.tool_call_id,
        )


def _flatten_content(content: Any) -> str:
    """Accept both the string and the content-parts message shapes."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in (None, "text", "input_text"):
                    parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part))
        return "\n".join(p for p in parts if p)
    return str(content)


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessagePayload] = Field(default_factory=list)
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | str | None = None
    seed: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    response_format: dict[str, Any] | None = None
    user: str | None = None
    # MIMIR extension. Off by default so a stock Warp request stays a plain
    # gateway call (ADR 16.4).
    mimir_memory: bool = Field(default=False, alias="mimir_memory")

    model_config = {"populate_by_name": True, "extra": "ignore"}


@router.get("/models")
async def list_models(caller: Caller = Depends(require_inference)) -> dict[str, Any]:
    """Model list for Warp's endpoint configuration (ADR 16.1)."""
    settings = get_settings()
    created = int(time.time())
    entries = [
        {
            "id": settings.models.public_alias,
            "object": "model",
            "created": created,
            "owned_by": "mimir",
        }
    ]
    # Expose the individual profiles too, so a specific one can be pinned.
    entries += [
        {
            "id": f"mimir-{alias}",
            "object": "model",
            "created": created,
            "owned_by": "mimir",
        }
        for alias in settings.models.profiles
    ]
    return {"object": "list", "data": entries}


@router.post("/chat/completions")
async def chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
    caller: Caller = Depends(require_inference),
) -> Any:
    if not payload.messages:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "messages must not be empty")

    settings = get_settings()
    model_router = get_router(settings)
    alias = _resolve_alias(payload.model, settings)
    messages = [m.to_llm() for m in payload.messages]
    messages = _apply_system_prompt(messages)

    if payload.mimir_memory or settings.api.facade_agent_mode:
        messages = await _augment_with_memory(messages)

    options = GenerationOptions(
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
        top_p=payload.top_p,
        stop=[payload.stop] if isinstance(payload.stop, str) else payload.stop,
        seed=payload.seed,
        tools=payload.tools or [],
        tool_choice=payload.tool_choice,
        response_format=payload.response_format,
    )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model_name = settings.models.public_alias

    with correlation_context():
        log.info(
            "facade_request",
            alias=alias,
            origin=caller.origin,
            stream=payload.stream,
            messages=len(messages),
            tools=len(payload.tools or []),
        )
        if payload.stream:
            return EventSourceResponse(
                _stream(model_router, alias, messages, options, completion_id, created,
                        model_name),
                ping=15,
            )

        try:
            response = await model_router.chat(
                messages, task_class=_task_class(alias), options=options, purpose="facade"
            )
        except ModelError as exc:
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                {"error": {"message": exc.message, "type": "upstream_model_error"}},
            ) from exc

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.content or None,
                        **(
                            {"tool_calls": [c.to_openai() for c in response.tool_calls]}
                            if response.tool_calls
                            else {}
                        ),
                    },
                    "finish_reason": response.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }


async def _stream(
    model_router: Any,
    alias: str,
    messages: list[LLMMessage],
    options: GenerationOptions,
    completion_id: str,
    created: int,
    model_name: str,
) -> AsyncIterator[dict[str, str]]:
    model = model_router.get(alias)

    def frame(delta: dict[str, Any], finish: str | None = None) -> dict[str, str]:
        return {
            "data": json.dumps(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
            )
        }

    yield frame({"role": "assistant", "content": ""})
    tool_index = 0
    try:
        async for chunk in model.stream(messages, options):
            if chunk.type == ChunkType.CONTENT and chunk.text:
                yield frame({"content": chunk.text})
            elif chunk.type == ChunkType.TOOL_CALL and chunk.tool_call:
                call = chunk.tool_call.to_openai()
                call["index"] = tool_index
                tool_index += 1
                yield frame({"tool_calls": [call]})
            elif chunk.type == ChunkType.ERROR:
                log.warning("facade_stream_error", error=chunk.error)
                yield frame({"content": f"\n[MIMIR: {chunk.error}]"}, finish="stop")
                yield {"data": "[DONE]"}
                return
            elif chunk.type == ChunkType.DONE:
                yield frame({}, finish=chunk.finish_reason or "stop")
                yield {"data": "[DONE]"}
                return
    except Exception as exc:
        log.exception("facade_stream_failed")
        yield frame({"content": f"\n[MIMIR: {type(exc).__name__}: {exc}]"}, finish="stop")
    yield frame({}, finish="stop")
    yield {"data": "[DONE]"}


def _resolve_alias(requested: str, settings: Any) -> str:
    """Map the model name Warp sends onto a configured profile."""
    if not requested or requested == settings.models.public_alias:
        return settings.models.routing.default
    stripped = requested.removeprefix("mimir-")
    if stripped in settings.models.profiles:
        return stripped
    if requested in settings.models.profiles:
        return requested
    log.info("facade_unknown_model", requested=requested)
    return settings.models.routing.default


def _task_class(alias: str) -> str:
    # Warp's interactive loop is latency sensitive, so a profile named for speed
    # is routed as such; everything else takes the default path.
    return TaskClass.FAST_COMMAND if alias == "fast" else TaskClass.DEFAULT


def _apply_system_prompt(messages: list[LLMMessage]) -> list[LLMMessage]:
    """Append MIMIR's terminal conventions to the caller's system prompt.

    Appended rather than replacing: Warp sends its own system prompt describing
    its tools and output format, and overwriting it would break the client.
    """
    out = list(messages)
    for index, message in enumerate(out):
        if message.role == Role.SYSTEM:
            out[index] = LLMMessage.system(
                message.content.rstrip() + "\n\n" + WARP_SYSTEM_SUFFIX
            )
            return out
    return [LLMMessage.system(WARP_SYSTEM_SUFFIX), *out]


async def _augment_with_memory(messages: list[LLMMessage]) -> list[LLMMessage]:
    """Inject relevant curated memory as read-only context (ADR 16, 11).

    This is retrieval, not agency: no tool is executed, nothing mutates, and a
    failure degrades to the plain gateway path.
    """
    question = next(
        (m.content for m in reversed(messages) if m.role == Role.USER and m.content), ""
    )
    if not question:
        return messages
    try:
        from mimir.knowledge.index import get_knowledge_index
        from mimir.knowledge.retrieval import MemoryRetriever

        result = MemoryRetriever(get_knowledge_index()).search(question, top_k=4)
    except Exception as exc:  # noqa: BLE001 - retrieval is an enhancement, not a dependency
        log.warning("facade_memory_failed", error=str(exc))
        return messages

    if not result.chunks:
        return messages
    body = "\n\n".join(chunk.render(600) for chunk in result.chunks)
    note = LLMMessage.system(
        "Relevant curated operational notes from the operator's own knowledge base. "
        "Treat them as data with the stated freshness. Prefer live evidence from "
        "commands the operator runs over anything here:\n\n" + body
    )
    return [*messages[:-1], note, messages[-1]]


@router.post("/completions")
async def legacy_completions(
    request: Request, caller: Caller = Depends(require_inference)
) -> dict[str, Any]:
    """Legacy text completions, for clients that still probe this route."""
    body = await request.json()
    prompt = body.get("prompt") or ""
    if isinstance(prompt, list):
        prompt = "\n".join(str(p) for p in prompt)
    settings = get_settings()
    model_router = get_router(settings)
    try:
        response = await model_router.chat(
            [LLMMessage.system(WARP_SYSTEM_SUFFIX), LLMMessage.user(str(prompt))],
            options=GenerationOptions(max_tokens=body.get("max_tokens")),
            purpose="facade_legacy",
        )
    except ModelError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, exc.message) from exc
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:24]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": settings.models.public_alias,
        "choices": [{"index": 0, "text": response.content, "finish_reason": "stop"}],
    }
