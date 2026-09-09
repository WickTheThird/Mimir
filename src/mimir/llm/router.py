"""Model routing and structured output (ADR 18.4).

Routing lets a fast model answer command-completion requests while a larger one
runs deep investigation (ADR R6 latency mitigation). It is optional by design:
if every task class points at the same alias, this collapses to a single model.

:class:`ModelRouter` also owns structured output, because getting a local model
to emit valid JSON reliably needs more than passing ``response_format``:

1. ask with a JSON schema when the runtime supports it,
2. otherwise ask for JSON mode with the schema restated in the prompt,
3. extract the first JSON object from a chatty reply,
4. repair common local-model failures (fenced blocks, trailing commas, single
   quotes, a schema echoed instead of an instance),
5. retry once with the validation error fed back,
6. fail loudly rather than returning a half-parsed object.
"""

from __future__ import annotations

import ast
import json
import re
import time
from collections.abc import Sequence
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from mimir.config import ModelProfile, Settings, get_settings
from mimir.llm.base import (
    ChatModel,
    ChatResponse,
    GenerationOptions,
    LLMMessage,
    ModelCallRecord,
    ModelError,
    messages_token_estimate,
    trim_to_context,
)
from mimir.llm.openai_compat import build_model
from mimir.logging import get_logger

log = get_logger(__name__)


def _specialist_from_purpose(purpose: str) -> str:
    """Extract the specialist from a purpose string like 'specialist:log_analyst'.

    Purpose is free text used for logging. Parsing it is what lets telemetry be
    grouped by specialist without threading another parameter through every
    call site.
    """
    if not purpose.startswith("specialist:"):
        return ""
    return purpose.split(":", 2)[1]

T = TypeVar("T", bound=BaseModel)


class TaskClass:
    """Routing keys. Strings rather than an enum so config stays open."""

    DEFAULT = "default"
    FAST_COMMAND = "fast_command"
    DEEP_INVESTIGATION = "deep_investigation"
    WEB_SYNTHESIS = "web_synthesis"
    EVIDENCE_VERIFICATION = "evidence_verification"
    FINAL_SYNTHESIS = "final_synthesis"
    CLASSIFICATION = "classification"
    EMBEDDING = "embedding"


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


class StructuredOutputError(ModelError):
    def __init__(self, message: str, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw


def _native_messages(messages: Sequence[LLMMessage]) -> list[dict[str, Any]]:
    """Serialise for the runtime's own chat API rather than the OpenAI one.

    The two disagree about tool turns. The OpenAI shape carries tool_call_id on
    a tool result and a tool_calls array on the assistant turn that caused it;
    the native endpoint rejects the request outright. A first constrained call
    therefore worked and the second, which was the first one that had a tool
    result in its history, returned 400.

    Under constrained decoding the assistant's move is already a JSON object,
    so it is carried as content, and a tool result is a tool turn with text.
    No structure is lost because none of it was in the tool-call channel.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        role = message.role.value
        content = message.content or ""
        if message.tool_calls and not content:
            content = json.dumps([
                {"tool": call.name, "arguments": call.arguments}
                for call in message.tool_calls
            ])
        entry: dict[str, Any] = {"role": role, "content": content}
        if role == "tool" and message.name:
            entry["name"] = message.name
        out.append(entry)
    return out


class ModelRouter:
    """Resolves task classes to models and owns the shared client lifecycle."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._models: dict[str, ChatModel] = {}
        self.call_log: list[ModelCallRecord] = []
        self.invocations_attempted = 0
        """Every runtime call this router has issued, including retried ones.

        This is the left-hand side of the telemetry invariant: it is incremented
        at the call site itself, so it cannot drift from reality the way a count
        derived from the log could. len(call_log) must equal it.
        """
        self._digests: dict[str, str] = {}

    # -- resolution ------------------------------------------------------

    def profile_for(self, alias: str) -> ModelProfile:
        profiles = self.settings.models.profiles
        if alias in profiles:
            return profiles[alias]
        fallback = self.settings.models.routing.default
        if fallback in profiles:
            log.warning("unknown_model_alias", alias=alias, using=fallback)
            return profiles[fallback]
        if profiles:
            first = next(iter(profiles.values()))
            log.warning("no_default_profile", alias=alias, using=first.alias)
            return first
        raise ModelError("no model profiles are configured; run 'mimir init'")

    def alias_for_task(self, task_class: str) -> str:
        routing = self.settings.models.routing
        return getattr(routing, task_class, routing.default)

    def get(self, alias: str) -> ChatModel:
        if alias not in self._models:
            self._models[alias] = build_model(self.profile_for(alias))
        return self._models[alias]

    def for_task(self, task_class: str = TaskClass.DEFAULT) -> ChatModel:
        return self.get(self.alias_for_task(task_class))

    def digest_for(self, alias: str) -> str:
        """Resolve and cache the served digest for an alias.

        Resolved once per process, not per call: the digest cannot change under
        a running runtime without a reload, and querying it on every invocation
        would add a network round trip to every model call.
        """
        if alias not in self._digests:
            try:
                from mimir.eval.provenance import resolve_model

                self._digests[alias] = resolve_model(alias, self.settings).digest
            except Exception:  # noqa: BLE001 - telemetry must never break a call
                self._digests[alias] = ""
        return self._digests[alias]

    # -- calls -----------------------------------------------------------

    async def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        task_class: str = TaskClass.DEFAULT,
        options: GenerationOptions | None = None,
        session_id: str | None = None,
        purpose: str = "",
        retries: int = 1,
        tool_calls_before: int = 0,
    ) -> ChatResponse:
        model = self.for_task(task_class)
        budget = int(model.context_window * 0.75)
        trimmed = list(messages)
        estimate = messages_token_estimate(trimmed)
        was_trimmed = estimate > budget
        if was_trimmed:
            trimmed = trim_to_context(trimmed, budget)
            log.info(
                "context_trimmed",
                alias=model.alias,
                estimated_tokens=messages_token_estimate(trimmed),
                budget=budget,
            )

        # Shared by every attempt, so a record can always be attributed to the
        # role and specialist that caused it rather than to an anonymous alias.
        common = {
            "runtime": model.profile.runtime if hasattr(model, "profile") else "",
            "digest": self.digest_for(model.alias),
            "task_class": task_class,
            "specialist": _specialist_from_purpose(purpose),
            "context_window": model.context_window,
            "context_estimate": estimate,
            "trimmed": was_trimmed,
            "tool_calls_before": tool_calls_before,
        }

        last_error: ModelError | None = None
        for attempt in range(retries + 1):
            # One record per attempt. A retried call really is two invocations
            # of the runtime and costs two invocations of compute; collapsing
            # them understates load and makes the retry rate unmeasurable.
            started = time.time()
            self.invocations_attempted += 1
            try:
                response = await model.chat(trimmed, options)
            except ModelError as exc:
                last_error = exc
                self.call_log.append(
                    ModelCallRecord(
                        alias=model.alias,
                        model=model.model,
                        started_at=started,
                        latency_s=time.time() - started,
                        prompt_tokens=0,
                        completion_tokens=0,
                        tool_calls=0,
                        finish_reason="error",
                        session_id=session_id,
                        purpose=purpose,
                        error=exc.message,
                        attempt=attempt,
                        **common,
                    )
                )
                if not exc.retryable or attempt >= retries:
                    raise
                log.warning("model_retry", alias=model.alias, attempt=attempt + 1,
                            error=exc.message)
                continue
            self.call_log.append(
                ModelCallRecord.from_response(
                    response,
                    session_id=session_id,
                    purpose=purpose,
                    started_at=started,
                    attempt=attempt,
                    **common,
                )
            )
            return response
        raise last_error or ModelError("model call failed")

    async def constrained(
        self,
        messages: Sequence[LLMMessage],
        schema: dict[str, Any],
        *,
        task_class: str = TaskClass.DEFAULT,
        session_id: str | None = None,
        purpose: str = "",
        max_tokens: int = 900,
        tool_calls_before: int = 0,
    ) -> str:
        """One call whose output must satisfy ``schema``.

        Goes through the runtime's native endpoint because that is where the
        grammar constraint lives; the OpenAI-compatible surface does not carry
        it. Recorded here rather than at the call site so a constrained
        invocation counts exactly like any other, which is the invariant that
        found the empty model_calls table.
        """
        import httpx

        from mimir.llm.base import ModelCallRecord

        model = self.for_task(task_class)
        profile = getattr(model, "profile", None)
        base = str(getattr(profile, "base_url", "")).rstrip("/").removesuffix("/v1")
        if not base:
            raise ModelError("constrained decoding needs a runtime base url")

        payload = {
            "model": model.model,
            "messages": _native_messages(messages),
            "stream": False,
            "format": schema,
            "options": {"temperature": 0, "num_predict": max_tokens},
        }
        started = time.time()
        self.invocations_attempted += 1
        try:
            async with httpx.AsyncClient(timeout=profile.request_timeout_s) as client:
                response = await client.post(f"{base}/api/chat", json=payload)
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPError as exc:
            self.call_log.append(
                ModelCallRecord(
                    alias=model.alias, model=model.model, started_at=started,
                    latency_s=time.time() - started, prompt_tokens=0,
                    completion_tokens=0, tool_calls=0, finish_reason="error",
                    session_id=session_id, purpose=purpose, error=str(exc),
                    attempt=0, task_class=task_class,
                    context_window=model.context_window,
                    tool_calls_before=tool_calls_before,
                )
            )
            raise ModelError(f"constrained call failed: {exc}", retryable=True) from exc

        content = ((body.get("message") or {}).get("content") or "").strip()
        self.call_log.append(
            ModelCallRecord(
                alias=model.alias, model=model.model, started_at=started,
                latency_s=time.time() - started,
                prompt_tokens=int(body.get("prompt_eval_count") or 0),
                completion_tokens=int(body.get("eval_count") or 0),
                tool_calls=0 if not content else 1,
                finish_reason=body.get("done_reason") or "stop",
                session_id=session_id, purpose=purpose, attempt=0,
                runtime=getattr(profile, "runtime", ""),
                digest=self.digest_for(model.alias),
                task_class=task_class, context_window=model.context_window,
                tool_calls_before=tool_calls_before,
            )
        )
        return content

    # -- structured output ------------------------------------------------

    async def structured(
        self,
        messages: Sequence[LLMMessage],
        schema: type[T],
        *,
        task_class: str = TaskClass.DEFAULT,
        options: GenerationOptions | None = None,
        session_id: str | None = None,
        purpose: str = "",
        max_attempts: int = 2,
    ) -> T:
        """Return a validated instance of ``schema`` or raise."""
        model = self.for_task(task_class)
        json_schema = schema.model_json_schema()
        opts = options or GenerationOptions()
        conversation = list(messages)

        if model.supports_json_schema:
            opts.response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": json_schema,
                    "strict": False,
                },
            }
        else:
            opts.response_format = {"type": "json_object"}
            conversation = [
                *conversation,
                LLMMessage.system(
                    "Reply with a single JSON object and nothing else. It must satisfy "
                    f"this JSON Schema:\n{json.dumps(json_schema, indent=2)}\n"
                    "Do not wrap it in Markdown. Do not return the schema itself."
                ),
            ]

        raw = ""
        for attempt in range(max_attempts):
            response = await self.chat(
                conversation,
                task_class=task_class,
                options=opts,
                session_id=session_id,
                purpose=purpose or f"structured:{schema.__name__}",
            )
            raw = response.content
            try:
                return schema.model_validate(extract_json(raw))
            except (StructuredOutputError, ValidationError, json.JSONDecodeError) as exc:
                if attempt == max_attempts - 1:
                    raise StructuredOutputError(
                        f"model did not produce valid {schema.__name__} after "
                        f"{max_attempts} attempts: {exc}",
                        raw=raw,
                    ) from exc
                log.warning(
                    "structured_output_retry",
                    schema=schema.__name__,
                    attempt=attempt + 1,
                    error=str(exc)[:300],
                )
                conversation = [
                    *conversation,
                    LLMMessage.assistant(raw[:2000]),
                    LLMMessage.user(
                        "That was not valid. Fix it and return only the corrected JSON "
                        f"object. The validation error was:\n{str(exc)[:800]}"
                    ),
                ]
        raise StructuredOutputError(f"unreachable: {schema.__name__}", raw=raw)

    # -- lifecycle -------------------------------------------------------

    async def health_report(self) -> dict[str, tuple[bool, str]]:
        """Probe every configured profile in the way it is actually used.

        An embedding model cannot answer a chat request, so probing it with one
        reports a healthy runtime as broken. The alias the routing table points
        at for embeddings is the authoritative signal for which probe to send.
        """
        embedding_alias = self.settings.models.routing.embedding
        out: dict[str, tuple[bool, str]] = {}
        for alias in self.settings.models.profiles:
            try:
                if alias == embedding_alias:
                    out[alias] = await self._embedding_health(alias)
                else:
                    out[alias] = await self.get(alias).health()
            except Exception as exc:  # noqa: BLE001 - doctor reports, never raises
                out[alias] = (False, f"{type(exc).__name__}: {exc}")
        return out

    async def _embedding_health(self, alias: str) -> tuple[bool, str]:
        model = self.get(alias)
        started = time.perf_counter()
        try:
            vectors = await model.embed(["health probe"])
        except Exception as exc:  # noqa: BLE001 - the probe reports, never raises
            return False, f"{type(exc).__name__}: {exc}"
        if not vectors or not vectors[0]:
            return False, f"{model.model} returned an empty embedding"
        elapsed = time.perf_counter() - started
        return True, f"{model.model} returned a {len(vectors[0])}-dim vector in {elapsed:.2f}s"

    async def close(self) -> None:
        for model in self._models.values():
            await model.close()
        self._models.clear()


def extract_json(text: str) -> Any:
    """Pull a JSON value out of a possibly chatty local-model reply."""
    if not text or not text.strip():
        raise StructuredOutputError("model returned an empty response", raw=text)

    candidates: list[str] = []
    fenced = _JSON_FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text)

    for candidate in candidates:
        stripped = candidate.strip()
        for attempt in (stripped, _repair(stripped), _balanced_slice(stripped)):
            if not attempt:
                continue
            try:
                parsed = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            # A local model sometimes echoes the schema instead of an instance.
            if isinstance(parsed, dict) and "properties" in parsed and "type" in parsed:
                continue
            return parsed

    # Last resort: a model that was "thinking in Python" emits a dict literal with
    # single quotes, which is not JSON. literal_eval parses only literals, so it
    # cannot execute anything from the model output.
    for candidate in candidates:
        try:
            parsed = ast.literal_eval(_balanced_slice_raw(candidate.strip()) or candidate.strip())
        except (ValueError, SyntaxError, MemoryError, TypeError):
            continue
        if isinstance(parsed, dict | list):
            return parsed

    raise StructuredOutputError("no JSON object found in model output", raw=text[:2000])


def _repair(text: str) -> str:
    repaired = _TRAILING_COMMA.sub(r"\1", text)
    repaired = repaired.replace("\u201c", '"').replace("\u201d", '"')
    repaired = repaired.replace("\u2018", "'").replace("\u2019", "'")
    # Python literals leaking out of a model that was thinking in Python.
    repaired = re.sub(r"\bNone\b", "null", repaired)
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    return repaired


def _balanced_slice(text: str) -> str:
    """First balanced JSON region, repaired. See :func:`_balanced_slice_raw`."""
    sliced = _balanced_slice_raw(text)
    return _repair(sliced) if sliced else ""


def _balanced_slice_raw(text: str) -> str:
    """First balanced {...} or [...] region, ignoring braces inside strings."""
    start = None
    opener = closer = ""
    for index, char in enumerate(text):
        if char in "{[":
            start = index
            opener = char
            closer = "}" if char == "{" else "]"
            break
    if start is None:
        return ""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


_router: ModelRouter | None = None


def get_router(settings: Settings | None = None) -> ModelRouter:
    global _router
    if _router is None:
        _router = ModelRouter(settings)
    return _router


def reset_router() -> None:
    global _router
    _router = None
