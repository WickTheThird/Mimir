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


class ModelRouter:
    """Resolves task classes to models and owns the shared client lifecycle."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._models: dict[str, ChatModel] = {}
        self.call_log: list[ModelCallRecord] = []

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
    ) -> ChatResponse:
        model = self.for_task(task_class)
        budget = int(model.context_window * 0.75)
        trimmed = list(messages)
        if messages_token_estimate(trimmed) > budget:
            trimmed = trim_to_context(trimmed, budget)
            log.info(
                "context_trimmed",
                alias=model.alias,
                estimated_tokens=messages_token_estimate(trimmed),
                budget=budget,
            )

        started = time.time()
        last_error: ModelError | None = None
        for attempt in range(retries + 1):
            try:
                response = await model.chat(trimmed, options)
            except ModelError as exc:
                last_error = exc
                if not exc.retryable or attempt >= retries:
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
                        )
                    )
                    raise
                log.warning("model_retry", alias=model.alias, attempt=attempt + 1,
                            error=exc.message)
                continue
            self.call_log.append(
                ModelCallRecord.from_response(
                    response, session_id=session_id, purpose=purpose, started_at=started
                )
            )
            return response
        raise last_error or ModelError("model call failed")

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
