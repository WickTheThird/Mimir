"""OpenAI-compatible runtime adapter (ADR 6.2 C3, 18.3).

One client covers Ollama, llama.cpp's server, vLLM, LM Studio, MLX-LM's server,
and LiteLLM, because they all expose ``/v1/chat/completions``. Runtime-specific
quirks are handled by small flags rather than by separate classes:

* Ollama accepts ``format: json`` but not full JSON Schema on older builds.
* llama.cpp ignores ``tool_choice`` and needs tools restated in the prompt.
* Some builds omit ``usage`` on streaming responses.

Nothing above this module needs to know which of those it is talking to.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from mimir.config import ModelProfile
from mimir.llm.base import (
    ChatModel,
    ChatResponse,
    ChunkType,
    GenerationOptions,
    LLMMessage,
    ModelError,
    StreamChunk,
    ToolCall,
    Usage,
)
from mimir.logging import get_logger
from mimir.redaction import register_secret

log = get_logger(__name__)

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenAICompatModel(ChatModel):
    def __init__(self, profile: ModelProfile, client: httpx.AsyncClient | None = None) -> None:
        self.profile = profile
        self.alias = profile.alias
        self.model = profile.model
        self.context_window = profile.context_window
        self.supports_tools = profile.supports_tools
        self.supports_json_schema = profile.supports_json_schema
        self._base_url = profile.base_url.rstrip("/")
        self._owns_client = client is None
        if profile.api_key:
            register_secret(profile.api_key)
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(profile.request_timeout_s, connect=10.0),
            headers=self._headers(),
            follow_redirects=False,
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.profile.api_key:
            headers["Authorization"] = f"Bearer {self.profile.api_key}"
        return headers

    # -- payload ---------------------------------------------------------

    def _build_payload(
        self,
        messages: Sequence[LLMMessage],
        options: GenerationOptions | None,
        stream: bool,
    ) -> dict[str, Any]:
        opts = options or GenerationOptions()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
            "stream": stream,
            "temperature": (
                opts.temperature if opts.temperature is not None else self.profile.temperature
            ),
            "max_tokens": opts.max_tokens or self.profile.max_output_tokens,
        }
        if opts.top_p is not None:
            payload["top_p"] = opts.top_p
        if opts.stop:
            payload["stop"] = list(opts.stop)
        if opts.seed is not None:
            payload["seed"] = opts.seed
        if opts.tools and self.supports_tools:
            payload["tools"] = opts.tools
            if opts.tool_choice is not None:
                payload["tool_choice"] = opts.tool_choice
        if opts.response_format is not None:
            payload["response_format"] = self._adapt_response_format(opts.response_format)
        if stream:
            # Ollama and vLLM only emit usage on the final chunk when asked.
            payload["stream_options"] = {"include_usage": True}
        payload.update(self.profile.extra_body)
        payload.update(opts.extra_body)
        return payload

    def _adapt_response_format(self, response_format: dict[str, Any]) -> dict[str, Any]:
        if self.supports_json_schema:
            return response_format
        # Degrade a json_schema request to plain JSON mode. The caller still
        # validates the result, so a runtime without schema support loses
        # guarantees but not correctness.
        return {"type": "json_object"}

    # -- requests --------------------------------------------------------

    async def chat(
        self, messages: Sequence[LLMMessage], options: GenerationOptions | None = None
    ) -> ChatResponse:
        payload = self._build_payload(messages, options, stream=False)
        started = time.perf_counter()
        timeout = (options.timeout_s if options else None) or self.profile.request_timeout_s

        try:
            response = await self._client.post(
                f"{self._base_url}/chat/completions", json=payload, timeout=timeout
            )
        except httpx.TimeoutException as exc:
            raise ModelError(
                f"model '{self.alias}' timed out after {timeout:.0f}s", retryable=True
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelError(
                f"cannot reach model runtime for '{self.alias}' at {self._base_url}: {exc}",
                retryable=True,
            ) from exc

        if response.status_code >= 400:
            raise ModelError(
                self._error_message(response),
                retryable=response.status_code in _RETRYABLE_STATUS,
                status=response.status_code,
            )

        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise ModelError(f"model returned non-JSON body: {response.text[:300]}") from exc

        return self._parse_response(body, time.perf_counter() - started)

    def _error_message(self, response: httpx.Response) -> str:
        detail = response.text[:400]
        try:
            body = response.json()
            detail = body.get("error", {}).get("message") or body.get("error") or detail
        except (json.JSONDecodeError, AttributeError):
            pass
        hint = ""
        if response.status_code == 404:
            hint = (
                f" (is model '{self.model}' pulled? try: ollama pull {self.model}, "
                "or check the alias in ~/.mimir/config.yaml)"
            )
        elif response.status_code in (401, 403):
            hint = " (check the api_key for this profile)"
        return f"model '{self.alias}' returned {response.status_code}: {detail}{hint}"

    def _parse_response(self, body: dict[str, Any], latency: float) -> ChatResponse:
        choices = body.get("choices") or []
        if not choices:
            raise ModelError(f"model returned no choices: {json.dumps(body)[:300]}")
        message = choices[0].get("message", {}) or {}
        raw_calls = message.get("tool_calls") or []
        usage_body = body.get("usage") or {}
        return ChatResponse(
            content=message.get("content") or "",
            tool_calls=[ToolCall.from_openai(c) for c in raw_calls],
            finish_reason=choices[0].get("finish_reason") or "stop",
            model=body.get("model", self.model),
            alias=self.alias,
            usage=Usage(
                prompt_tokens=usage_body.get("prompt_tokens", 0),
                completion_tokens=usage_body.get("completion_tokens", 0),
                total_tokens=usage_body.get("total_tokens", 0),
            ),
            latency_s=latency,
            raw=body,
        )

    # -- streaming -------------------------------------------------------

    async def stream(  # type: ignore[override]
        self, messages: Sequence[LLMMessage], options: GenerationOptions | None = None
    ) -> AsyncIterator[StreamChunk]:
        payload = self._build_payload(messages, options, stream=True)
        timeout = (options.timeout_s if options else None) or self.profile.request_timeout_s
        # Tool calls arrive as fragments across chunks and must be reassembled by
        # index before they mean anything.
        partial: dict[int, dict[str, Any]] = {}

        try:
            async with self._client.stream(
                "POST", f"{self._base_url}/chat/completions", json=payload, timeout=timeout
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    yield StreamChunk(type=ChunkType.ERROR, error=self._error_message(response))
                    return

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    for choice in event.get("choices") or []:
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if content:
                            yield StreamChunk(type=ChunkType.CONTENT, text=content)
                        for fragment in delta.get("tool_calls") or []:
                            index = fragment.get("index", 0)
                            slot = partial.setdefault(
                                index, {"id": "", "function": {"name": "", "arguments": ""}}
                            )
                            if fragment.get("id"):
                                slot["id"] = fragment["id"]
                            function = fragment.get("function") or {}
                            if function.get("name"):
                                slot["function"]["name"] = function["name"]
                            if function.get("arguments"):
                                slot["function"]["arguments"] += function["arguments"]
                        finish = choice.get("finish_reason")
                        if finish:
                            for slot in partial.values():
                                if slot["function"]["name"]:
                                    yield StreamChunk(
                                        type=ChunkType.TOOL_CALL,
                                        tool_call=ToolCall.from_openai(slot),
                                    )
                            partial.clear()
                            yield StreamChunk(type=ChunkType.DONE, finish_reason=finish)
                            return
        except httpx.TimeoutException:
            yield StreamChunk(
                type=ChunkType.ERROR, error=f"model '{self.alias}' timed out after {timeout:.0f}s"
            )
            return
        except httpx.HTTPError as exc:
            yield StreamChunk(
                type=ChunkType.ERROR,
                error=f"cannot reach model runtime for '{self.alias}': {exc}",
            )
            return

        # Some runtimes end the stream without a finish_reason.
        for slot in partial.values():
            if slot["function"]["name"]:
                yield StreamChunk(type=ChunkType.TOOL_CALL, tool_call=ToolCall.from_openai(slot))
        yield StreamChunk(type=ChunkType.DONE, finish_reason="stop")

    # -- embeddings ------------------------------------------------------

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            response = await self._client.post(
                f"{self._base_url}/embeddings",
                json={"model": self.model, "input": list(texts)},
            )
        except httpx.HTTPError as exc:
            raise ModelError(f"embedding request failed: {exc}", retryable=True) from exc
        if response.status_code >= 400:
            raise ModelError(self._error_message(response), status=response.status_code)
        body = response.json()
        return [item["embedding"] for item in body.get("data", [])]

    async def list_models(self) -> list[str]:
        """Used by ``mimir doctor`` to show what the runtime actually has."""
        try:
            response = await self._client.get(f"{self._base_url}/models", timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError:
            return []
        return [m.get("id", "") for m in response.json().get("data", [])]

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class OllamaModel(OpenAICompatModel):
    """Ollama speaks the OpenAI API at /v1 but has two quirks worth encoding."""

    def _adapt_response_format(self, response_format: dict[str, Any]) -> dict[str, Any]:
        # Ollama's OpenAI shim accepts json_schema on recent builds and
        # json_object everywhere. Prefer schema, fall back cleanly.
        if response_format.get("type") == "json_schema" and not self.supports_json_schema:
            return {"type": "json_object"}
        return response_format


class MLXModel(OpenAICompatModel):
    """mlx_lm.server is OpenAI-compatible.

    Start it with: ``python -m mlx_lm.server --model <hf-repo> --port 8080``.
    It does not implement /v1/embeddings, so embedding falls back to the
    configured embed profile.
    """

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise ModelError(
            "mlx_lm.server does not serve embeddings; point the 'embed' profile at "
            "Ollama or another runtime"
        )


def build_model(profile: ModelProfile) -> ChatModel:
    """Instantiate the adapter for a profile (ADR 25 keeps this swappable)."""
    if profile.runtime == "echo":
        from mimir.llm.base import EchoModel

        return EchoModel(alias=profile.alias)
    if profile.runtime == "ollama":
        return OllamaModel(profile)
    if profile.runtime == "mlx":
        return MLXModel(profile)
    # openai_compat, llamacpp, and litellm are all the same wire protocol.
    return OpenAICompatModel(profile)
