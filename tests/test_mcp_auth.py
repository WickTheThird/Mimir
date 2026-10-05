"""A key buys the three MCP tools over HTTP; loopback passes; nothing else does."""

import pytest

from mimir.mcp.server import KeyRequired


class Settings:
    class api:
        api_keys = ["sk-good"]
        keys = []
        allow_loopback_without_auth = True
        facade_rate_limit_per_minute = 120
        max_request_bytes = 1_000_000


async def _call(middleware, client, headers=()):
    sent = []

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})

    async def send(msg):
        sent.append(msg)

    scope = {"type": "http", "client": (client, 1234), "path": "/mcp",
             "headers": [(k.encode(), v.encode()) for k, v in headers]}
    mw = KeyRequired(app, Settings())
    await mw(scope, None, send)
    return sent[0]["status"]


@pytest.mark.asyncio
async def test_loopback_passes_without_a_key():
    assert await _call(None, "127.0.0.1") == 200


@pytest.mark.asyncio
async def test_remote_without_a_key_is_refused():
    assert await _call(None, "203.0.113.5") == 401


@pytest.mark.asyncio
async def test_remote_with_the_key_passes_bearer_or_header():
    assert await _call(None, "203.0.113.5", [("authorization", "Bearer sk-good")]) == 200
    assert await _call(None, "203.0.113.5", [("x-api-key", "sk-good")]) == 200


@pytest.mark.asyncio
async def test_a_wrong_key_is_refused():
    assert await _call(None, "203.0.113.5", [("authorization", "Bearer sk-bad")]) == 401
