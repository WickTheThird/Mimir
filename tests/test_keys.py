"""Named keys stored as hashes, revoked independently, shared by facade and MCP."""

import pytest

from mimir.api.auth import generate_api_key, key_digest, validate_key
from mimir.config import ApiKeyEntry


class S:
    class api:
        api_keys = ["legacy-plain"]
        keys = [ApiKeyEntry(label="warp-cloud", sha256=key_digest("k-warp")),
                ApiKeyEntry(label="wick-local", sha256=key_digest("k-wick"), revoked=True)]
        allow_loopback_without_auth = True
        facade_rate_limit_per_minute = 2
        max_request_bytes = 100


def test_a_key_is_32_random_bytes():
    assert len(generate_api_key()) >= 6 + 43  # prefix + urlsafe(32)


def test_keys_resolve_to_their_label_and_revoked_ones_do_not():
    assert validate_key(S, "k-warp") == "warp-cloud"
    assert validate_key(S, "k-wick") is None
    assert validate_key(S, "legacy-plain") == "legacy0"
    assert validate_key(S, "nope") is None and validate_key(S, None) is None


async def _call(client, headers=(), length=None):
    from mimir.mcp.server import KeyRequired
    sent = []
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
    async def send(msg): sent.append(msg)
    hdrs = [(k.encode(), v.encode()) for k, v in headers]
    if length is not None: hdrs.append((b"content-length", str(length).encode()))
    mw = KeyRequired(app, S)
    await mw({"type": "http", "client": (client, 1), "path": "/mcp", "headers": hdrs}, None, send)
    return sent[0]["status"], mw


@pytest.mark.asyncio
async def test_mcp_uses_the_same_validator_as_the_facade():
    assert (await _call("203.0.113.5", [("authorization", "Bearer k-warp")]))[0] == 200
    assert (await _call("203.0.113.5", [("authorization", "Bearer k-wick")]))[0] == 401


@pytest.mark.asyncio
async def test_oversized_requests_are_refused_before_the_app_sees_them():
    assert (await _call("203.0.113.5", [("x-api-key", "k-warp")], length=101))[0] == 413


@pytest.mark.asyncio
async def test_rate_limit_is_per_key():
    from mimir.mcp.server import KeyRequired
    sent = []
    async def app(scope, receive, send): await send({"type": "http.response.start", "status": 200, "headers": []})
    async def send(msg): sent.append(msg)
    mw = KeyRequired(app, S)
    scope = {"type": "http", "client": ("203.0.113.5", 1), "path": "/mcp", "headers": [(b"x-api-key", b"k-warp")]}
    for _ in range(3): await mw(scope, None, send)
    assert [m["status"] for m in sent if m["type"] == "http.response.start"] == [200, 200, 429]
