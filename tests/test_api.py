"""API surface tests (ADR 6.2 C1/C9, 16).

The property that matters most: a valid API key buys the inference facade and
nothing else. Privileged execution stays loopback-only regardless of credentials
(ADR 16.5, NG5).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mimir.config import ModelProfile

API_KEY = "mimir_test_key_0123456789abcdef"


@pytest.fixture
def client(settings):
    for alias in list(settings.models.profiles):
        settings.models.profiles[alias] = ModelProfile(
            alias=alias, runtime="echo", model="echo"
        )
    settings.api.api_keys = [API_KEY]

    from mimir.api.auth import reset_authenticator
    from mimir.llm.router import reset_router

    reset_authenticator()
    reset_router()

    from mimir.api.app import create_app

    # TestClient presents as a non-loopback peer, which is exactly the case that
    # needs testing: it exercises the remote path rather than the local bypass.
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def auth():
    return {"Authorization": f"Bearer {API_KEY}"}


def test_health_is_unauthenticated(client):
    """A tunnel health check must not need a credential, or it will flap."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_inference_requires_a_key(client):
    assert client.get("/v1/models").status_code == 401
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 401


def test_invalid_key_is_refused(client):
    response = client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
    assert response.status_code == 401


def test_valid_key_reaches_inference(client, auth):
    response = client.get("/v1/models", headers=auth)
    assert response.status_code == 200
    ids = [m["id"] for m in response.json()["data"]]
    assert "mimir-local-ops" in ids


def test_chat_completions_is_openai_shaped(client, auth):
    response = client.post(
        "/v1/chat/completions",
        headers=auth,
        json={"model": "mimir-local-ops", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert "usage" in body


def test_chat_accepts_content_parts_shape(client, auth):
    """Some clients send content as a list of parts rather than a string."""
    response = client.post(
        "/v1/chat/completions",
        headers=auth,
        json={
            "model": "x",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        },
    )
    assert response.status_code == 200
    assert "hello" in response.json()["choices"][0]["message"]["content"]


def test_streaming_terminates_properly(client, auth):
    response = client.post(
        "/v1/chat/completions",
        headers=auth,
        json={"model": "x", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert response.status_code == 200
    assert "chat.completion.chunk" in response.text
    assert "[DONE]" in response.text


def test_api_key_does_not_grant_privileged_execution(client, auth):
    """ADR 16.5. This is the single most important API property.

    Warp reaches MIMIR through a public tunnel with a key. That key must not
    become a remote shell into the operator's machine.
    """
    for path, payload in [
        ("/api/investigations", {"question": "x"}),
        ("/api/investigations/stream", {"question": "x"}),
    ]:
        response = client.post(path, headers=auth, json=payload)
        assert response.status_code == 403, path
        assert "loopback-only" in response.json()["detail"]


def test_privileged_reads_are_also_refused_remotely(client, auth):
    for path in ["/api/sessions", "/api/approvals", "/api/artifacts/art_x"]:
        assert client.get(path, headers=auth).status_code == 403, path


def test_forwarded_header_cannot_forge_loopback(client, auth):
    """X-Forwarded-For is attacker-controlled and must be ignored."""
    response = client.post(
        "/api/investigations",
        headers={**auth, "X-Forwarded-For": "127.0.0.1", "X-Real-IP": "127.0.0.1"},
        json={"question": "x"},
    )
    assert response.status_code == 403


def test_correlation_id_is_returned(client):
    response = client.get("/api/health")
    assert response.headers.get("X-Correlation-Id")


def test_openapi_document_builds(client):
    """A broken route signature only shows up when the schema is generated."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/v1/chat/completions" in response.json()["paths"]
