"""Authentication and network posture (ADR 6.2 C1, 13.4, 16.5)."""

from __future__ import annotations

import hmac
from typing import Any
import ipaddress
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import StrEnum

from fastapi import HTTPException, Request, status

from mimir.config import Settings, get_settings
from mimir.logging import get_logger
from mimir.redaction import register_secret

log = get_logger(__name__)


class Surface(StrEnum):
    """Which surface a route belongs to."""

    PUBLIC_INFERENCE = "public_inference"
    """Reachable through the tunnel with a valid key. Model responses only."""

    LOCAL_PRIVILEGED = "local_privileged"
    """Loopback only, always. Execution, filesystem, cluster, database."""

    LOCAL_UI = "local_ui"
    """Loopback or authenticated, for the web UI and CLI."""


@dataclass(slots=True)
class Caller:
    address: str
    is_loopback: bool
    authenticated: bool
    key_id: str | None = None

    @property
    def origin(self) -> str:
        return "local" if self.is_loopback else "remote"


def _client_address(request: Request) -> str:
    """The immediate peer address."""
    return request.client.host if request.client else ""


def is_loopback(address: str) -> bool:
    if not address:
        return False
    try:
        return ipaddress.ip_address(address).is_loopback
    except ValueError:
        return address in ("localhost", "::1")


class RateLimiter:
    """Fixed-window limiter for the public facade (ADR R8)."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.time()
        window = self._hits[key]
        while window and now - window[0] > 60.0:
            window.popleft()
        if len(window) >= self.per_minute:
            return False
        window.append(now)
        return True


class Authenticator:
    def __init__(self, settings: Settings | None = None) -> None:
        from mimir import config as _config

        self.settings = settings or _config.get_settings()
        for key in self.settings.api.api_keys:
            register_secret(key)
        self.limiter = RateLimiter(self.settings.api.facade_rate_limit_per_minute)

    def _extract_key(self, request: Request) -> str | None:
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            return header[7:].strip()
        return request.headers.get("x-api-key") or None

    def _valid(self, presented: str | None) -> str | None:
        return validate_key(self.settings, presented)

    def identify(self, request: Request) -> Caller:
        address = _client_address(request)
        loopback = is_loopback(address)
        key_id = self._valid(self._extract_key(request))
        return Caller(
            address=address,
            is_loopback=loopback,
            authenticated=key_id is not None,
            key_id=key_id,
        )

    def authorise(self, request: Request, surface: Surface) -> Caller:
        caller = self.identify(request)
        api = self.settings.api

        privileged_remote = surface == Surface.LOCAL_PRIVILEGED and not caller.is_loopback
        if privileged_remote and not api.expose_privileged_routes_publicly:
            log.warning(
                    "privileged_route_refused",
                address=caller.address,
                path=request.url.path,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "this endpoint is loopback-only by design (ADR 16.5); "
                    "privileged execution is never exposed through the public endpoint"
                ),
            )

        if caller.is_loopback and api.allow_loopback_without_auth:
            return caller

        if not active_keys(self.settings):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "no API keys are configured, so non-loopback access is refused; "
                    "run 'mimir keys create' before exposing this service"
                ),
            )

        if not caller.authenticated:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing or invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if surface == Surface.PUBLIC_INFERENCE and not self.limiter.check(
            caller.key_id or caller.address
        ):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="rate limit exceeded",
                headers={"Retry-After": "60"},
            )
        return caller


_authenticator: Authenticator | None = None


def get_authenticator(settings: Settings | None = None) -> Authenticator:
    global _authenticator
    if _authenticator is None:
        _authenticator = Authenticator(settings)
    return _authenticator


def reset_authenticator() -> None:
    global _authenticator
    _authenticator = None


def require(surface: Surface):
    async def dependency(request: Request) -> Caller:
        return get_authenticator().authorise(request, surface)

    return dependency


require_local = require(Surface.LOCAL_PRIVILEGED)
require_ui = require(Surface.LOCAL_UI)
require_inference = require(Surface.PUBLIC_INFERENCE)


def active_keys(settings: Any) -> int:
    """Labelled keys not revoked, plus legacy plaintext ones."""
    return sum(1 for k in settings.api.keys if not k.revoked) + len(settings.api.api_keys)


def key_digest(key: str) -> str:
    import hashlib

    return hashlib.sha256(key.encode()).hexdigest()


def validate_key(settings: Any, presented: str | None) -> str | None:
    """The label of the key presented, or None; constant-time on both stores."""
    if not presented:
        return None
    digest = key_digest(presented)
    for entry in settings.api.keys:
        if not entry.revoked and hmac.compare_digest(digest, entry.sha256):
            return entry.label
    for index, key in enumerate(settings.api.api_keys):
        if hmac.compare_digest(presented, key):
            return f"legacy{index}"
    return None


def generate_api_key() -> str:
    import secrets

    return "mimir_" + secrets.token_urlsafe(32)
