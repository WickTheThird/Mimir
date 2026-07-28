"""Authentication and network posture (ADR 6.2 C1, 13.4, 16.5).

Two rules the ADR states plainly:

* Privileged helpers stay bound to loopback. Only the authenticated inference
  facade is ever exposed through Cloudflare (ADR 16.5).
* Non-loopback access requires authentication (ADR 6.2 C1).

The distinction that matters here is not "is the caller authenticated" but
"which surface is the caller on". A valid API key gets you the inference facade.
It does not get you shell, Kubernetes, SDM, database, or filesystem execution,
because those routes refuse non-loopback callers regardless of credentials.
"""

from __future__ import annotations

import hmac
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
    """The immediate peer address.

    Deliberately ignores X-Forwarded-For. A proxy header is attacker-controlled
    input, and trusting it here would let a remote caller claim to be loopback
    and reach the privileged surface. If a reverse proxy is ever put in front of
    MIMIR, this needs an explicit trusted-proxy allow list, not a header read.
    """
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
        self.settings = settings or get_settings()
        for key in self.settings.api.api_keys:
            register_secret(key)
        self.limiter = RateLimiter(self.settings.api.facade_rate_limit_per_minute)

    def _extract_key(self, request: Request) -> str | None:
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            return header[7:].strip()
        return request.headers.get("x-api-key") or None

    def _valid(self, presented: str | None) -> str | None:
        if not presented:
            return None
        for index, key in enumerate(self.settings.api.api_keys):
            # Constant-time comparison: a timing oracle on a key check is cheap
            # to avoid and this endpoint is internet reachable.
            if hmac.compare_digest(presented, key):
                return f"key{index}"
        return None

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

        if not api.api_keys:
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


def generate_api_key() -> str:
    import secrets

    return "mimir_" + secrets.token_urlsafe(32)
