"""Network containment for offline evaluation."""

from __future__ import annotations

import ipaddress
import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from mimir.logging import get_logger

log = get_logger(__name__)

_PROXY_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "NO_PROXY",
    "no_proxy",
)


class OfflineViolation(OSError):
    """Raised inside a contained run when something reaches for the network."""


@dataclass
class ContainmentReport:
    blocked_hosts: list[str] = field(default_factory=list)
    allowed_loopback: int = 0

    @property
    def external_calls(self) -> int:
        return len(self.blocked_hosts)

    @property
    def clean(self) -> bool:
        return self.external_calls == 0

    def summary(self) -> str:
        if self.clean:
            return f"no external network attempts ({self.allowed_loopback} loopback)"
        unique = sorted(set(self.blocked_hosts))
        return (
            f"{self.external_calls} external attempt(s) blocked across "
            f"{len(unique)} host(s): {', '.join(unique[:6])}"
        )


def _is_loopback(host: str) -> bool:
    if host in ("localhost", "localhost.localdomain", "", "::1"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@contextmanager
def network_containment(*, enabled: bool = True) -> Iterator[ContainmentReport]:
    """Permit loopback, refuse and count everything else."""
    report = ContainmentReport()
    if not enabled:
        yield report
        return

    original_getaddrinfo = socket.getaddrinfo
    original_create_connection = socket.create_connection
    saved_env = {name: os.environ.pop(name, None) for name in _PROXY_VARS}

    def guarded_getaddrinfo(host, port, *args, **kwargs):  # type: ignore[no-untyped-def]
        name = str(host) if host is not None else ""
        if _is_loopback(name):
            report.allowed_loopback += 1
            return original_getaddrinfo(host, port, *args, **kwargs)
        report.blocked_hosts.append(name)
        log.warning("offline_network_blocked", host=name, port=port)
        raise OfflineViolation(
            f"offline evaluation blocked a network call to {name!r}. "
            "A tool that should not be reachable offline was invoked, or a tool "
            "is missing an offline_safe classification."
        )

    def guarded_create_connection(address, *args, **kwargs):  # type: ignore[no-untyped-def]
        host = address[0] if isinstance(address, tuple) and address else ""
        if _is_loopback(str(host)):
            report.allowed_loopback += 1
            return original_create_connection(address, *args, **kwargs)
        report.blocked_hosts.append(str(host))
        log.warning("offline_connection_blocked", host=host)
        raise OfflineViolation(f"offline evaluation blocked a connection to {host!r}")

    socket.getaddrinfo = guarded_getaddrinfo  # type: ignore[assignment]
    socket.create_connection = guarded_create_connection  # type: ignore[assignment]
    try:
        yield report
    finally:
        socket.getaddrinfo = original_getaddrinfo  # type: ignore[assignment]
        socket.create_connection = original_create_connection  # type: ignore[assignment]
        for name, value in saved_env.items():
            if value is not None:
                os.environ[name] = value
        if not report.clean:
            log.warning("offline_run_contaminated", summary=report.summary())
