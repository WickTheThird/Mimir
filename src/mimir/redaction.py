"""Secret redaction (ADR 13.4, 20)."""

from __future__ import annotations

import re
from typing import Any

REDACTED = "[REDACTED]"

# Ordered most specific first so that a token is not partially matched by a
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "private_key",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S
        ),
    ),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("google_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("bearer", re.compile(r"(?i)\b(bearer|token)\s+([A-Za-z0-9._~+/=-]{16,})")),
    ("basic_auth_url", re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)([^/\s:@]+):([^/\s@]+)@")),
    ("pg_url", re.compile(r"(?i)\b(postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://([^:\s/]+):([^@\s]+)@")),
    (
        "assignment",
        re.compile(
            r"(?i)\b([A-Za-z0-9_.-]*(?:password|passwd|secret|token|api[_-]?key|apikey|"
            r"access[_-]?key|private[_-]?key|credential|auth)[A-Za-z0-9_.-]*)"
            r"(\s*[:=]\s*)([\"']?)([^\s\"',;}]{6,})\3"
        ),
    ),
    (
        # Prose form: "my password is hunter2", "the token was abc123".
        "prose_credential",
        re.compile(
            r"(?i)\b((?:my |the |a |your |our )?"
            r"(?:password|passwd|secret|token|api[ _-]?key|access[ _-]?key|"
            r"credential|passphrase))\s+(?:is|was|=)\s+"
            r"([\"']?)((?=\S*\d)(?=\S{8,})[^\s\"',;]+|[^\s\"',;]*[^\w\s][^\s\"',;]{7,})\2"
        ),
    ),
    ("k8s_secret_data", re.compile(r"(?m)^(\s*[A-Za-z0-9_.-]+:\s*)([A-Za-z0-9+/]{40,}={0,2})\s*$")),
]

# Values registered at runtime, for example configured API keys, that must never
_LITERALS: set[str] = set()


def register_secret(value: str | None) -> None:
    """Register a known literal secret so it is always scrubbed."""
    if value and len(value) >= 6:
        _LITERALS.add(value)


def clear_registered_secrets() -> None:
    _LITERALS.clear()


def _sub_assignment(match: re.Match[str]) -> str:
    return f"{match.group(1)}{match.group(2)}{match.group(3)}{REDACTED}{match.group(3)}"


def _sub_prose(match: re.Match[str]) -> str:
    return f"{match.group(1)} is {match.group(2)}{REDACTED}{match.group(2)}"


def _sub_bearer(match: re.Match[str]) -> str:
    return f"{match.group(1)} {REDACTED}"


def _sub_url_auth(match: re.Match[str]) -> str:
    return f"{match.group(1)}{match.group(2)}:{REDACTED}@"


def _sub_k8s(match: re.Match[str]) -> str:
    return f"{match.group(1)}{REDACTED}"


_SPECIAL_SUBS = {
    "assignment": _sub_assignment,
    "prose_credential": _sub_prose,
    "bearer": _sub_bearer,
    "basic_auth_url": _sub_url_auth,
    "pg_url": _sub_url_auth,
    "k8s_secret_data": _sub_k8s,
}


def redact(text: str, *, enabled: bool = True) -> str:
    """Return ``text`` with likely secrets replaced."""
    if not enabled or not text:
        return text
    out = text
    for literal in _LITERALS:
        out = out.replace(literal, REDACTED)
    for name, pattern in _PATTERNS:
        sub = _SPECIAL_SUBS.get(name)
        out = pattern.sub(sub if sub else REDACTED, out)
    return out


def find_secrets(text: str) -> list[str]:
    """Names of the rules that fired. Used by hooks and tests, not for display."""
    hits = []
    for name, pattern in _PATTERNS:
        if pattern.search(text):
            hits.append(name)
    if any(literal in text for literal in _LITERALS):
        hits.append("registered_literal")
    return hits


def redact_mapping(data: dict[str, Any], *, enabled: bool = True) -> dict[str, Any]:
    """Recursively redact string values in a mapping."""
    if not enabled:
        return data
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str):
            out[key] = redact(value)
        elif isinstance(value, dict):
            out[key] = redact_mapping(value)
        elif isinstance(value, list):
            out[key] = [redact(v) if isinstance(v, str) else v for v in value]
        else:
            out[key] = value
    return out


_ENV_SENSITIVE = re.compile(
    r"(?i)(token|secret|password|passwd|key|credential|auth|session|cookie)"
)


def safe_env_snapshot(env: dict[str, str], keep: tuple[str, ...] = ()) -> dict[str, str]:
    """Environment metadata worth auditing, with sensitive names dropped."""
    out = {}
    for key, value in env.items():
        if key in keep:
            out[key] = value
        elif _ENV_SENSITIVE.search(key):
            continue
        else:
            out[key] = value
    return out
