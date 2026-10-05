"""Model runtime state."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

from mimir.config import Settings, get_settings

MIN_PULL_BYTES = 1 << 20
"""Blobs smaller than this are metadata, not a download worth reporting."""


@dataclass
class LoadedModel:
    name: str
    size_bytes: int = 0
    vram_bytes: int = 0
    quantisation: str = ""
    parameter_size: str = ""
    context_length: int = 0
    expires_at: float | None = None

    @property
    def ttl_s(self) -> float | None:
        if self.expires_at is None:
            return None
        return max(0.0, self.expires_at - time.time())

    @property
    def on_gpu(self) -> bool:
        return self.vram_bytes > 0 and self.vram_bytes >= self.size_bytes * 0.9


@dataclass
class RoleBinding:
    """A configured role and what the runtime is doing about it."""

    role: str
    alias: str
    model: str
    runtime: str
    loaded: bool = False
    installed: bool = True
    note: str = ""


@dataclass
class PullProgress:
    """An in-flight ``ollama pull``, measured from partial blobs on disk."""

    blob: str
    downloaded_bytes: int
    total_bytes: int
    modified_at: float

    @property
    def fraction(self) -> float:
        if self.total_bytes <= 0:
            return 0.0
        return min(1.0, self.downloaded_bytes / self.total_bytes)

    @property
    def stale_s(self) -> float:
        return max(0.0, time.time() - self.modified_at)


@dataclass
class RuntimeState:
    reachable: bool = False
    endpoint: str = ""
    version: str = ""
    error: str = ""
    loaded: list[LoadedModel] = field(default_factory=list)
    bindings: list[RoleBinding] = field(default_factory=list)
    installed: list[str] = field(default_factory=list)
    pulls: list[PullProgress] = field(default_factory=list)

    @property
    def vram_bytes(self) -> int:
        return sum(m.vram_bytes for m in self.loaded)


def normalise_tag(name: str) -> str:
    """Ollama's implicit ``:latest``."""
    if not name or ":" in name:
        return name
    return f"{name}:latest"


def _parse_expiry(raw: str) -> float | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).timestamp()
    except ValueError:
        return None


def _blob_progress(root: Path | None = None) -> list[PullProgress]:
    base = root or (Path.home() / ".ollama" / "models" / "blobs")
    if not base.is_dir():
        return []
    out: list[PullProgress] = []
    for path in base.glob("*-partial*"):
        if path.suffix in (".json", ".lock"):
            continue
        # Manifests and config blobs are a few dozen bytes and are always at
        try:
            stat = path.stat()
        except OSError:
            continue
        # Manifests and config blobs are a few dozen bytes and always read
        if stat.st_size < MIN_PULL_BYTES:
            continue
        # st_blocks counts 512-byte blocks actually allocated.
        real = getattr(stat, "st_blocks", 0) * 512
        out.append(
            PullProgress(
                blob=path.name.split("-partial")[0][:19],
                downloaded_bytes=min(real, stat.st_size) if real else 0,
                total_bytes=stat.st_size,
                modified_at=stat.st_mtime,
            )
        )
    out.sort(key=lambda p: p.total_bytes, reverse=True)
    return out


def probe(settings: Settings | None = None, *, timeout: float = 3.0) -> RuntimeState:
    """One read-only sweep of the runtime."""
    active = settings or get_settings()
    routing = active.models.routing
    state = RuntimeState()

    roles = [
        (role, alias)
        for role, alias in sorted(routing.model_dump().items())
        if isinstance(alias, str) and alias
    ]
    primary = active.models.profiles.get(routing.default)
    if primary is None:
        state.error = f"routing default {routing.default!r} has no profile"
        return state

    root = primary.base_url.rsplit("/v1", 1)[0]
    state.endpoint = root

    if primary.runtime == "echo":
        state.reachable = True
        state.version = "n/a (echo runtime)"
    else:
        try:
            version = httpx.get(f"{root}/api/version", timeout=timeout)
            state.reachable = version.status_code == 200
            if state.reachable:
                state.version = str(version.json().get("version", ""))
        except (httpx.HTTPError, ValueError) as exc:
            state.error = f"{type(exc).__name__}: {exc}"

    if state.reachable and primary.runtime != "echo":
        try:
            response = httpx.get(f"{root}/api/ps", timeout=timeout)
            if response.status_code == 200:
                for entry in response.json().get("models", []):
                    details = entry.get("details") or {}
                    state.loaded.append(
                        LoadedModel(
                            name=entry.get("name", ""),
                            size_bytes=int(entry.get("size") or 0),
                            vram_bytes=int(entry.get("size_vram") or 0),
                            quantisation=details.get("quantization_level", ""),
                            parameter_size=details.get("parameter_size", ""),
                            context_length=int(entry.get("context_length") or 0),
                            expires_at=_parse_expiry(entry.get("expires_at", "")),
                        )
                    )
        except (httpx.HTTPError, ValueError):
            pass
        try:
            tags = httpx.get(f"{root}/api/tags", timeout=timeout)
            if tags.status_code == 200:
                state.installed = [
                    entry.get("name", "") for entry in tags.json().get("models", [])
                ]
        except (httpx.HTTPError, ValueError):
            pass

    resident = {normalise_tag(m.name) for m in state.loaded}
    installed = {normalise_tag(name) for name in state.installed}
    for role, alias in roles:
        profile = active.models.profiles.get(alias)
        if profile is None:
            state.bindings.append(
                RoleBinding(role, alias, "<unconfigured>", "unknown", note="no profile")
            )
            continue
        tag = normalise_tag(profile.model)
        is_installed = (
            True if not state.installed or profile.runtime == "echo" else tag in installed
        )
        state.bindings.append(
            RoleBinding(
                role=role,
                alias=alias,
                model=profile.model,
                runtime=profile.runtime,
                loaded=tag in resident,
                installed=is_installed,
                note="" if is_installed else "not pulled",
            )
        )

    state.pulls = _blob_progress()
    return state
