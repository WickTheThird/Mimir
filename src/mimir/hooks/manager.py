"""Lifecycle hooks (ADR 10.4)."""

from __future__ import annotations

import asyncio
import json
import shlex
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from mimir.config import Settings, get_settings
from mimir.logging import get_logger

log = get_logger(__name__)


class HookEvent(StrEnum):
    BEFORE_TOOL = "before_tool"
    AFTER_TOOL = "after_tool"
    BEFORE_MUTATION = "before_mutation"
    AFTER_MUTATION = "after_mutation"
    ON_APPROVAL_REQUEST = "on_approval_request"
    ON_SESSION_COMPLETE = "on_session_complete"
    ON_MEMORY_PROMOTION = "on_memory_promotion"
    ON_WEB_INGEST = "on_web_ingest"


@dataclass(slots=True)
class HookVerdict:
    allowed: bool = True
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    require_extra_approval: bool = False
    mutated_payload: Any = None

    @classmethod
    def deny(cls, reason: str) -> HookVerdict:
        return cls(allowed=False, reason=reason)


@dataclass(slots=True)
class HookContext:
    event: HookEvent
    payload: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    tool_name: str | None = None
    settings: Settings | None = None


HookFn = Callable[[HookContext], Awaitable[HookVerdict | None]]


@dataclass(slots=True)
class ExternalHook:
    """A hook implemented as a local command (Claude Code style)."""

    event: HookEvent
    command: str
    timeout_s: float = 15.0
    blocking: bool = True
    """When true a non-zero exit denies the action."""

    name: str = ""

    async def run(self, ctx: HookContext) -> HookVerdict:
        argv = shlex.split(self.command)
        payload = json.dumps(
            {
                "event": ctx.event.value,
                "session_id": ctx.session_id,
                "tool": ctx.tool_name,
                "payload": ctx.payload,
            },
            default=str,
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await asyncio.wait_for(
                process.communicate(payload.encode()), timeout=self.timeout_s
            )
        except TimeoutError:
            message = f"hook '{self.name or self.command}' timed out after {self.timeout_s}s"
            log.warning("hook_timeout", hook=self.name, command=self.command)
            return HookVerdict.deny(message) if self.blocking else HookVerdict()
        except (OSError, ValueError) as exc:
            message = f"hook '{self.name or self.command}' failed to start: {exc}"
            log.warning("hook_start_failed", hook=self.name, error=str(exc))
            return HookVerdict.deny(message) if self.blocking else HookVerdict()

        stdout = stdout_b.decode("utf-8", "replace").strip()
        stderr = stderr_b.decode("utf-8", "replace").strip()
        if process.returncode != 0 and self.blocking:
            return HookVerdict.deny(
                f"hook '{self.name or self.command}' rejected the action: {stderr or stdout}"
            )

        metadata: dict[str, Any] = {}
        if stdout.startswith("{"):
            try:
                parsed = json.loads(stdout)
                if isinstance(parsed, dict):
                    metadata = parsed
                    if parsed.get("deny"):
                        return HookVerdict.deny(str(parsed.get("reason", "denied by hook")))
            except json.JSONDecodeError:
                pass
        return HookVerdict(metadata=metadata)


class HookManager:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._hooks: dict[HookEvent, list[tuple[str, HookFn]]] = {e: [] for e in HookEvent}
        self._external: dict[HookEvent, list[ExternalHook]] = {e: [] for e in HookEvent}

    # -- registration -----------------------------------------------------

    def register(self, event: HookEvent, fn: HookFn, name: str = "") -> Callable[[], None]:
        entry = (name or getattr(fn, "__name__", "hook"), fn)
        self._hooks[event].append(entry)

        def remove() -> None:
            if entry in self._hooks[event]:
                self._hooks[event].remove(entry)

        return remove

    def register_external(self, hook: ExternalHook) -> None:
        self._external[hook.event].append(hook)

    def load_config(self, path: Path | None = None) -> int:
        """Load external hooks from ``$MIMIR_HOME/hooks.yaml``."""
        config_path = path or (self.settings.home / "hooks.yaml")
        if not config_path.is_file():
            return 0
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        entries = data.get("hooks", {}) if isinstance(data, dict) else {}
        loaded = 0
        for event_name, hooks in entries.items():
            try:
                event = HookEvent(event_name)
            except ValueError:
                log.warning("unknown_hook_event", event=event_name)
                continue
            for item in hooks or []:
                self.register_external(
                    ExternalHook(
                        event=event,
                        command=item["command"],
                        timeout_s=float(item.get("timeout_s", 15.0)),
                        blocking=bool(item.get("blocking", True)),
                        name=item.get("name", ""),
                    )
                )
                loaded += 1
        log.info("hooks_loaded", count=loaded, path=str(config_path))
        return loaded

    # -- dispatch ---------------------------------------------------------

    async def emit(self, ctx: HookContext) -> HookVerdict:
        ctx.settings = ctx.settings or self.settings
        merged = HookVerdict()
        for name, fn in self._hooks[ctx.event]:
            try:
                verdict = await fn(ctx)
            except Exception as exc:
                log.exception("hook_error", hook=name, event=ctx.event.value)
                verdict = HookVerdict.deny(f"hook '{name}' raised {type(exc).__name__}: {exc}")
            if verdict is None:
                continue
            merged.metadata.update(verdict.metadata)
            merged.require_extra_approval |= verdict.require_extra_approval
            if verdict.mutated_payload is not None:
                merged.mutated_payload = verdict.mutated_payload
            if not verdict.allowed:
                merged.allowed = False
                merged.reason = verdict.reason
                return merged

        for hook in self._external[ctx.event]:
            verdict = await hook.run(ctx)
            merged.metadata.update(verdict.metadata)
            if not verdict.allowed:
                merged.allowed = False
                merged.reason = verdict.reason
                return merged
        return merged

    # -- typed shortcuts used by the tool contract and executor ------------

    async def before_tool(self, spec: Any, args: Any, ctx: Any) -> HookVerdict:
        return await self.emit(
            HookContext(
                event=HookEvent.BEFORE_TOOL,
                tool_name=spec.name,
                session_id=getattr(ctx, "session_id", None),
                payload={
                    "tool": spec.name,
                    "capability": spec.capability.value,
                    "risk": spec.risk.value,
                    "mutating": spec.mutating,
                    "arguments": args.model_dump() if hasattr(args, "model_dump") else {},
                },
            )
        )

    async def after_tool(self, spec: Any, args: Any, result: Any, ctx: Any) -> Any:
        verdict = await self.emit(
            HookContext(
                event=HookEvent.AFTER_TOOL,
                tool_name=spec.name,
                session_id=getattr(ctx, "session_id", None),
                payload={
                    "tool": spec.name,
                    "ok": result.ok,
                    "summary": result.summary[:500],
                    "artifact_ref": result.artifact_ref,
                },
            )
        )
        if verdict.mutated_payload is not None:
            return verdict.mutated_payload
        if verdict.metadata:
            result.data.setdefault("_hook_metadata", verdict.metadata)
        return result

    async def before_mutation(self, command: Any, assessment: Any) -> HookVerdict:
        return await self.emit(
            HookContext(
                event=HookEvent.BEFORE_MUTATION,
                payload={
                    "command": command.display,
                    "risk": assessment.risk.value,
                    "context": command.context.model_dump(exclude_none=True),
                    "reversible": assessment.reversible,
                },
            )
        )

    async def after_mutation(self, command: Any, record: Any) -> HookVerdict:
        return await self.emit(
            HookContext(
                event=HookEvent.AFTER_MUTATION,
                payload={
                    "command": command.display,
                    "outcome": record.outcome.value,
                    "exit_code": record.exit_code,
                },
            )
        )

    async def on_approval_request(self, command: Any, assessment: Any) -> HookVerdict:
        return await self.emit(
            HookContext(
                event=HookEvent.ON_APPROVAL_REQUEST,
                payload={"command": command.display, "risk": assessment.risk.value},
            )
        )

    async def on_session_complete(self, state: Any) -> HookVerdict:
        return await self.emit(
            HookContext(
                event=HookEvent.ON_SESSION_COMPLETE,
                session_id=getattr(state, "session_id", None),
                payload={
                    "task_type": getattr(state.task_type, "value", None),
                    "evidence_count": len(getattr(state, "evidence", [])),
                    "confidence": getattr(state, "final_confidence", 0.0),
                },
            )
        )

    async def on_memory_promotion(self, proposal: Any) -> HookVerdict:
        return await self.emit(
            HookContext(
                event=HookEvent.ON_MEMORY_PROMOTION,
                payload={
                    "title": proposal.title,
                    "category": proposal.category,
                    "verification_status": proposal.verification_status,
                },
            )
        )

    async def on_web_ingest(self, url: str, content: str) -> HookVerdict:
        return await self.emit(
            HookContext(
                event=HookEvent.ON_WEB_INGEST,
                payload={"url": url, "bytes": len(content)},
            )
        )


_manager: HookManager | None = None


def get_hook_manager(settings: Settings | None = None) -> HookManager:
    global _manager
    if _manager is None:
        _manager = HookManager(settings)
        from mimir.hooks.builtin import register_builtin_hooks

        register_builtin_hooks(_manager)
        _manager.load_config()
    return _manager


def reset_hook_manager() -> None:
    global _manager
    _manager = None
