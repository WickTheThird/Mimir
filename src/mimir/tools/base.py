"""Typed helper tool contract (ADR 9).

The model never receives a single unrestricted shell tool. It receives a set of
constrained helpers with typed inputs and structured outputs. This module
defines that contract:

* :class:`ToolSpec`   - name, description, JSON schema, risk, capability tag.
* :class:`ToolResult` - structured output plus the evidence it produced.
* :class:`ToolRegistry` - lookup, filtering by specialist, OpenAI schema export.
* :func:`tool`        - decorator that derives a spec from a pydantic input model.

Handlers are async and receive a :class:`ToolContext` carrying the session id,
settings, artifact store, and approval callback.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field, ValidationError

from mimir.config import Settings, get_settings
from mimir.logging import get_logger
from mimir.models.command import ProposedCommand, RiskClass
from mimir.models.evidence import Evidence
from mimir.models.specialist import SpecialistName

log = get_logger(__name__)

InputT = TypeVar("InputT", bound=BaseModel)


class Capability(StrEnum):
    """Coarse capability groups, used for policy and for the ADR 16.5 rule that
    privileged surfaces stay loopback-only."""

    REPOSITORY = "repository"
    KUBERNETES = "kubernetes"
    SDM = "sdm"
    DATABASE = "database"
    LOGS = "logs"
    WEB = "web"
    MEMORY = "memory"
    SKILLS = "skills"
    SANDBOX = "sandbox"
    SHELL = "shell"
    INTERNAL = "internal"


#: Capabilities that touch only local, immutable-during-a-run state and may
#: therefore run in an offline evaluation. This is an ALLOWLIST on purpose: a new
#: capability is unsafe until somebody argues otherwise, which is the opposite of
#: the denylist that let web search into a supposedly offline benchmark.
#:
#: Deliberately excluded: WEB (mutable third-party state, and it transmits the
#: prompt off the machine), KUBERNETES, SDM, DATABASE (live infrastructure),
#: SHELL (arbitrary execution).
OFFLINE_SAFE_CAPABILITIES: frozenset[Capability] = frozenset(
    {
        Capability.REPOSITORY,
        Capability.LOGS,
        Capability.MEMORY,
        Capability.SKILLS,
        Capability.SANDBOX,
        Capability.INTERNAL,
    }
)

#: Capabilities that must never be reachable from the public inference facade.
PRIVILEGED_CAPABILITIES: frozenset[Capability] = frozenset(
    {
        Capability.KUBERNETES,
        Capability.SDM,
        Capability.DATABASE,
        Capability.SANDBOX,
        Capability.SHELL,
    }
)


class ToolError(Exception):
    """Raised by a handler when the call cannot be completed.

    Carries a machine-readable code so the graph can decide whether to retry,
    re-plan, or surface the failure to the user.
    """

    def __init__(self, message: str, *, code: str = "tool_error", retryable: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.retryable = retryable


class ApprovalRequired(ToolError):
    """Raised when a helper needs an approval that has not been granted."""

    def __init__(self, command: ProposedCommand, message: str = "approval required") -> None:
        super().__init__(message, code="approval_required")
        self.command = command


class ToolResult(BaseModel):
    """Uniform structured output from every helper."""

    ok: bool = True
    tool: str = ""
    summary: str = ""
    """One or two lines the model can read without pulling in the full payload."""

    data: dict[str, Any] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    artifact_ref: str | None = None
    truncated: bool = False
    error: str | None = None
    error_code: str | None = None
    duration_s: float = 0.0
    proposed_commands: list[ProposedCommand] = Field(default_factory=list)
    """Commands the helper wants to run but could not, pending approval."""

    def render(self, max_chars: int = 4000) -> str:
        """Compact text form fed back into the model context."""
        if not self.ok:
            return f"{self.tool} failed: {self.error}"
        parts = [self.summary.strip()] if self.summary else []
        if self.data:
            import json

            body = json.dumps(self.data, indent=2, default=str, ensure_ascii=False)
            if len(body) > max_chars:
                body = body[:max_chars] + f"\n...[truncated, {len(body) - max_chars} more chars]"
            parts.append(body)
        if self.truncated:
            parts.append("[output truncated; full text in artifact store]")
        return "\n".join(parts)

    @classmethod
    def failure(cls, tool: str, message: str, code: str = "tool_error") -> ToolResult:
        return cls(ok=False, tool=tool, error=message, error_code=code, summary=message)


@dataclass
class ToolContext:
    """Ambient services a handler may use."""

    settings: Settings = field(default_factory=get_settings)
    session_id: str | None = None
    specialist: SpecialistName | None = None
    artifacts: Any = None  # mimir.tools.artifacts.ArtifactStore
    executor: Any = None  # mimir.tools.exec.CommandExecutor
    approvals: Any = None  # mimir.safety.approvals.ApprovalBroker
    hooks: Any = None  # mimir.hooks.manager.HookManager
    environment: Any = None  # mimir.models.state.EnvironmentContext
    registry: Any = None  # mimir.tools.base.ToolRegistry
    """The registry a dispatching tool must resolve through.

    Tools that fan out to other tools (parallel_search) previously reached the
    global REGISTRY at call time, which let them invoke helpers that had been
    deliberately excluded from a filtered registry. Any tool that dispatches
    must use this when it is set."""
    extra: dict[str, Any] = field(default_factory=dict)

    def child(self, **overrides: Any) -> ToolContext:
        data = {
            "settings": self.settings,
            "session_id": self.session_id,
            "specialist": self.specialist,
            "artifacts": self.artifacts,
            "executor": self.executor,
            "approvals": self.approvals,
            "hooks": self.hooks,
            "environment": self.environment,
            "registry": self.registry,
            "extra": dict(self.extra),
        }
        data.update(overrides)
        return ToolContext(**data)


Handler = Callable[[Any, ToolContext], Awaitable[ToolResult]]


@dataclass(slots=True)
class ToolSpec(Generic[InputT]):
    name: str
    description: str
    input_model: type[InputT]
    handler: Handler
    capability: Capability
    risk: RiskClass = RiskClass.R1
    """Baseline risk. The policy engine may raise it based on arguments."""

    specialists: tuple[SpecialistName, ...] = ()
    """Which specialists may call this. Empty means all of them."""

    mutating: bool = False
    requires_approval: bool = False
    long_running: bool = False
    tags: tuple[str, ...] = ()
    offline_safe: bool | None = None
    """Whether this tool may run in an offline evaluation.

    ``None`` derives from :data:`OFFLINE_SAFE_CAPABILITIES`, which is an
    allowlist: anything not named there is unsafe. A denylist was tried first
    and failed exactly as denylists do, by omitting a capability nobody
    remembered to add.

    Set explicitly only to make a tool MORE restricted than its capability."""

    @property
    def is_offline_safe(self) -> bool:
        if self.offline_safe is not None:
            return self.offline_safe
        return self.capability in OFFLINE_SAFE_CAPABILITIES

    def json_schema(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        return schema

    def openai_schema(self) -> dict[str, Any]:
        """Tool definition in the shape both OpenAI-compatible runtimes expect."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description.strip(),
                "parameters": self.json_schema(),
            },
        }

    def available_to(self, specialist: SpecialistName | None) -> bool:
        if not self.specialists:
            return True
        if specialist is None:
            return True
        return specialist in self.specialists

    async def invoke(self, raw_args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        started = time.perf_counter()
        try:
            parsed = self.input_model.model_validate(raw_args or {})
        except ValidationError as exc:
            return ToolResult.failure(
                self.name,
                f"invalid arguments: {exc.errors(include_url=False)}",
                code="invalid_arguments",
            )

        hooks = ctx.hooks
        if hooks is not None:
            verdict = await hooks.before_tool(self, parsed, ctx)
            if verdict is not None and not verdict.allowed:
                return ToolResult.failure(
                    self.name, verdict.reason or "blocked by hook", code="hook_denied"
                )

        try:
            result = await self.handler(parsed, ctx)
        except ApprovalRequired as exc:
            result = ToolResult(
                ok=False,
                tool=self.name,
                summary=exc.message,
                error=exc.message,
                error_code=exc.code,
                proposed_commands=[exc.command],
            )
        except ToolError as exc:
            log.warning("tool_failed", tool=self.name, code=exc.code, error=exc.message)
            result = ToolResult.failure(self.name, exc.message, code=exc.code)
        except Exception as exc:
            log.exception("tool_crashed", tool=self.name)
            result = ToolResult.failure(self.name, f"{type(exc).__name__}: {exc}", code="crash")

        result.tool = result.tool or self.name
        result.duration_s = time.perf_counter() - started

        if hooks is not None:
            result = await hooks.after_tool(self, parsed, result, ctx)
        return result


class ToolRegistry:
    """Central registry. Modules register at import time via :func:`tool`."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec[Any]] = {}

    def register(self, spec: ToolSpec[Any]) -> ToolSpec[Any]:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = spec
        return spec

    def get(self, name: str) -> ToolSpec[Any] | None:
        return self._tools.get(name)

    def require(self, name: str) -> ToolSpec[Any]:
        spec = self._tools.get(name)
        if spec is None:
            raise ToolError(f"unknown tool: {name}", code="unknown_tool")
        return spec

    def names(self) -> list[str]:
        return sorted(self._tools)

    def all(self) -> list[ToolSpec[Any]]:
        return [self._tools[name] for name in sorted(self._tools)]

    def select(
        self,
        *,
        specialist: SpecialistName | None = None,
        capabilities: Sequence[Capability] | None = None,
        names: Sequence[str] | None = None,
        include_mutating: bool = True,
        max_risk: RiskClass | None = None,
    ) -> list[ToolSpec[Any]]:
        caps = set(capabilities) if capabilities else None
        wanted = set(names) if names else None
        out = []
        for spec in self.all():
            if wanted is not None and spec.name not in wanted:
                continue
            if caps is not None and spec.capability not in caps:
                continue
            if not spec.available_to(specialist):
                continue
            if not include_mutating and spec.mutating:
                continue
            if max_risk is not None and spec.risk.rank > max_risk.rank:
                continue
            out.append(spec)
        return out

    def openai_schemas(self, specs: Sequence[ToolSpec[Any]] | None = None) -> list[dict[str, Any]]:
        return [spec.openai_schema() for spec in (specs if specs is not None else self.all())]

    async def invoke(
        self, name: str, arguments: dict[str, Any], ctx: ToolContext
    ) -> ToolResult:
        spec = self.get(name)
        if spec is None:
            return ToolResult.failure(name, f"unknown tool: {name}", code="unknown_tool")
        if not spec.available_to(ctx.specialist):
            return ToolResult.failure(
                name,
                f"tool {name} is not available to {ctx.specialist}",
                code="not_permitted",
            )
        return await spec.invoke(arguments, ctx)

    def clear(self) -> None:
        self._tools.clear()


REGISTRY = ToolRegistry()


def tool(
    name: str,
    *,
    description: str,
    capability: Capability,
    risk: RiskClass = RiskClass.R1,
    specialists: Sequence[SpecialistName] = (),
    mutating: bool = False,
    requires_approval: bool = False,
    long_running: bool = False,
    tags: Sequence[str] = (),
    registry: ToolRegistry | None = None,
) -> Callable[[Handler], ToolSpec[Any]]:
    """Register an async handler as a typed tool.

    The handler signature must be ``async def h(args: SomeModel, ctx: ToolContext)``
    where ``SomeModel`` is a pydantic model; the schema is derived from it.
    """

    def decorator(handler: Handler) -> ToolSpec[Any]:
        hints = inspect.get_annotations(handler, eval_str=True)
        params = [p for p in inspect.signature(handler).parameters if p != "self"]
        if not params:
            raise TypeError(f"tool handler {handler.__name__} needs an input parameter")
        input_model = hints.get(params[0])
        if input_model is None or not (
            isinstance(input_model, type) and issubclass(input_model, BaseModel)
        ):
            raise TypeError(
                f"tool handler {handler.__name__} first parameter must be annotated "
                "with a pydantic BaseModel subclass"
            )
        if not inspect.iscoroutinefunction(handler):
            raise TypeError(f"tool handler {handler.__name__} must be async")

        spec: ToolSpec[Any] = ToolSpec(
            name=name,
            description=description,
            input_model=input_model,
            handler=handler,
            capability=capability,
            risk=risk,
            specialists=tuple(specialists),
            mutating=mutating,
            requires_approval=requires_approval,
            long_running=long_running,
            tags=tuple(tags),
        )
        (registry or REGISTRY).register(spec)
        return spec

    return decorator


def load_all_tools() -> ToolRegistry:
    """Import every helper module so decorators run. Idempotent."""
    from importlib import import_module

    for module in (
        "mimir.tools.repo",
        "mimir.tools.kubernetes",
        "mimir.tools.sdm",
        "mimir.tools.database",
        "mimir.tools.logs",
        "mimir.tools.web",
        "mimir.tools.sandbox",
        "mimir.tools.memory",
        "mimir.tools.search",
        "mimir.tools.reader",
        "mimir.tools.skills",
    ):
        try:
            import_module(module)
        except ImportError as exc:  # pragma: no cover - optional extras
            log.warning("tool_module_unavailable", module=module, error=str(exc))
    return REGISTRY
