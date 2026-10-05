"""``mimir doctor``: check that everything MIMIR depends on is reachable."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Literal

from rich.console import Console
from rich.table import Table
from rich.text import Text

from mimir.api.auth import active_keys
from mimir.config import get_settings

Status = Literal["ok", "warn", "fail", "skip"]

_STYLES: dict[Status, str] = {
    "ok": "green",
    "warn": "yellow",
    "fail": "red",
    "skip": "dim",
}


@dataclass(slots=True)
class Check:
    name: str
    status: Status
    detail: str
    hint: str = ""

    @property
    def blocking(self) -> bool:
        return self.status == "fail"


async def run_doctor(console: Console) -> bool:
    settings = get_settings()
    checks: list[Check] = []

    checks.append(
        Check(
            "config",
            "ok",
            f"{settings.home}",
            "" if (settings.home / "config.yaml").is_file() else "run 'mimir init'",
        )
    )

    checks.extend(_binary_checks(settings))
    checks.append(_repo_check(settings))
    checks.extend(await _model_checks())
    checks.append(_tools_check())
    checks.append(_context_check(settings))
    checks.append(_knowledge_check(settings))
    checks.append(_skills_check())
    checks.append(_persistence_check())
    checks.append(_safety_check(settings))
    checks.append(_exposure_check(settings))

    table = Table(box=None, header_style="dim")
    table.add_column("check", width=22)
    table.add_column("", width=6)
    table.add_column("detail", overflow="fold")
    for check in checks:
        table.add_row(
            check.name,
            Text(check.status.upper(), style=_STYLES[check.status]),
            check.detail + (f"\n[dim]{check.hint}[/dim]" if check.hint else ""),
        )
    console.print(table)

    failures = [c for c in checks if c.blocking]
    warnings = [c for c in checks if c.status == "warn"]
    console.print()
    if failures:
        console.print(
            Text(f"{len(failures)} blocking problem(s). MIMIR will not work until fixed.",
                 style="bold red")
        )
    elif warnings:
        console.print(
            Text(f"usable, with {len(warnings)} degraded capability/ies.", style="yellow")
        )
    else:
        console.print(Text("everything checks out.", style="green"))
    return not failures


def _binary_checks(settings) -> list[Check]:
    out: list[Check] = []
    required = [("git", "repository history"), ("rg", "fast repository search")]
    optional = [
        (settings.kubernetes.kubectl_path, "Kubernetes investigation"),
        (settings.sdm.sdm_path, "SDM-mediated access"),
        (settings.database_tool.psql_path, "database inspection"),
        (settings.sdm.container_cli, "container inspection"),
    ]
    for binary, purpose in required:
        path = shutil.which(binary)
        out.append(
            Check(
                binary,
                "ok" if path else "warn",
                path or f"not on PATH; {purpose} degrades to a slower fallback",
                "" if path else f"brew install {binary}",
            )
        )
    for binary, purpose in optional:
        path = shutil.which(binary)
        out.append(
            Check(
                binary,
                "ok" if path else "skip",
                path or f"not installed; {purpose} unavailable",
            )
        )
    return out


def _repo_check(settings) -> Check:
    try:
        from mimir.tools.repo import get_repository_directory

        repos = get_repository_directory(settings).all()
        if not repos:
            return Check(
                "repositories",
                "warn",
                "no repositories configured or discovered",
                "add paths under 'repos.roots' or 'repos.entries' in the config",
            )
        names = ", ".join(r.name for r in repos[:8])
        more = f" (+{len(repos) - 8} more)" if len(repos) > 8 else ""
        return Check("repositories", "ok", f"{len(repos)} found: {names}{more}")
    except Exception as exc:  # noqa: BLE001
        return Check("repositories", "warn", f"discovery failed: {exc}")


async def _model_checks() -> list[Check]:
    from mimir.llm.router import get_router

    router = get_router()
    out: list[Check] = []
    try:
        report = await router.health_report()
    except Exception as exc:  # noqa: BLE001
        return [Check("models", "fail", f"router failed: {exc}")]

    for alias, (ok, detail) in report.items():
        profile = router.profile_for(alias)
        out.append(
            Check(
                f"model:{alias}",
                "ok" if ok else "fail",
                f"{profile.runtime} {profile.model} at {profile.base_url}\n{detail}",
                ""
                if ok
                else "start the runtime (for example 'ollama serve') and pull the model, "
                "or point this profile elsewhere with 'mimir models set'",
            )
        )
    await router.close()
    if not out:
        out.append(Check("models", "fail", "no model profiles configured", "run 'mimir init'"))
    return out


def _tools_check() -> Check:
    from mimir.tools.base import load_all_tools

    registry = load_all_tools()
    specs = registry.all()
    by_capability: dict[str, int] = {}
    for spec in specs:
        by_capability[spec.capability.value] = by_capability.get(spec.capability.value, 0) + 1
    detail = ", ".join(f"{k}:{v}" for k, v in sorted(by_capability.items()))
    return Check("tools", "ok" if specs else "fail", f"{len(specs)} registered ({detail})")


def _context_check(settings) -> Check:
    """Is the runtime serving the context window we configured?"""
    from mimir.eval.provenance import resolve_model

    problems = []
    for alias in sorted(settings.models.profiles):
        if settings.models.profiles[alias].runtime != "ollama":
            continue
        identity = resolve_model(alias, settings)
        if identity.context_mismatch:
            problems.append(
                f"{alias} ({identity.name}) configured {identity.context_window} "
                f"but served {identity.served_context}"
            )
    if problems:
        return Check(
            "context window",
            "warn",
            "; ".join(problems) + ". Set OLLAMA_CONTEXT_LENGTH on the server: "
            "the OpenAI shim ignores num_ctx.",
        )
    return Check("context window", "ok", "runtime serves the configured window")


def _knowledge_check(settings) -> Check:
    try:
        from mimir.knowledge.index import get_knowledge_index
        from mimir.knowledge.store import get_knowledge_store

        documents = get_knowledge_store(settings).documents()
        counts = get_knowledge_index(settings).counts()
        if not documents:
            return Check(
                "knowledge",
                "warn",
                f"no documents under {settings.knowledge.root}",
                "MIMIR works without memory, but it is much less useful. "
                "Add runbooks, or run 'mimir memory import' on prior work.",
            )
        return Check(
            "knowledge",
            "ok",
            f"{len(documents)} document(s), {counts.get('chunks', 0)} indexed chunk(s)",
        )
    except Exception as exc:  # noqa: BLE001
        return Check("knowledge", "warn", f"unavailable: {exc}")


def _skills_check() -> Check:
    try:
        from mimir.skills.registry import get_skill_registry

        registry = get_skill_registry().reload()
        skills = registry.all()
        if not skills:
            return Check("skills", "warn", "no skills found", "seed them with 'mimir init'")
        return Check(
            "skills",
            "ok",
            f"{len(skills)} skill(s), catalogue about {registry.catalogue_tokens()} tokens",
        )
    except Exception as exc:  # noqa: BLE001
        return Check("skills", "warn", f"unavailable: {exc}")


def _persistence_check() -> Check:
    try:
        from mimir.persistence.db import get_database

        database = get_database()
        database.ensure_schema()
        health = database.health()
        detail = ", ".join(f"{k}={v}" for k, v in health.items() if k != "error")
        status = "warn" if health.get("error") else "ok"
        return Check("persistence", status, detail or "reachable", str(health.get("error") or ""))
    except Exception as exc:  # noqa: BLE001
        return Check(
            "persistence",
            "warn",
            f"unavailable: {exc}",
            "sessions will not be resumable",
        )


def _safety_check(settings) -> Check:
    ceiling = settings.safety.auto_execute_max_risk
    status: Status = "ok"
    hint = ""
    if ceiling in ("R3", "R4"):
        status = "fail"
        hint = (
            "auto_execute_max_risk is set to a mutating class. MIMIR would change "
            "live state without asking. Set it back to R1."
        )
    elif ceiling == "R2":
        status = "warn"
        hint = "R2 lets elevated inspection run unattended (exec, port-forward, db sessions)"
    return Check(
        "safety",
        status,
        f"auto-execute ceiling {ceiling}; "
        f"mutations require approval: {settings.safety.require_approval_for_mutations}",
        hint,
    )


def _exposure_check(settings) -> Check:
    api = settings.api
    if api.expose_privileged_routes_publicly:
        return Check(
            "exposure",
            "fail",
            "privileged routes are marked publicly exposed",
            "ADR 16.5 requires shell, Kubernetes, SDM, and database helpers to stay "
            "loopback-only. Set expose_privileged_routes_publicly back to false.",
        )
    if api.host not in ("127.0.0.1", "localhost", "::1") and not active_keys(settings):
        return Check(
            "exposure",
            "fail",
            f"API binds {api.host} with no API keys configured",
            "run 'mimir keys create' before binding a non-loopback address",
        )
    detail = f"binds {api.host}:{api.port}"
    if active_keys(settings):
        detail += f", {active_keys(settings)} API key(s) configured"
    if api.facade_agent_mode:
        detail += "; facade runs the full agent graph (ADR 16.4 advises against this)"
        return Check("exposure", "warn", detail)
    return Check("exposure", "ok", detail)
