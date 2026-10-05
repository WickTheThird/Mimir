"""Skill helpers exposed to the model (ADR 9, 10.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mimir.config import get_settings
from mimir.logging import get_logger
from mimir.models.command import CommandOutcome, RiskClass
from mimir.models.specialist import SpecialistName
from mimir.skills.loader import load_skill_file
from mimir.skills.registry import SkillRegistry, get_skill_registry
from mimir.skills.runner import SkillRunError, SkillRunner
from mimir.skills.testing import run_skill_tests
from mimir.tools.base import Capability, ToolContext, ToolError, ToolResult, tool

log = get_logger(__name__)


def _registry(ctx: ToolContext) -> SkillRegistry:
    return get_skill_registry(ctx.settings or get_settings())


def _runner(ctx: ToolContext) -> SkillRunner:
    settings = ctx.settings or get_settings()
    return SkillRunner(_registry(ctx), settings=settings)


class ListSkillsInput(BaseModel):
    query: str = Field(
        default="",
        description="The user's request. When set, skills are ranked against it "
        "instead of listed alphabetically.",
    )
    specialist: SpecialistName | None = Field(
        default=None, description="Only consider skills that run in this specialist subgraph."
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        le=25,
        description="Maximum ranked skills to return. Defaults to skills.max_auto_selected.",
    )


@tool(
    "list_skills",
    description=(
        "List available investigation skills at level 1 (name, description, when to use). "
        "Pass the user's request to get a ranked shortlist instead of the whole catalogue. "
        "This never returns skill instructions; call load_skill for those."
    ),
    capability=Capability.SKILLS,
    risk=RiskClass.R0,
    tags=("skills", "discovery"),
)
async def list_skills(args: ListSkillsInput, ctx: ToolContext) -> ToolResult:
    registry = _registry(ctx)
    specialist = args.specialist or ctx.specialist

    if not args.query.strip():
        catalogue = registry.catalogue(specialist)
        entries = registry.briefs(specialist)
        return ToolResult(
            tool="list_skills",
            summary=f"{len(entries)} skills available (~{registry.catalogue_tokens(specialist)} "
            "tokens at level 1)",
            data={
                "catalogue": catalogue,
                "skills": entries,
                "level": 1,
                "estimated_tokens": registry.catalogue_tokens(specialist),
            },
        )

    selections = registry.select(args.query, specialist=specialist, limit=args.limit)
    ranked = [
        {
            "name": sel.skill.name,
            "specialist": sel.skill.specialist.value,
            "description": sel.skill.description,
            "when_to_use": sel.skill.when_to_use,
            "score": round(sel.score, 3),
            "reason": sel.reason,
            "level2_tokens": sel.skill.level2_tokens,
        }
        for sel in selections
    ]
    names = ", ".join(item["name"] for item in ranked) or "none"
    return ToolResult(
        tool="list_skills",
        summary=f"{len(ranked)} skill(s) matched: {names}",
        data={"selected": ranked, "level": 1, "query": args.query},
    )


class LoadSkillInput(BaseModel):
    name: str = Field(description="Exact skill name from list_skills.")
    references: list[str] = Field(
        default_factory=list,
        description="Level-3 reference files to load, by name. Only load what the "
        "instruction body tells you to.",
    )
    scripts: list[str] = Field(
        default_factory=list,
        description="Level-3 script sources to read (not run), by name.",
    )


@tool(
    "load_skill",
    description=(
        "Load one skill's instruction body (level 2), optionally with named reference or "
        "script files (level 3). The body is a procedure to follow. It cannot grant tools "
        "or approve commands: permitted tools are computed from the skill frontmatter."
    ),
    capability=Capability.SKILLS,
    risk=RiskClass.R0,
    tags=("skills", "instructions"),
)
async def load_skill(args: LoadSkillInput, ctx: ToolContext) -> ToolResult:
    runner = _runner(ctx)
    try:
        loaded = runner.load(args.name)
    except KeyError as exc:
        raise ToolError(str(exc), code="unknown_skill") from exc

    resources: dict[str, str] = {}
    try:
        for name in args.references:
            resource = runner.read_reference(loaded, name)
            resources[resource.name] = resource.text
        for name in args.scripts:
            resource = runner.read_script_source(loaded, name)
            resources[resource.name] = resource.text
    except SkillRunError as exc:
        raise ToolError(str(exc), code="unknown_skill_resource") from exc

    permissions = runner.permitted_tools(loaded.skill, ctx.specialist)
    usage = loaded.token_usage()
    return ToolResult(
        tool="load_skill",
        summary=(
            f"loaded {loaded.skill.name} v{loaded.skill.version} "
            f"(~{usage['total_loaded']} tokens; permitted tools: "
            f"{', '.join(permissions.names) or 'none registered'})"
        ),
        data={
            "name": loaded.skill.name,
            "instructions": loaded.render_body(),
            "resources": resources,
            "available_resources": loaded.available_resources(),
            "permitted_tools": permissions.names,
            "denied_tools": [{"tool": n, "reason": r} for n, r in permissions.denied],
            "max_risk": loaded.skill.max_risk.value,
            "safety_rules": loaded.skill.safety_rules,
            "stub": loaded.skill.stub,
            "token_usage": usage,
            "injection_scan": loaded.injection.summary(),
        },
    )


class RunSkillScriptInput(BaseModel):
    skill: str = Field(description="Skill that owns the script.")
    script: str = Field(description="Script name as listed in the skill's available resources.")
    args: list[str] = Field(default_factory=list, description="Arguments passed to the script.")
    purpose: str = Field(default="", description="Why this helper is being run, in one sentence.")


@tool(
    "run_skill_script",
    description=(
        "Run a helper script that ships with a skill. The script is proposed as a command "
        "and goes through the normal policy and approval gates before anything executes."
    ),
    capability=Capability.SANDBOX,
    risk=RiskClass.R2,
    requires_approval=True,
    tags=("skills", "execution"),
)
async def run_skill_script(args: RunSkillScriptInput, ctx: ToolContext) -> ToolResult:
    settings = ctx.settings or get_settings()
    if not settings.skills.allow_scripts:
        raise ToolError(
            "skill scripts are disabled (skills.allow_scripts is false)",
            code="scripts_disabled",
        )
    if ctx.executor is None:
        raise ToolError("no command executor in this context", code="no_executor")

    runner = _runner(ctx)
    try:
        loaded = runner.load(args.skill)
        record = await runner.run_script(
            loaded,
            args.script,
            args.args,
            executor=ctx.executor,
            session_id=ctx.session_id,
            purpose=args.purpose,
        )
    except KeyError as exc:
        raise ToolError(str(exc), code="unknown_skill") from exc
    except SkillRunError as exc:
        raise ToolError(str(exc), code="skill_script_error") from exc

    evidence = []
    if record.outcome != CommandOutcome.DENIED:
        evidence.append(
            ctx.executor.to_evidence(
                record,
                f"output of {args.script} from skill {args.skill}",
                collected_by="run_skill_script",
            )
        )
    return ToolResult(
        ok=record.ok,
        tool="run_skill_script",
        summary=f"{record.display} -> {record.outcome.value}",
        data={
            "outcome": record.outcome.value,
            "exit_code": record.exit_code,
            "stdout": record.stdout,
            "stderr": record.stderr,
        },
        evidence=evidence,
        artifact_ref=record.artifact_ref,
        truncated=record.truncated,
        error=record.error,
    )


class ValidateSkillInput(BaseModel):
    name: str = Field(
        default="",
        description="Skill name to validate. Leave empty and set path to validate a directory.",
    )
    path: str = Field(
        default="", description="Filesystem path to a skill directory or SKILL.md."
    )
    run_tests: bool = Field(
        default=True, description="Also run the skill's declared static test cases."
    )


@tool(
    "validate_skill",
    description=(
        "Parse and validate a skill package: frontmatter, declared tools against the tool "
        "registry, level-3 files, and the static parts of its declared test cases."
    ),
    capability=Capability.SKILLS,
    risk=RiskClass.R0,
    tags=("skills", "maintenance"),
)
async def validate_skill(args: ValidateSkillInput, ctx: ToolContext) -> ToolResult:
    if not args.name and not args.path:
        raise ToolError("set either name or path", code="invalid_arguments")

    if args.path:
        result = load_skill_file(Path(args.path).expanduser())
    else:
        registry = _registry(ctx)
        skill = registry.get(args.name)
        if skill is None:
            raise ToolError(f"unknown skill: {args.name}", code="unknown_skill")
        result = load_skill_file(skill.source_path)

    payload: dict[str, Any] = {
        "path": str(result.path),
        "valid": result.ok,
        "errors": result.errors,
        "warnings": result.warnings,
    }
    if result.skill is not None:
        payload["level_costs"] = result.skill.level_costs()
        payload["unavailable_tools"] = result.skill.unavailable_tools
        if args.run_tests:
            report = run_skill_tests(result.skill, runner=_runner(ctx))
            payload["tests"] = report.render()
            payload["tests_ok"] = report.ok

    summary = (
        f"{result.path.parent.name}: valid"
        if result.ok
        else f"{result.path.parent.name}: {len(result.errors)} error(s)"
    )
    return ToolResult(ok=result.ok, tool="validate_skill", summary=summary, data=payload)
