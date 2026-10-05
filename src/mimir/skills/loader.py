"""SKILL.md parsing and validation (ADR 10.1, ADR 27 [S1])."""

from __future__ import annotations

import difflib
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from mimir.logging import get_logger
from mimir.models.command import RiskClass
from mimir.models.specialist import SpecialistName
from mimir.tools.base import REGISTRY, ToolRegistry

log = get_logger(__name__)

SKILL_FILENAME = "SKILL.md"
REFERENCES_DIRNAME = "references"
SCRIPTS_DIRNAME = "scripts"

# : Crude but stable context accounting.
CHARS_PER_TOKEN = 4

MAX_DESCRIPTION_CHARS = 200
MAX_WHEN_TO_USE_CHARS = 300

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")
_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?", re.DOTALL)

#: Helper names the ADR names in section 9 but which may not be registered yet
PLANNED_TOOL_NAMES: frozenset[str] = frozenset(
    {
        # ADR 9.1 repository helpers
        "list_repositories",
        "search_repository",
        "find_symbol",
        "read_file_range",
        "find_references",
        "inspect_git_history",
        "locate_tests",
        "build_import_graph",
        "build_flow_evidence",
        # ADR 9.2 kubernetes helpers
        "get_current_context",
        "list_namespaces",
        "list_workloads",
        "describe_resource",
        "get_logs",
        "get_events",
        "get_resource_usage",
        "get_rollout_status",
        "exec_readonly",
        "prepare_mutation",
        "execute_approved_mutation",
        # ADR 9.3 SDM and container helpers
        "get_sdm_status",
        "list_sdm_resources",
        "resolve_sdm_resource",
        "connect_sdm_resource",
        "list_remote_containers",
        "inspect_remote_container",
        "get_remote_container_logs",
        "run_remote_readonly",
        # ADR 9.4 database helpers
        "inspect_database_schema",
        "run_readonly_query",
        "explain_query",
        "prepare_database_mutation",
        "execute_approved_database_mutation",
        # ADR 9.5 log helpers
        "filter_logs",
        "group_repeated_errors",
        "extract_correlation_ids",
        "correlate_logs",
        "detect_timeout_patterns",
        "compare_before_after",
        "summarise_log_volume",
        # ADR 9.6 web helpers
        "web_search",
        "web_open",
        "web_find",
        "web_follow",
        "web_extract",
        "web_cite",
        # ADR 9.7 leaves the restricted code runner unnamed; this is the name
        "run_code",
    }
)


def estimate_tokens(text: str | int) -> int:
    """Rough token count from a string or a character count."""
    chars = text if isinstance(text, int) else len(text)
    return (chars + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


class SkillValidationError(ValueError):
    """Raised when a SKILL.md cannot be turned into a usable :class:`Skill`."""

    def __init__(self, path: Path, errors: list[str]) -> None:
        joined = "\n  - ".join(errors)
        super().__init__(f"invalid skill at {path}:\n  - {joined}")
        self.path = path
        self.errors = errors


class SkillIO(BaseModel):
    """One declared input or output (ADR 10.1 "Example inputs and outputs")."""

    name: str
    description: str = ""
    required: bool = False
    example: str = ""


class SkillResource(BaseModel):
    """A level-3 file: a reference document or an executable helper."""

    name: str
    """Path relative to ``references/`` or ``scripts/``. Used as the load key."""

    path: Path
    description: str = ""
    chars: int = 0
    declared: bool = True
    """False when the file exists on disk but the frontmatter does not list it."""

    @property
    def token_estimate(self) -> int:
        return estimate_tokens(self.chars)


class SkillExample(BaseModel):
    title: str = ""
    request: str = ""
    outcome: str = ""


class SkillTestCase(BaseModel):
    """A declared test case (ADR 10.1 "Test cases")."""

    name: str
    input: str = ""
    expected_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    """Free-text expectations about a model run. Skipped without a transcript."""

    assertions: list[str] = Field(default_factory=list)
    notes: str = ""


class Skill(BaseModel):
    """Validated skill metadata."""

    # -- frontmatter ------------------------------------------------------
    name: str
    version: str = "0.1.0"
    description: str
    when_to_use: str
    specialist: SpecialistName
    allowed_tools: list[str] = Field(default_factory=list)
    max_risk: RiskClass = RiskClass.R1
    inputs: list[SkillIO] = Field(default_factory=list)
    outputs: list[SkillIO] = Field(default_factory=list)
    safety_rules: list[str] = Field(default_factory=list)
    references: list[SkillResource] = Field(default_factory=list)
    scripts: list[SkillResource] = Field(default_factory=list)
    examples: list[SkillExample] = Field(default_factory=list)
    tests: list[SkillTestCase] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    author: str = ""
    updated_at: str = ""
    stub: bool = False
    """True when the body is a placeholder awaiting approved local documentation (ADR 5.4 and 9.3: environment specifics must not be invented)."""

    # -- derived ----------------------------------------------------------
    directory: Path
    source_path: Path
    body_chars: int = 0
    unavailable_tools: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("updated_at", mode="before")
    @classmethod
    def _stringify_date(cls, v: Any) -> Any:
        if isinstance(v, date | datetime):
            return v.isoformat()
        return v

    @field_validator("inputs", "outputs", mode="before")
    @classmethod
    def _normalise_io(cls, v: Any) -> Any:
        return _normalise_io_list(v)

    @field_validator("tags", "safety_rules", "allowed_tools", mode="before")
    @classmethod
    def _normalise_str_list(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, str):
            return [part.strip() for part in v.split(",") if part.strip()]
        return v

    # -- level accounting -------------------------------------------------

    def brief(self) -> dict[str, str]:
        """Level 1. The only thing the coordinator sees for every skill."""
        return {
            "name": self.name,
            "specialist": self.specialist.value,
            "description": self.description,
            "when_to_use": self.when_to_use,
        }

    def brief_line(self, max_chars: int = 240) -> str:
        line = (
            f"{self.name} [{self.specialist.value}] {self.description} "
            f"| when: {self.when_to_use}"
        )
        line = " ".join(line.split())
        if len(line) > max_chars:
            line = line[: max_chars - 1].rstrip() + "…"
        return line

    @property
    def level1_tokens(self) -> int:
        return estimate_tokens(self.brief_line())

    @property
    def level2_tokens(self) -> int:
        return estimate_tokens(self.body_chars)

    @property
    def level3_tokens(self) -> int:
        return sum(r.token_estimate for r in (*self.references, *self.scripts))

    def level_costs(self) -> dict[str, int]:
        """Estimated token cost of each disclosure level, for graph budgeting."""
        return {
            "level1": self.level1_tokens,
            "level2": self.level2_tokens,
            "level3": self.level3_tokens,
        }

    def reference(self, name: str) -> SkillResource | None:
        return _find_resource(self.references, name)

    def script(self, name: str) -> SkillResource | None:
        return _find_resource(self.scripts, name)

    @property
    def searchable_text(self) -> str:
        return " ".join([self.name.replace("-", " "), self.description, self.when_to_use,
                         " ".join(self.tags)])


class SkillLoadResult(BaseModel):
    """Non-raising load outcome, used by the ``validate_skill`` tool."""

    path: Path
    skill: Skill | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.skill is not None and not self.errors


def _find_resource(items: list[SkillResource], name: str) -> SkillResource | None:
    wanted = name.strip().lstrip("./")
    for item in items:
        if item.name == wanted or Path(item.name).name == wanted:
            return item
    return None


def _normalise_io_list(value: Any) -> Any:
    """Accept ``[str]``, ``{name: description}``, or a list of mappings."""
    if value is None:
        return []
    if isinstance(value, dict):
        return [{"name": k, "description": str(v)} for k, v in value.items()]
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, str):
                out.append({"name": item})
            elif isinstance(item, dict) and "name" not in item and len(item) == 1:
                key, val = next(iter(item.items()))
                out.append({"name": key, "description": str(val)})
            else:
                out.append(item)
        return out
    return value


def _normalise_keys(data: dict[str, Any]) -> dict[str, Any]:
    """``allowed-tools`` and ``allowed_tools`` mean the same thing."""
    return {str(k).strip().replace("-", "_").lower(): v for k, v in data.items()}


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Return ``(frontmatter, body)``. Raises ``ValueError`` on a malformed head."""
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        raise ValueError("missing YAML frontmatter; the file must start with a '---' line")
    raw = match.group(1)
    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"frontmatter is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("frontmatter must be a YAML mapping")
    return _normalise_keys(data), text[match.end() :]


def read_body(skill: Skill) -> str:
    """Level 2. Read the instruction body, without the frontmatter."""
    text = skill.source_path.read_text(encoding="utf-8")
    _, body = split_frontmatter(text)
    return body.strip()


def known_tool_names(tool_registry: ToolRegistry | None = None) -> set[str]:
    """Registered helpers plus the ADR 9 helpers that are not built yet."""
    registry = tool_registry or REGISTRY
    return set(registry.names()) | set(PLANNED_TOOL_NAMES)


def _collect_resources(
    directory: Path,
    subdir: str,
    declared: Any,
    errors: list[str],
    warnings: list[str],
) -> list[SkillResource]:
    """Merge frontmatter-declared resources with what is actually on disk."""
    base = directory / subdir
    declared_entries = _normalise_io_list(declared) or []
    resources: dict[str, SkillResource] = {}

    for entry in declared_entries:
        if not isinstance(entry, dict):
            errors.append(f"{subdir} entry must be a string or a mapping, got {entry!r}")
            continue
        name = str(entry.get("name", "")).strip().lstrip("./")
        if not name:
            errors.append(f"{subdir} entry is missing a name")
            continue
        if ".." in Path(name).parts or Path(name).is_absolute():
            errors.append(f"{subdir}/{name} escapes the skill directory")
            continue
        path = base / name
        if not path.is_file():
            errors.append(f"declared {subdir}/{name} does not exist at {path}")
            continue
        resources[name] = SkillResource(
            name=name,
            path=path,
            description=str(entry.get("description", "")),
            chars=path.stat().st_size,
            declared=True,
        )

    if base.is_dir():
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            name = path.relative_to(base).as_posix()
            if name.startswith(".") or name in resources:
                continue
            warnings.append(f"{subdir}/{name} exists but is not declared in frontmatter")
            resources[name] = SkillResource(
                name=name, path=path, chars=path.stat().st_size, declared=False
            )
    return list(resources.values())


def _validate_tools(
    names: list[str],
    known: set[str],
    registered: set[str],
    errors: list[str],
) -> list[str]:
    """Return the subset that is known but not yet registered."""
    unavailable = []
    for name in names:
        if name in registered:
            continue
        if name in known:
            unavailable.append(name)
            continue
        suggestion = difflib.get_close_matches(name, sorted(known), n=3, cutoff=0.6)
        hint = f" Did you mean: {', '.join(suggestion)}?" if suggestion else ""
        errors.append(
            f"allowed_tools lists '{name}', which is not a registered tool and is not "
            f"part of the ADR 9 helper vocabulary.{hint} A skill may only name helpers "
            "that exist; it cannot define new ones."
        )
    return unavailable


def load_skill_file(
    path: Path,
    *,
    tool_registry: ToolRegistry | None = None,
) -> SkillLoadResult:
    """Parse and validate one SKILL.md. Never raises for skill-authoring mistakes."""
    path = Path(path)
    if path.is_dir():
        path = path / SKILL_FILENAME
    errors: list[str] = []
    warnings: list[str] = []

    if not path.is_file():
        return SkillLoadResult(path=path, errors=[f"no {SKILL_FILENAME} at {path}"])

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return SkillLoadResult(path=path, errors=[f"cannot read {path}: {exc}"])

    try:
        front, body = split_frontmatter(text)
    except ValueError as exc:
        return SkillLoadResult(path=path, errors=[str(exc)])

    directory = path.parent
    name = str(front.get("name", "") or "").strip()
    if not name:
        errors.append("frontmatter must set 'name'")
    elif not _NAME_RE.match(name):
        errors.append(
            f"name '{name}' must be lowercase kebab-case matching {_NAME_RE.pattern}"
        )
    elif name != directory.name:
        errors.append(f"name '{name}' must match the directory name '{directory.name}'")

    description = str(front.get("description", "") or "").strip()
    when_to_use = str(front.get("when_to_use", "") or "").strip()
    if not description:
        errors.append("frontmatter must set 'description'")
    elif len(description) > MAX_DESCRIPTION_CHARS:
        warnings.append(
            f"description is {len(description)} chars; keep it under "
            f"{MAX_DESCRIPTION_CHARS} so the level-1 catalogue stays cheap"
        )
    if not when_to_use:
        errors.append("frontmatter must set 'when_to_use'")
    elif len(when_to_use) > MAX_WHEN_TO_USE_CHARS:
        warnings.append(
            f"when_to_use is {len(when_to_use)} chars; keep it under {MAX_WHEN_TO_USE_CHARS}"
        )

    raw_specialist = str(front.get("specialist", "") or "").strip()
    specialist: SpecialistName | None = None
    if not raw_specialist:
        errors.append(
            "frontmatter must set 'specialist' (ADR 10.5: a skill runs in a "
            f"specialist subgraph). One of: {', '.join(s.value for s in SpecialistName)}"
        )
    else:
        try:
            specialist = SpecialistName(raw_specialist)
        except ValueError:
            errors.append(
                f"specialist '{raw_specialist}' is not a SpecialistName. One of: "
                f"{', '.join(s.value for s in SpecialistName)}"
            )

    raw_risk = str(front.get("max_risk", RiskClass.R1.value) or RiskClass.R1.value).strip()
    try:
        max_risk = RiskClass(raw_risk.upper())
    except ValueError:
        max_risk = RiskClass.R1
        errors.append(
            f"max_risk '{raw_risk}' is not a RiskClass. One of: "
            f"{', '.join(r.value for r in RiskClass)}"
        )

    raw_tools = front.get("allowed_tools") or []
    if isinstance(raw_tools, str):
        raw_tools = [p.strip() for p in raw_tools.split(",") if p.strip()]
    tool_names = [str(t).strip() for t in raw_tools if str(t).strip()]
    if not tool_names and max_risk != RiskClass.R0:
        # An R0 skill is analysis only by definition (command-explainer,
        warnings.append("allowed_tools is empty; this skill can only reason, not act")
    registry = tool_registry or REGISTRY
    registered = set(registry.names())
    unavailable = _validate_tools(tool_names, known_tool_names(registry), registered, errors)

    references = _collect_resources(directory, REFERENCES_DIRNAME, front.get("references"),
                                   errors, warnings)
    scripts = _collect_resources(directory, SCRIPTS_DIRNAME, front.get("scripts"),
                                 errors, warnings)

    tests_raw = front.get("tests") or front.get("test_cases") or []
    tests: list[SkillTestCase] = []
    if not isinstance(tests_raw, list):
        errors.append("tests must be a list of mappings")
    else:
        for index, item in enumerate(tests_raw):
            if not isinstance(item, dict):
                errors.append(f"tests[{index}] must be a mapping")
                continue
            payload = _normalise_keys(item)
            payload.setdefault("name", f"case-{index + 1}")
            try:
                tests.append(SkillTestCase.model_validate(payload))
            except Exception as exc:  # noqa: BLE001 - surfaced as an author-facing error
                errors.append(f"tests[{index}] is invalid: {exc}")

    examples_raw = front.get("examples") or []
    examples: list[SkillExample] = []
    if isinstance(examples_raw, list):
        for index, item in enumerate(examples_raw):
            payload = _normalise_keys(item) if isinstance(item, dict) else {"title": str(item)}
            try:
                examples.append(SkillExample.model_validate(payload))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"examples[{index}] is invalid: {exc}")
    else:
        errors.append("examples must be a list")

    if errors or specialist is None:
        return SkillLoadResult(path=path, errors=errors, warnings=warnings)

    try:
        skill = Skill(
            name=name,
            version=str(front.get("version", "0.1.0")),
            description=description,
            when_to_use=when_to_use,
            specialist=specialist,
            allowed_tools=tool_names,
            max_risk=max_risk,
            inputs=front.get("inputs") or [],
            outputs=front.get("outputs") or [],
            safety_rules=front.get("safety_rules") or [],
            references=references,
            scripts=scripts,
            examples=examples,
            tests=tests,
            tags=front.get("tags") or [],
            author=str(front.get("author", "") or ""),
            updated_at=front.get("updated_at") or "",
            stub=bool(front.get("stub", False)),
            directory=directory,
            source_path=path,
            body_chars=len(body.strip()),
            unavailable_tools=unavailable,
            warnings=warnings,
        )
    except Exception as exc:  # noqa: BLE001 - pydantic message is the author's answer
        return SkillLoadResult(path=path, errors=[str(exc)], warnings=warnings)

    if not body.strip():
        return SkillLoadResult(
            path=path,
            errors=["SKILL.md has frontmatter but no instruction body"],
            warnings=warnings,
        )

    return SkillLoadResult(path=path, skill=skill, warnings=warnings)


def parse_skill_file(path: Path, *, tool_registry: ToolRegistry | None = None) -> Skill:
    """Strict variant of :func:`load_skill_file`."""
    result = load_skill_file(path, tool_registry=tool_registry)
    if result.skill is None:
        raise SkillValidationError(result.path, result.errors or ["unknown error"])
    return result.skill
