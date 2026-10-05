"""Skill test-case runner (ADR 10.1 "Test cases")."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from mimir.logging import get_logger
from mimir.models.command import RiskClass
from mimir.models.specialist import SpecialistName
from mimir.skills.loader import Skill, SkillTestCase, read_body
from mimir.skills.runner import SkillRunner

log = get_logger(__name__)

_CODE_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_-]*\n(.*?)```", re.DOTALL)
_PREDICATE_RE = re.compile(r"^\s*([a-z_]+)\s*:\s*(.+)$", re.DOTALL)

_STATIC_PREDICATES = frozenset(
    {
        "contains",
        "not_contains",
        "matches",
        "not_matches",
        "command_contains",
        "reference_exists",
        "script_exists",
        "tool_allowed",
        "tool_not_allowed",
        "max_risk_at_most",
        "specialist_is",
    }
)


class TestOutcome(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass(slots=True)
class SkillTranscript:
    """What a real run of a skill did. Optional input to the test runner."""

    tools_called: list[str] = field(default_factory=list)
    output: str = ""


@dataclass(slots=True)
class CheckResult:
    name: str
    outcome: TestOutcome
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.outcome == TestOutcome.PASSED


@dataclass(slots=True)
class SkillTestResult:
    skill: str
    case: str
    outcome: TestOutcome
    checks: list[CheckResult] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"{self.outcome.value.upper():<8} {self.skill} :: {self.case}"]
        lines.extend(
            f"    {c.outcome.value:<8} {c.name}" + (f" - {c.detail}" if c.detail else "")
            for c in self.checks
        )
        return "\n".join(lines)


@dataclass(slots=True)
class SkillTestReport:
    results: list[SkillTestResult] = field(default_factory=list)
    package_issues: list[str] = field(default_factory=list)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.outcome in (TestOutcome.FAILED, TestOutcome.ERROR))

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.outcome == TestOutcome.PASSED)

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.outcome == TestOutcome.SKIPPED)

    @property
    def ok(self) -> bool:
        return self.failed == 0 and not self.package_issues

    def render(self) -> str:
        lines = [r.render() for r in self.results]
        if self.package_issues:
            lines.append("package issues:")
            lines.extend(f"    ! {issue}" for issue in self.package_issues)
        lines.append(
            f"{self.passed} passed, {self.failed} failed, {self.skipped} skipped"
        )
        return "\n".join(lines)


def validate_skill_package(skill: Skill, runner: SkillRunner | None = None) -> list[str]:
    """Structural problems that are not fatal to loading but should be fixed."""
    issues: list[str] = list(skill.warnings)

    for resource in (*skill.references, *skill.scripts):
        if not resource.path.is_file():
            issues.append(f"declared resource {resource.name} is missing at {resource.path}")
        elif not resource.declared:
            issues.append(f"{resource.name} is on disk but undeclared in frontmatter")

    if skill.unavailable_tools:
        issues.append(
            "allowed_tools names helpers that are not registered in this build: "
            + ", ".join(sorted(skill.unavailable_tools))
        )

    if not skill.stub and not skill.tests:
        issues.append("skill declares no test cases")

    if runner is not None:
        permissions = runner.permitted_tools(skill)
        if skill.allowed_tools and not permissions.allowed and not skill.unavailable_tools:
            issues.append(
                "no declared tool survives the specialist intersection: "
                + "; ".join(f"{n} ({r})" for n, r in permissions.denied)
            )
    return issues


def _code_lines(body: str) -> list[str]:
    lines: list[str] = []
    for block in _CODE_BLOCK_RE.findall(body):
        lines.extend(line.strip().lstrip("$ ").strip() for line in block.splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def _check_static(
    assertion: str, skill: Skill, body: str
) -> CheckResult:
    match = _PREDICATE_RE.match(assertion)
    if match and match.group(1) in _STATIC_PREDICATES:
        predicate, argument = match.group(1), match.group(2).strip()
    else:
        predicate, argument = "contains", assertion.strip()

    name = f"{predicate}: {argument}"
    haystack = body.lower()

    try:
        if predicate == "contains":
            ok = argument.lower() in haystack
            detail = "" if ok else "not found in the instruction body"
        elif predicate == "not_contains":
            ok = argument.lower() not in haystack
            detail = "" if ok else "unexpectedly present in the instruction body"
        elif predicate in ("matches", "not_matches"):
            found = re.search(argument, body, re.IGNORECASE | re.MULTILINE) is not None
            ok = found if predicate == "matches" else not found
            verb = "did not match" if predicate == "matches" else "matched"
            detail = "" if ok else f"regex {verb}"
        elif predicate == "command_contains":
            binary, _, fragment = argument.partition("::")
            binary, fragment = binary.strip(), fragment.strip()
            if not fragment:
                return CheckResult(name, TestOutcome.ERROR, "expected 'binary::fragment'")
            offenders = [
                line
                for line in _code_lines(body)
                if line.split(" ", 1)[0] == binary and fragment not in line
            ]
            ok = not offenders
            detail = "" if ok else f"{len(offenders)} invocation(s) missing it: {offenders[0]}"
        elif predicate == "reference_exists":
            resource = skill.reference(argument)
            ok = resource is not None and resource.path.is_file()
            detail = "" if ok else "no such reference"
        elif predicate == "script_exists":
            resource = skill.script(argument)
            ok = resource is not None and resource.path.is_file()
            detail = "" if ok else "no such script"
        elif predicate == "tool_allowed":
            ok = argument in skill.allowed_tools
            detail = "" if ok else f"allowed_tools = {skill.allowed_tools}"
        elif predicate == "tool_not_allowed":
            ok = argument not in skill.allowed_tools
            detail = "" if ok else "tool is in allowed_tools"
        elif predicate == "max_risk_at_most":
            ceiling = RiskClass(argument.upper())
            ok = skill.max_risk.rank <= ceiling.rank
            detail = "" if ok else f"skill max_risk is {skill.max_risk.value}"
        elif predicate == "specialist_is":
            ok = skill.specialist == SpecialistName(argument)
            detail = "" if ok else f"skill specialist is {skill.specialist.value}"
        else:  # pragma: no cover - guarded by _STATIC_PREDICATES
            return CheckResult(name, TestOutcome.ERROR, "unknown predicate")
    except (re.error, ValueError) as exc:
        return CheckResult(name, TestOutcome.ERROR, str(exc))

    return CheckResult(name, TestOutcome.PASSED if ok else TestOutcome.FAILED, detail)


def _check_case(
    case: SkillTestCase,
    skill: Skill,
    body: str,
    transcript: SkillTranscript | None,
) -> SkillTestResult:
    checks: list[CheckResult] = []

    # Structural: a case that expects a tool outside the allowlist can never pass,
    for tool_name in case.expected_tools:
        if tool_name not in skill.allowed_tools:
            checks.append(
                CheckResult(
                    f"expected_tools declares {tool_name}",
                    TestOutcome.FAILED,
                    "not in the skill's allowed_tools, so this case is unsatisfiable",
                )
            )
    for tool_name in case.forbidden_tools:
        if tool_name in skill.allowed_tools:
            checks.append(
                CheckResult(
                    f"forbidden_tools declares {tool_name}",
                    TestOutcome.FAILED,
                    "the skill allowlist grants it, so it can be called",
                )
            )

    checks.extend(_check_static(assertion, skill, body) for assertion in case.assertions)

    if transcript is None:
        if case.expected_tools:
            checks.append(
                CheckResult("expected_tools were called", TestOutcome.SKIPPED,
                            "needs a recorded run")
            )
        if case.expected_outputs:
            checks.append(
                CheckResult("expected_outputs were produced", TestOutcome.SKIPPED,
                            "needs a recorded run")
            )
    else:
        called = set(transcript.tools_called)
        for tool_name in case.expected_tools:
            ok = tool_name in called
            checks.append(
                CheckResult(f"called {tool_name}",
                            TestOutcome.PASSED if ok else TestOutcome.FAILED,
                            "" if ok else f"tools called: {sorted(called) or 'none'}")
            )
        for tool_name in case.forbidden_tools:
            ok = tool_name not in called
            checks.append(
                CheckResult(f"did not call {tool_name}",
                            TestOutcome.PASSED if ok else TestOutcome.FAILED)
            )
        for expected in case.expected_outputs:
            ok = expected.lower() in transcript.output.lower()
            checks.append(
                CheckResult(f"output contains: {expected}",
                            TestOutcome.PASSED if ok else TestOutcome.FAILED)
            )

    if any(c.outcome == TestOutcome.ERROR for c in checks):
        outcome = TestOutcome.ERROR
    elif any(c.outcome == TestOutcome.FAILED for c in checks):
        outcome = TestOutcome.FAILED
    elif not checks or all(c.outcome == TestOutcome.SKIPPED for c in checks):
        outcome = TestOutcome.SKIPPED
    else:
        outcome = TestOutcome.PASSED

    return SkillTestResult(skill=skill.name, case=case.name, outcome=outcome, checks=checks)


def run_skill_tests(
    skill: Skill,
    *,
    runner: SkillRunner | None = None,
    transcripts: dict[str, SkillTranscript] | None = None,
) -> SkillTestReport:
    """Run every declared case for one skill."""
    body = read_body(skill)
    report = SkillTestReport(package_issues=validate_skill_package(skill, runner))
    for case in skill.tests:
        report.results.append(
            _check_case(case, skill, body, (transcripts or {}).get(case.name))
        )
    return report


def run_all_skill_tests(
    skills: list[Skill], *, runner: SkillRunner | None = None
) -> SkillTestReport:
    """Run the declared cases across a whole library."""
    combined = SkillTestReport()
    for skill in skills:
        single = run_skill_tests(skill, runner=runner)
        combined.results.extend(single.results)
        combined.package_issues.extend(f"{skill.name}: {i}" for i in single.package_issues)
    return combined
