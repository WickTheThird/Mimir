"""Every change is checked before it is allowed to stand.

MIMIR wrote files and hoped. The model was asked to know the language, the
framework and the codebase, and when it got one of them wrong the edit stayed,
looked plausible in a diff, and was found later by whoever ran the code.

None of those three need a model to check. A file either parses or it does not.
The language server either reports a new error or it does not. A project rule
either matches or it does not. So the model proposes and the rules dispose,
which is the same argument as the risk classifier: the part that must be
correct does not get to depend on the part that is probabilistic.

Two kinds of outcome, and the distinction is deliberate. A file that no longer
parses is reverted, because there is no reading under which that is an
improvement and leaving it breaks every later tool call on the same file.
Everything else is reported and left in place: an edit that introduces a type
error may be the first half of a change the next step completes, and reverting
it would make the loop unable to work in two steps.

It can only ever block on evidence. No language server means no diagnostics
check, not a failed one.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mimir.logging import get_logger
from mimir.verify.rules import Violation, check_rules, load_rules, rule_roots

log = get_logger(__name__)

#: Extensions whose syntax can be checked in-process, with no server.
_PARSERS: dict[str, str] = {".py": "python", ".json": "json"}

_LSP_EXTENSIONS = frozenset({
    ".py", ".go", ".rs", ".c", ".h", ".cc", ".cpp", ".ts", ".tsx", ".js", ".jsx",
})


@dataclass
class ChangeReport:
    path: str
    reverted: bool = False
    violations: list[Violation] = field(default_factory=list)
    new_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    new_lint: list[dict[str, Any]] = field(default_factory=list)
    broke_formatting: bool = False
    checks_run: list[str] = field(default_factory=list)
    checks_skipped: list[str] = field(default_factory=list)

    @property
    def blocking(self) -> list[Violation]:
        return [v for v in self.violations if v.blocking]

    @property
    def ok(self) -> bool:
        return not (
            self.blocking or self.new_diagnostics or self.new_lint
            or self.reverted or self.broke_formatting
        )

    def summary(self) -> str:
        if self.reverted:
            return f"reverted: {self.violations[0].title}" if self.violations else "reverted"
        bits: list[str] = []
        if self.violations:
            errors = len(self.blocking)
            bits.append(
                f"{len(self.violations)} rule violation(s)"
                + (f", {errors} blocking" if errors else "")
            )
        if self.new_diagnostics:
            bits.append(f"{len(self.new_diagnostics)} new error(s) from the language server")
        if self.new_lint:
            codes = ", ".join(sorted({str(f["code"]) for f in self.new_lint})[:4])
            bits.append(f"{len(self.new_lint)} new lint finding(s): {codes}")
        if self.broke_formatting:
            bits.append("the file no longer matches the project formatter")
        if not bits:
            checked = ", ".join(self.checks_run) or "nothing to check"
            return f"verified ({checked})"
        return "; ".join(bits)

    def detail(self) -> list[str]:
        lines = [v.render() for v in self.violations[:8]]
        lines += [
            f"{self.path}:{d.get('line', 1)} {str(d.get('message', ''))[:120]}"
            for d in self.new_diagnostics[:8]
        ]
        lines += [
            f"{self.path}:{f.get('line', 1)} {f.get('code', '')} "
            f"{str(f.get('message', ''))[:110]}"
            for f in self.new_lint[:8]
        ]
        return lines


def check_syntax(path: str, text: str) -> Violation | None:
    """Does this file parse at all.

    In-process and instant for the languages that allow it. This is the check
    that earns the revert: a file that does not parse is not a partial change,
    it is a broken one, and every later read of it returns nonsense.
    """
    kind = _PARSERS.get(Path(path).suffix.lower())
    if kind == "python":
        try:
            ast.parse(text)
        except SyntaxError as exc:
            return Violation(
                rule="syntax",
                title=f"does not parse: {exc.msg}",
                path=path,
                line=exc.lineno or 1,
                excerpt=(exc.text or "").strip()[:120],
                why="A file that does not parse cannot be read by any later step.",
            )
    elif kind == "json":
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            return Violation(
                rule="syntax",
                title=f"is not valid JSON: {exc.msg}",
                path=path,
                line=exc.lineno,
                excerpt="",
                why="A file that does not parse cannot be read by any later step.",
            )
    return None


def _lint(root: Path, relative: str) -> list[dict[str, Any]] | None:
    """What the project's own linter says, or ``None`` when it cannot say.

    Parsing is not enough and a real run proved it. Asked to add one method,
    the model inserted its block in the middle of another method, leaving that
    method's tail orphaned after a comment and its own definition duplicated.
    The file parsed. The new method worked. It was broken, and the syntax gate
    passed it, and the loop reported success.

    A linter finds that in milliseconds: redefinition of an existing name, and
    a variable assigned and never used where the tail was severed. These are
    the shape of mistake an editing model makes, and they are exactly what a
    linter is for.
    """
    if Path(relative).suffix.lower() != ".py":
        return None
    binary = shutil.which("ruff") or str(Path(sys.executable).parent / "ruff")
    if not Path(binary).exists() and not shutil.which("ruff"):
        return None
    try:
        finished = subprocess.run(
            [binary, "check", "--output-format", "json", "--force-exclude", relative],
            cwd=str(root), capture_output=True, text=True, timeout=60, check=False,
        )
        findings = json.loads(finished.stdout or "[]")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None
    out = [
        {
            "line": (f.get("location") or {}).get("row", 1),
            "code": f.get("code") or "",
            "message": f.get("message") or "",
        }
        for f in findings
        if isinstance(f, dict)
    ]
    # E902 is the linter saying it could not read the file, not something it
    # found in it. Counting that as a new finding would report a defect on the
    # evidence that no evidence was gathered, which is the shape this whole
    # gate exists to refuse.
    if any(f["code"] == "E902" for f in out):
        return None
    return out


def _formatted(root: Path, relative: str) -> bool | None:
    """Whether the file matches the project's formatter, or ``None`` if unknown.

    Only useful as a before-and-after pair. A repository that does not use the
    formatter has every file report unformatted, so the answer is meaningless
    on its own and a blanket check would flag every edit ever made.
    """
    if Path(relative).suffix.lower() != ".py" or not shutil.which("ruff"):
        return None
    try:
        finished = subprocess.run(
            ["ruff", "format", "--check", "--force-exclude", relative],
            cwd=str(root), capture_output=True, text=True, timeout=60, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return finished.returncode == 0


def _diagnostics(root: Path, relative: str) -> list[dict[str, Any]] | None:
    """Errors the language server reports, or ``None`` when it cannot say."""
    if Path(relative).suffix.lower() not in _LSP_EXTENSIONS:
        return None
    try:
        from mimir.lsp import LspUnavailable, get_client

        client = get_client(root, root / relative)
        items = client.diagnostics(root / relative)
    except LspUnavailable:
        return None
    except Exception as exc:  # noqa: BLE001 - a flaky server must not block a write
        log.debug("diagnostics_unavailable", path=relative, error=str(exc))
        return None
    return [d for d in items if d.get("severity") == "error"]


def verify_change(
    root: Path,
    relative: str,
    *,
    updated: str,
    original: str | None,
    settings: Any,
    baseline: list[dict[str, Any]] | None = None,
    baseline_lint: list[dict[str, Any]] | None = None,
    was_formatted: bool = False,
) -> ChangeReport:
    """Check one written file. Reverts only what cannot be read.

    ``original`` is ``None`` for a new file, in which case a revert deletes it.
    ``baseline`` is the diagnostics before the change, so a file that was
    already failing to compile is not blamed on the edit that touched it.
    """
    report = ChangeReport(path=relative)
    target = Path(root) / relative

    syntax = check_syntax(relative, updated)
    if syntax is not None:
        report.violations.append(syntax)
        report.checks_run.append("syntax")
        try:
            if original is None:
                target.unlink(missing_ok=True)
            else:
                target.write_text(original, encoding="utf-8")
            report.reverted = True
        except OSError as exc:  # pragma: no cover - the write just succeeded
            log.warning("revert_failed", path=relative, error=str(exc))
        return report
    if Path(relative).suffix.lower() in _PARSERS:
        report.checks_run.append("syntax")

    # Did the change connect what it added. Four attempts at one task passed
    # syntax, lint and tests while two of them introduced a constant and never
    # used it, and the incomplete ones had the smallest diffs.
    from mimir.verify import definitions

    for issue in definitions.check(original, updated):
        report.violations.append(
            Violation(
                rule=f"definition-{issue.kind}",
                title=issue.message,
                path=relative,
                line=issue.line,
                excerpt="",
                why=(
                    "A change that adds something nothing uses has not finished "
                    "connecting it."
                    if issue.kind == "dead"
                    else "One definition too many."
                ),
                severity="error",
            )
        )
    report.checks_run.append("definitions")

    rules = load_rules(rule_roots(settings, root))
    if rules:
        report.violations.extend(check_rules(rules, relative, updated))
        report.checks_run.append(f"{len(rules)} project rule(s)")
    else:
        report.checks_skipped.append("no project rules configured")

    # Only when the file was formatted before the change. Otherwise the
    # repository does not use the formatter and every edit would be flagged.
    if was_formatted:
        after_formatting = _formatted(Path(root), relative)
        if after_formatting is None:
            report.checks_skipped.append("no formatter for this file")
        else:
            report.broke_formatting = not after_formatting
            report.checks_run.append("formatting")
    else:
        report.checks_skipped.append("file was not formatted before the change")

    lint_after = _lint(Path(root), relative)
    if lint_after is None:
        report.checks_skipped.append("no linter for this file")
    else:
        known_lint = {
            (f.get("code"), f.get("message")) for f in (baseline_lint or [])
        }
        report.new_lint = [
            f for f in lint_after if (f.get("code"), f.get("message")) not in known_lint
        ]
        report.checks_run.append("linter")

    after = _diagnostics(Path(root), relative)
    if after is None:
        report.checks_skipped.append("no language server for this file")
    else:
        # Only what this change introduced. A file that was already failing is
        # not the fault of the edit that touched it, and blaming it there
        # teaches the loop to avoid the file rather than fix it.
        known = {(d.get("line"), d.get("message")) for d in (baseline or [])}
        report.new_diagnostics = [
            d for d in after if (d.get("line"), d.get("message")) not in known
        ]
        report.checks_run.append("language server")

    return report


def baseline_diagnostics(root: Path, relative: str) -> list[dict[str, Any]] | None:
    """Diagnostics before a change, to subtract from those after it."""
    return _diagnostics(Path(root), relative)


def baseline_formatted(root: Path, relative: str) -> bool:
    """Whether the file matched the formatter before the change."""
    return _formatted(Path(root), relative) is True


def baseline_lint(root: Path, relative: str) -> list[dict[str, Any]] | None:
    """Lint findings before a change, so a file that was already failing its
    own linter is not blamed on the edit that touched it."""
    return _lint(Path(root), relative)


__all__ = [
    "ChangeReport",
    "baseline_diagnostics",
    "baseline_formatted",
    "baseline_lint",
    "check_syntax",
    "verify_change",
]
