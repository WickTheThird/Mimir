"""Did the change actually connect what it added.

Four attempts at one task all passed syntax, the linter and the tests. Asked to
introduce a constant and use it as a default, two of them introduced the
constant and never used it, one introduced it twice, and one did the job. The
gate could not tell them apart, and because the incomplete attempts produce the
smallest diffs, a selector preferring small diffs would have chosen one of
those.

Nothing here needs a model, and neither case is a matter of taste. A module
level name that is defined and never read is dead: whatever the change was for,
it did not connect. A name defined twice at module level is one definition too
many, and the linter does not flag it for assignments the way it does for
functions.

Both are read from the syntax tree, and both are reported only when the change
introduced them, so a file that already had dead constants is not blamed on the
edit that touched it.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class DefinitionIssue:
    name: str
    line: int
    kind: str
    """dead or duplicate."""

    @property
    def message(self) -> str:
        if self.kind == "duplicate":
            return f"{self.name} is defined more than once at module level"
        return f"{self.name} is defined here and never used anywhere in the file"


def _module_assignments(tree: ast.Module) -> dict[str, list[int]]:
    """Module level names bound by assignment, with the lines that bind them."""
    out: dict[str, list[int]] = {}
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign) and node.target is not None:
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                out.setdefault(target.id, []).append(node.lineno)
    return out


def _loads(tree: ast.Module) -> set[str]:
    """Every name the module reads, anywhere, at any depth."""
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def check(original: str | None, updated: str) -> list[DefinitionIssue]:
    """Names the change introduced that are dead or defined twice.

    A new file is not judged. The question is whether the change connected what
    it added, and for a file that did not exist a moment ago the answer is not
    in the file: a module of constants written for its importers looks exactly
    like a module of constants nobody uses.
    """
    if original is None:
        return []
    try:
        new_tree = ast.parse(updated)
    except SyntaxError:
        return []
    try:
        old_tree = ast.parse(original) if original else None
    except SyntaxError:
        old_tree = None

    before = _module_assignments(old_tree) if old_tree is not None else {}
    after = _module_assignments(new_tree)
    read = _loads(new_tree)
    exported = "__all__" in after

    issues: list[DefinitionIssue] = []
    for name, lines in after.items():
        if name.startswith("__"):
            continue
        introduced = name not in before
        if len(lines) > len(before.get(name, [])) and len(lines) > 1:
            issues.append(DefinitionIssue(name, lines[-1], "duplicate"))
            continue
        if not introduced:
            continue
        if name in read:
            continue
        # A name listed in __all__ is used by whoever imports the module, and
        # this file cannot see that. Only names the module keeps to itself can
        # be called dead from here.
        if exported and _in_all(new_tree, name):
            continue
        issues.append(DefinitionIssue(name, lines[0], "dead"))
    return issues


def _in_all(tree: ast.Module, name: str) -> bool:
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            continue
        value = getattr(node, "value", None)
        if isinstance(value, ast.List | ast.Tuple | ast.Set):
            for element in value.elts:
                if isinstance(element, ast.Constant) and element.value == name:
                    return True
    return False


__all__ = ["DefinitionIssue", "check"]
