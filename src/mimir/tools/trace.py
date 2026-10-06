"""Trace a feature end to end: where it enters, what handles it, what it calls, what consumes it."""

from __future__ import annotations

import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mimir.logging import get_logger
from mimir.safety.risk import RiskClass
from mimir.tools.base import Capability, ToolContext, ToolResult, tool
from mimir.tools.code import _repo_root

log = get_logger(__name__)

EXCLUDE = ("vendor", ".claude", "node_modules", "third_party", ".git", "dist", "build", ".venv", "venv")
CODE_GLOBS = ("*.go", "*.py", "*.ts", "*.tsx", "*.js", "*.java", "*.kt", "*.rb", "*.rs")
_ROUTE = re.compile(
    r"(Handle(Func)?\(|\.(Post|Get|Put|Patch|Delete|Handle)\(|router\.|@(app|router|bp)\.(get|post|put|patch|delete|route)"
    r"|\bpath\(|\bre_path\(|app\.(get|post|put|patch|delete)\(|@(Get|Post|Put|Delete|Request)Mapping)", re.I)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_STOP = {"if", "for", "func", "return", "len", "make", "append", "string", "int", "error", "nil", "fmt",
         "Errorf", "Sprintf", "Logf", "Printf", "New", "Error", "Context", "Background", "WithTimeout",
         "Marshal", "Unmarshal", "ReadAll", "Header", "Add", "Set", "Get", "Write", "WriteHeader",
         "Encode", "Decode", "NewEncoder", "NewDecoder", "Quote", "Itoa", "Atoi", "print", "range", "def",
         "self", "await", "super", "dict", "list", "str"}


class TraceInput(BaseModel):
    feature: str = Field(description="The feature in the operator's words, e.g. 'embedded signup'.")
    repo: str | None = None
    include_tests: bool = False


def _rg(root: Path, pattern: str, *, ignore_case: bool = True, tests: bool = False, max_count: int = 400) -> list[tuple[str, int, str]]:
    exe = shutil.which("rg")
    if not exe:
        return _py_grep(root, pattern, ignore_case=ignore_case, tests=tests)[:max_count]
    cmd = [exe, "-n", "--no-heading", "--color=never", "-m", "50"]
    if ignore_case:
        cmd.append("-i")
    for g in CODE_GLOBS:
        cmd += ["--glob", g]
    for d in EXCLUDE:
        cmd += ["--glob", f"!**/{d}/**"]
    if not tests:
        cmd += ["--glob", "!*_test.go", "--glob", "!**/test_*.py", "--glob", "!**/*.test.*", "--glob", "!**/*.spec.*"]
    cmd += ["-e", pattern, str(root)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows: list[tuple[str, int, str]] = []
    for line in out.splitlines()[:max_count]:
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[1].isdigit():
            rows.append((str(Path(parts[0]).relative_to(root)), int(parts[1]), parts[2].strip()))
    return rows


def _py_grep(root: Path, pattern: str, *, ignore_case: bool, tests: bool) -> list[tuple[str, int, str]]:
    rx = re.compile(pattern, re.I if ignore_case else 0)
    rows = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix not in {g[1:] for g in CODE_GLOBS} or any(part in EXCLUDE for part in p.parts):
            continue
        if not tests and (p.name.endswith("_test.go") or p.name.startswith("test_")):
            continue
        try:
            for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                if rx.search(line):
                    rows.append((str(p.relative_to(root)), i, line.strip()))
        except OSError:
            continue
    return rows


def key_terms(feature: str) -> list[str]:
    """The phrase, its joined identifier forms, then single words, most specific first."""
    words = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", feature) if len(w) > 2]
    out: list[str] = []
    if len(words) >= 2:
        out += ["".join(words), "_".join(words), words[0] + "".join(w.title() for w in words[1:])]
    out += sorted(words, key=len, reverse=True)
    return list(dict.fromkeys(out))


def choose_term(root: Path, feature: str, tests: bool) -> tuple[str, int]:
    """The term with the most identifier and route hits in code, not comments or strings alone."""
    best, best_score = "", 0
    for term in key_terms(feature):
        rows = _rg(root, re.escape(term), tests=tests)
        score = sum(2 if _ROUTE.search(t) else 1 for _, _, t in rows if not t.lstrip().startswith(("//", "#", "*")))
        if score > best_score:
            best, best_score = term, score
        if best_score >= 20 and term.count("_") == 0 and " " not in term and term in feature.lower().replace(" ", ""):
            break
    return best, best_score


def _definition(root: Path, name: str, tests: bool) -> tuple[str, int, str] | None:
    pat = rf"(func (\([^)]*\) )?{name}\b|def {name}\b|function {name}\b|class {name}\b|type {name}\b|const {name}\b)"
    rows = _rg(root, pat, ignore_case=False, tests=tests, max_count=5)
    return rows[0] if rows else None


def _lines(root: Path, path: str) -> list[str]:
    try:
        return (root / path).read_text(errors="replace").splitlines()
    except OSError:
        return []


def _calls_in(root: Path, path: str, line: int, *, exclude: set[str]) -> list[str]:
    """Calls in a function body, method calls included, in order of first use."""
    text = "\n".join(_body(root, path, line))
    names = re.findall(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", text) + _CALL.findall(text)
    return [c for c in dict.fromkeys(names) if c not in _STOP and c not in exclude and len(c) > 3]


def _body(root: Path, path: str, line: int, limit: int = 160) -> list[str]:
    try:
        lines = (root / path).read_text(errors="replace").splitlines()
    except OSError:
        return []
    body = []
    for i in range(line, min(len(lines), line + limit)):
        text = lines[i]
        if i > line and re.match(r"^(func |def |class |type |export function |function )", text):
            break
        body.append(text)
    return body


def trace(root: Path, feature: str, *, tests: bool = False) -> dict[str, Any]:
    term, score = choose_term(root, feature, tests)
    result: dict[str, Any] = {"feature": feature, "term": term, "routes": [], "handlers": [], "calls": [],
                              "consumers": [], "states": [], "wiring": []}
    if not term:
        return result
    rx = re.compile(re.escape(term), re.I)
    # 1. entry points: route registrations naming the term; the handler may be on the next line
    for path, line, text in _rg(root, re.escape(term), tests=tests):
        if _ROUTE.search(text):
            skip = ("Handle", "HandleFunc", "Post", "Get", "Put", "Patch", "Delete", "pat")
            pick = lambda txt: [m for m in _CALL.findall(txt) if m not in _STOP and m not in skip]  # noqa: E731
            handlers = pick(text)
            if not handlers:
                # Only a registration that left its handler for the next line may borrow from it.
                handlers = pick(" ".join(l.strip() for l in _lines(root, path)[line:line + 1]))
            result["routes"].append({"path": path, "line": line, "text": text[:200], "handlers": handlers[:3]})
    # 2. handlers: their definitions, and what they call
    seen_calls: set[str] = set()
    for name in dict.fromkeys(h for r in result["routes"] for h in r["handlers"]):
        d = _definition(root, name, tests)
        if not d:
            continue
        calls = _calls_in(root, d[0], d[1], exclude={name})
        # A handler that only delegates (postXPublic -> postXCommon) is followed one level down.
        for inner in [c for c in calls if rx.search(c)][:2]:
            d2 = _definition(root, inner, tests)
            if d2:
                calls += [c for c in _calls_in(root, d2[0], d2[1], exclude={inner, name}) if c not in calls]
        result["handlers"].append({"name": name, "path": d[0], "line": d[1], "calls": calls[:14]})
        seen_calls.update(calls)
    # 3. the calls that resolve to code in this repository
    for name in sorted(seen_calls, key=lambda c: (not rx.search(c), c))[:30]:
        d = _definition(root, name, tests)
        if d and not any(c["name"] == name for c in result["calls"]):
            result["calls"].append({"name": name, "path": d[0], "line": d[1], "text": d[2][:160]})
    # 4. consumers: types and functions named for the term that no handler calls (workers, FSMs, jobs)
    named = _rg(root, rf"(func (\([^)]*\) )?\w*{term}\w*\s*\(|type \w*{term}\w*|class \w*{term}\w*|def \w*{term}\w*)", tests=tests)
    handler_names = {h["name"] for h in result["handlers"]} | seen_calls
    for path, line, text in named:
        m = re.search(r"(?:func (?:\([^)]*\) )?|type |class |def )(\w+)", text)
        ident = m.group(1) if m else ""
        if ident and ident not in handler_names and not any(c["path"] == path and c["line"] == line for c in result["consumers"]):
            result["consumers"].append({"name": ident, "path": path, "line": line, "text": text[:160]})
    # Workers and storage before request types: the interesting hops are the ones no handler names.
    noise = re.compile(r"(payload|response|request|session|error|args|input)$", re.I)
    result["consumers"].sort(key=lambda c: (bool(noise.search(c["name"])), "/http/" in c["path"], c["path"], c["line"]))
    result["consumers"] = result["consumers"][:25]
    # 5. where the consumers are constructed or started (main, cmd, wiring)
    for c in [c for c in result["consumers"] if not c["text"].lstrip().startswith("type ")
              or re.search(r"(FSM|Worker|Consumer|Job|Machine|Processor|Runner)$", c["name"])][:12]:
        for path, line, text in _rg(root, rf"\b{c['name']}\(", ignore_case=False, tests=tests, max_count=20):
            if (path, line) != (c["path"], c["line"]) and re.search(r"(^|/)(cmd|main|app|server|wire|bootstrap)", path):
                result["wiring"].append({"name": c["name"], "path": path, "line": line, "text": text[:160]})
    # 6. state machine: string cases in consumer files, in source order
    for path in dict.fromkeys(c["path"] for c in result["consumers"]):
        lines = _lines(root, path)
        cases = [(ln, t) for p, ln, t in _rg(root, r'case\s+"[A-Z][A-Z0-9_]+"\s*:', ignore_case=False, tests=tests) if p == path]
        for i, (line, text) in enumerate(cases):
            end = cases[i + 1][0] - 1 if i + 1 < len(cases) else min(len(lines), line + 80)
            block = "\n".join(lines[line:end])
            calls = [c for c in dict.fromkeys(re.findall(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", block))
                     if c not in _STOP and len(c) > 3][:6]
            goes_to = re.findall(r'State\s*=\s*"([A-Z0-9_]+)"', block)
            result["states"].append({"path": path, "line": line, "state": re.search(r'"([A-Z0-9_]+)"', text).group(1),
                                     "calls": calls, "next": list(dict.fromkeys(goes_to))})
    return result


def render(t: dict[str, Any]) -> str:
    if not t["term"]:
        return f"no code names anything like '{t['feature']}'"
    lines = [f"traced '{t['feature']}' as '{t['term']}'"]
    if t["routes"]:
        lines.append("ENTRY (routes):")
        lines += [f"  {r['path']}:{r['line']}  {r['text'][:120]}" for r in t["routes"][:12]]
    if t["handlers"]:
        lines.append("HANDLERS:")
        lines += [f"  {h['name']}  {h['path']}:{h['line']}  calls {', '.join(h['calls'][:8])}" for h in t["handlers"]]
    if t["calls"]:
        lines.append("CALLED:")
        lines += [f"  {c['name']}  {c['path']}:{c['line']}" for c in t["calls"][:15]]
    if t["consumers"]:
        lines.append("CONSUMERS / WORKERS:")
        lines += [f"  {c['name']}  {c['path']}:{c['line']}" for c in t["consumers"][:12]]
    if t["wiring"]:
        lines.append("STARTED FROM:")
        lines += [f"  {w['name']}  {w['path']}:{w['line']}  {w['text'][:100]}" for w in t["wiring"][:6]]
    if t["states"]:
        lines.append("STATES (in source order):")
        lines += [f"  {s['state']}  {s['path']}:{s['line']}  calls {', '.join(s.get('calls', [])) or '-'}"
                  + (f"  -> {', '.join(s['next'])}" if s.get("next") else "") for s in t["states"][:20]]
    return "\n".join(lines)


_VERB_PATH = re.compile(r'\.(Post|Get|Put|Patch|Delete|Handle)\(\s*"([^"]+)"|@(?:app|router|bp)\.(get|post|put|patch|delete)\(\s*["\']([^"\']+)', re.I)


def answer_markdown(t: dict[str, Any]) -> str:
    """The trace as an ordered, cited end-to-end answer. Every line comes from the code."""
    if not t["term"]:
        return f"No code in this repository names anything like '{t['feature']}'."
    where = {c["name"]: f"{c['path']}:{c['line']}" for c in t["calls"]}
    rx = re.compile(re.escape(t["term"]), re.I)
    out = [f"**{t['feature']}** is implemented under the name `{t['term']}`. End to end:", ""]
    step = 1
    primary = [r for r in t["routes"] if rx.search(r["text"].split('"')[1] if '"' in r["text"] else r["text"])]
    if primary:
        out.append(f"**{step}. Entry points**"); step += 1
        for r in primary[:10]:
            m = _VERB_PATH.search(r["text"])
            verb, path = ((m.group(1) or m.group(3) or "").upper(), m.group(2) or m.group(4)) if m else ("", "")
            handler = r["handlers"][0] if r["handlers"] else "?"
            out.append(f"- `{verb} {path}` → `{handler}` ({r['path']}:{r['line']})" if path else f"- {r['text'][:100]} ({r['path']}:{r['line']})")
        out.append("")
    if t["handlers"]:
        out.append(f"**{step}. Handlers and what they call**"); step += 1
        routed = {h for r in primary for h in r["handlers"]}
        for h in [h for h in t["handlers"] if h["name"] in routed][:8]:
            hops = [f"`{c}` ({where[c]})" for c in h["calls"] if c in where and c != h["name"]]
            out.append(f"- `{h['name']}` ({h['path']}:{h['line']})" + (f" calls {', '.join(hops[:6])}" if hops else ""))
        out.append("")
    workers = [c for c in t["consumers"] if not re.search(r"(Type|Metadata|Log|Filter|PhoneNumber|Payload|Response|Request|Session|Error)$", c["name"])]
    if workers or t["wiring"]:
        out.append(f"**{step}. Background processing**"); step += 1
        for c in workers[:8]:
            out.append(f"- `{c['name']}` ({c['path']}:{c['line']})")
        for w in t["wiring"][:4]:
            out.append(f"- started by `{w['name']}` in {w['path']}:{w['line']}")
        out.append("")
    if t["states"]:
        out.append(f"**{step}. States, in order**"); step += 1
        for i, st in enumerate(t["states"][:20], 1):
            calls = ", ".join(f"`{c}`" for c in st.get("calls", [])[:5])
            nxt = " → " + " or ".join(f"`{n}`" for n in st["next"]) if st.get("next") else ""
            out.append(f"{i}. `{st['state']}` ({st['path']}:{st['line']})" + (f": {calls}" if calls else "") + nxt)
        out.append("")
    other = [r for r in t["routes"] if r not in primary]
    if other:
        out.append("**Related routes that mention it**")
        out += [f"- {r['text'][:90]} ({r['path']}:{r['line']})" for r in other[:6]]
    return "\n".join(out).rstrip()


@tool(
    "trace_feature",
    description=(
        "Trace a feature end to end in a repository without guessing: route registrations that "
        "name it, the handler each route calls and what that handler calls, the workers or state "
        "machines named for it, where those are started, and their states in order. Each hop is "
        "file:line. Use this first for 'where is X implemented' and 'trace X end to end'."
    ),
    capability=Capability.REPOSITORY,
    risk=RiskClass.R1,
    tags=("repository", "flow", "trace"),
)
async def trace_feature(args: TraceInput, ctx: ToolContext) -> ToolResult:
    root = await _repo_root(ctx, args.repo)
    t = trace(root, args.feature, tests=args.include_tests)
    return ToolResult(ok=bool(t["term"]), tool="trace_feature", summary=render(t), data=t)


__all__ = ["answer_markdown", "choose_term", "key_terms", "render", "trace", "trace_feature"]
