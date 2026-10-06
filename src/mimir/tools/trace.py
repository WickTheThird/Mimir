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


_FUNC_START = re.compile(r"^\s*(func\s+(\([^)]*\)\s*)?(\w+)|def\s+(\w+)|(?:export\s+)?(?:async\s+)?function\s+(\w+)|(?:public|private|protected)[^(=]*\s(\w+)\s*\()")
_COMPARE = re.compile(r"\bcase\b|==|!=|\bswitch\b|\bif\b|\bmatch\b|\bwhen\b|\bin\s*\(")
CODE_TOKEN = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+){1,}\b|\b[a-z]+(?:[A-Z][a-z0-9]+){2,}\b|\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+){2,}\b|\b[a-z][a-z0-9]*(?:_[a-z0-9]+){2,}\b|`([^`]{3,80})`")


def code_tokens(text: str) -> list[str]:
    """Identifiers and literals the operator typed: SCREAMING_SNAKE, camelCase, snake_case, `backticked`."""
    out = []
    for m in CODE_TOKEN.finditer(text):
        out.append(m.group(1) or m.group(0))
    return list(dict.fromkeys(out))


def _enclosing(root: Path, path: str, line: int) -> tuple[str, int] | None:
    lines = _lines(root, path)
    for i in range(min(line, len(lines)) - 1, -1, -1):
        m = _FUNC_START.match(lines[i])
        if m:
            name = next((g for g in (m.group(3), m.group(4), m.group(5), m.group(6)) if g), None)
            if name:
                return name, i + 1
    return None


def trace_symbol(root: Path, token: str, *, tests: bool = False) -> dict[str, Any]:
    """Where a literal or identifier is defined, bound, compared, and how requests reach it."""
    r: dict[str, Any] = {"token": token, "literal": [], "bindings": [], "uses": [], "callers": [], "routes": []}
    for path, line, text in _rg(root, re.escape(token), ignore_case=False, tests=tests):
        r["literal"].append({"path": path, "line": line, "text": text[:180]})
        m = re.match(r"^\s*(?:const\s+|var\s+|let\s+|final\s+|static\s+)?([A-Za-z_]\w*)\s*(?::\s*\w+\s*)?:?=\s*[\"'`]" + re.escape(token), text)
        if m:
            r["bindings"].append({"name": m.group(1), "path": path, "line": line})
    names = [b["name"] for b in r["bindings"]] or [token]
    seen_funcs: dict[str, tuple[str, int]] = {}
    for name in names:
        for path, line, text in _rg(root, rf"\b{re.escape(name)}\b", ignore_case=False, tests=tests):
            if any(b["path"] == path and b["line"] == line for b in r["bindings"]):
                continue
            enc = _enclosing(root, path, line)
            kind = "compared" if _COMPARE.search(text) else "used"
            r["uses"].append({"name": name, "path": path, "line": line, "text": text[:180], "kind": kind,
                              "function": enc[0] if enc else "", "function_line": enc[1] if enc else 0})
            if enc and enc[0] != name:
                seen_funcs[enc[0]] = (path, enc[1])
    # Who calls the functions that read it, one level up, and the routes that reach those callers.
    for func, (fpath, fline) in list(seen_funcs.items())[:6]:
        for path, line, text in _rg(root, rf"\b{re.escape(func)}\s*\(", ignore_case=False, tests=tests, max_count=30):
            if (path, line) == (fpath, fline):
                continue
            enc = _enclosing(root, path, line)
            caller = enc[0] if enc else ""
            r["callers"].append({"function": func, "caller": caller, "path": path, "line": line, "text": text[:180]})
    handler_names = {c["caller"] for c in r["callers"] if c["caller"]}
    # A delegating caller (xPublic -> xCommon) is followed one more level so its route is found.
    for c in list(r["callers"]):
        if not c["caller"]:
            continue
        for path, line, text in _rg(root, rf"\b{re.escape(c['caller'])}\s*\(", ignore_case=False, tests=tests, max_count=20):
            enc = _enclosing(root, path, line)
            if enc and enc[0] != c["caller"]:
                handler_names.add(enc[0])
    for path, line, text in _rg(root, r"(Handle(Func)?\(|\.(Post|Get|Put|Patch|Delete)\(|@(app|router)\.)", ignore_case=False, tests=tests):
        span = " ".join([text, *[l.strip() for l in _lines(root, path)[line:line + 1]]])
        hit = [h for h in handler_names if re.search(rf"\b{re.escape(h)}\s*\(", span)]
        if hit:
            m = _VERB_PATH.search(text)
            r["routes"].append({"path": path, "line": line, "handler": hit[0],
                                "verb": (m.group(1) or m.group(3) or "").upper() if m else "",
                                "url": (m.group(2) or m.group(4)) if m else ""})
    return r


def symbol_markdown(r: dict[str, Any]) -> str:
    if not r["literal"] and not r["uses"]:
        return f"`{r['token']}` does not appear in this repository's code."
    out = [f"**`{r['token']}`**", ""]
    if r["bindings"]:
        out.append("**Defined as**")
        out += [f"- `{b['name']}` ({b['path']}:{b['line']})" for b in r["bindings"]]
        out.append("")
    compared = [u for u in r["uses"] if u["kind"] == "compared"]
    if compared:
        out.append("**Where it is received and checked**")
        out += [f"- {u['path']}:{u['line']} in `{u['function']}`: `{u['text'][:110]}`" for u in compared]
        out.append("")
    other = [u for u in r["uses"] if u["kind"] != "compared"]
    if other:
        out.append("**Other uses**")
        out += [f"- {u['path']}:{u['line']}" + (f" in `{u['function']}`" if u["function"] else "") + f": `{u['text'][:100]}`" for u in other[:10]]
        out.append("")
    if r["callers"]:
        out.append("**How requests get there**")
        out += [f"- `{c['caller'] or '?'}` calls `{c['function']}` ({c['path']}:{c['line']}): `{c['text'][:100]}`" for c in r["callers"][:8]]
        out += [f"- `{rt['verb']} {rt['url']}` → `{rt['handler']}` ({rt['path']}:{rt['line']})" for rt in r["routes"][:6] if rt["url"]]
        out.append("")
    stray = [l for l in r["literal"] if not any(b["path"] == l["path"] and b["line"] == l["line"] for b in r["bindings"])]
    if stray:
        out.append("**The literal also appears**")
        out += [f"- {l['path']}:{l['line']}: `{l['text'][:100]}`" for l in stray[:6]]
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


__all__ = ["gap_verdict", "gap_markdown", "repo_shape", "shape_markdown", "code_tokens", "symbol_markdown", "trace_symbol", "answer_markdown", "choose_term", "key_terms", "render", "trace", "trace_feature"]


class SymbolInput(BaseModel):
    symbol: str = Field(description="An identifier or literal exactly as written, e.g. FINISH_WHATSAPP_BUSINESS_APP_ONBOARDING.")
    repo: str | None = None
    include_tests: bool = False


@tool(
    "trace_symbol",
    description=(
        "Follow one identifier or literal through a repository without guessing: the constant it is "
        "bound to, where that is compared or received, the functions that do it, their callers, and "
        "the routes that reach them. Use for 'where do we receive/handle/check X'."
    ),
    capability=Capability.REPOSITORY,
    risk=RiskClass.R1,
    tags=("repository", "trace", "symbol"),
)
async def trace_symbol_tool(args: SymbolInput, ctx: ToolContext) -> ToolResult:
    root = await _repo_root(ctx, args.repo)
    r = trace_symbol(root, args.symbol, tests=args.include_tests)
    return ToolResult(ok=bool(r["literal"] or r["uses"]), tool="trace_symbol", summary=symbol_markdown(r), data=r)


def repo_shape(root: Path, nouns: list[str]) -> dict[str, Any]:
    """Facts a change plan should rest on: where clients, routes, workers and migrations live."""
    shape: dict[str, Any] = {"migrations": {}, "packages": [], "routes_file": "", "workers": []}
    migs = sorted(p for p in root.rglob("*") if p.is_file() and "migrations" in p.parts
                  and not any(x in p.parts for x in EXCLUDE) and re.match(r"^\d{3,}", p.name))
    if migs:
        last = migs[-1]
        digits = re.match(r"^(\d+)", last.name).group(1)
        shape["migrations"] = {"dir": str(last.parent.relative_to(root)), "latest": last.name,
                               "next_number": str(int(digits) + 1).zfill(len(digits))}
    for noun in nouns:
        for d in sorted({p.parent for p in root.rglob("*.go")} | {p.parent for p in root.rglob("*.py")}):
            rel = str(d.relative_to(root))
            if any(x in d.parts for x in EXCLUDE):
                continue
            if noun.lower() in rel.lower() and rel not in shape["packages"]:
                shape["packages"].append(rel)
    routes = _rg(root, r"(Handle(Func)?\(|\.(Post|Get)\(|@(app|router)\.(get|post))", ignore_case=False, max_count=200)
    if routes:
        shape["routes_file"] = Counter(p for p, _, _ in routes).most_common(1)[0][0]
    for path, line, text in _rg(root, r"(type \w*(FSM|Worker|Coordinator|Consumer|Poller)\b|class \w*(Worker|Consumer|Poller))", ignore_case=False, max_count=20):
        shape["workers"].append({"path": path, "line": line, "text": text[:120]})
    return shape


def shape_markdown(shape: dict[str, Any]) -> str:
    lines = []
    m = shape.get("migrations") or {}
    if m:
        lines.append(f"- migrations live in `{m['dir']}`, latest `{m['latest']}`, next number `{m['next_number']}`")
    if shape.get("routes_file"):
        lines.append(f"- routes are registered in `{shape['routes_file']}`")
    if shape.get("packages"):
        lines.append("- related packages: " + ", ".join(f"`{p}`" for p in shape["packages"][:8]))
    for w in shape.get("workers", [])[:5]:
        lines.append(f"- background worker pattern: `{w['text'][:80]}` ({w['path']}:{w['line']})")
    return "\n".join(lines)


_TEST_PATH = re.compile(r"(_test\.go|(^|/)test_[^/]*\.py|_test\.py|\.(test|spec)\.[jt]sx?|(^|/)(tests?|mocks?|fakes?|testdata)/)")


def gap_verdict(root: Path, names: list[str]) -> list[dict[str, Any]]:
    """For each name: implemented in production code, only in tests or docs, or absent."""
    out = []
    for name in names:
        rows = _rg(root, re.escape(name), ignore_case=False, tests=True)
        code = [(p, l, t) for p, l, t in rows if not _TEST_PATH.search(p) and not t.lstrip().startswith(("//", "#", "*"))]
        tests = [(p, l, t) for p, l, t in rows if _TEST_PATH.search(p)]
        sites = []
        for p, l, t in code[:6]:
            enc = _enclosing(root, p, l)
            sites.append({"path": p, "line": l, "function": enc[0] if enc else "", "text": t[:140]})
        # The function named for it first: resume_migration -> ResumePaymentMethodMigration over an error string.
        words = {w for w in re.split(r"[_\W]+", name.lower()) if len(w) > 2}
        sites.sort(key=lambda x: -len(words & {w.lower() for w in re.findall(r"[A-Z]?[a-z]+", x["function"])}))
        status = "implemented" if code else ("tests_only" if tests else "absent")
        out.append({"name": name, "status": status, "sites": sites, "tests": len(tests)})
    return out


def gap_markdown(verdicts: list[dict[str, Any]]) -> str:
    label = {"implemented": "implemented", "tests_only": "only in tests", "absent": "not found in the code"}
    out = ["| name | status | where |", "|---|---|---|"]
    for v in verdicts:
        where = "; ".join(f"`{s['function'] or '?'}` {s['path']}:{s['line']}" for s in v["sites"][:3]) or "-"
        tests = f" ({v['tests']} test reference{'s' if v['tests'] != 1 else ''})" if v["tests"] else ""
        out.append(f"| `{v['name']}` | {label[v['status']]}{tests} | {where} |")
    return "\n".join(out)
