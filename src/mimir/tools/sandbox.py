"""Restricted code runner, the K3 programmatic execution helper (ADR 8 K3, 9.7)."""

from __future__ import annotations

import ast
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mimir.logging import get_logger
from mimir.models.command import RiskClass
from mimir.tools.base import Capability, ToolContext, ToolError, ToolResult, tool

log = get_logger(__name__)

# : Modules the runner refuses outright.
_DENIED_IMPORTS = frozenset(
    {
        "socket",
        "subprocess",
        "ctypes",
        "multiprocessing",
        "asyncio",
        "http",
        "urllib",
        "urllib3",
        "requests",
        "httpx",
        "ftplib",
        "smtplib",
        "telnetlib",
        "paramiko",
        "pty",
        "signal",
        "importlib",
        "pickle",
        "shelve",
        "webbrowser",
        "xmlrpc",
        "pdb",
        "code",
        "codeop",
    }
)

#: Callables that provide an escape hatch out of the AST check itself.
_DENIED_CALLS = frozenset(
    {"eval", "exec", "compile", "__import__", "breakpoint", "input", "memoryview"}
)

_DENIED_ATTRIBUTES = frozenset(
    {"system", "popen", "spawn", "spawnv", "spawnl", "execv", "execl", "fork", "forkpty"}
)

_ALLOWED_OPEN_MODES = frozenset({"r", "rb", "rt", "br", "tr"})

PRELUDE = '''\
"""Injected by MIMIR. INPUTS maps a name to a local file path."""
import json as _json, sys as _sys, os as _os

INPUTS = _json.loads(_os.environ.get("MIMIR_INPUTS", "{}"))


def read_input(name, mode="r"):
    """Read one named input artifact."""
    with open(INPUTS[name], mode) as handle:
        return handle.read()


def read_lines(name):
    return read_input(name).splitlines()


def emit(value):
    """Return structured data to MIMIR. Call this once, at the end."""
    print("@@MIMIR_EMIT@@" + _json.dumps(value, default=str))
'''

#: Marker separating emitted JSON from ordinary stdout, so a script can print
_EMIT_MARKER = "@@MIMIR_EMIT@@"


class SandboxViolation(ToolError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="sandbox_violation")


def _static_check(source: str) -> None:
    """Reject obviously dangerous constructs before execution."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ToolError(f"script has a syntax error: {exc}", code="invalid_arguments") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _DENIED_IMPORTS:
                    raise SandboxViolation(
                        f"import of '{alias.name}' is not allowed in the sandbox. "
                        "Pass data in as an artifact instead of fetching it."
                    )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in _DENIED_IMPORTS:
                raise SandboxViolation(
                    f"import from '{node.module}' is not allowed in the sandbox."
                )
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _DENIED_CALLS:
                raise SandboxViolation(f"call to '{func.id}' is not allowed in the sandbox.")
            if isinstance(func, ast.Attribute) and func.attr in _DENIED_ATTRIBUTES:
                raise SandboxViolation(
                    f"call to '.{func.attr}' is not allowed in the sandbox."
                )
            if isinstance(func, ast.Name) and func.id == "open":
                _check_open(node)


def _check_open(node: ast.Call) -> None:
    """Allow reads, refuse writes. Output leaves through ``emit``, not files."""
    mode: str | None = None
    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
        mode = str(node.args[1].value)
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            mode = str(keyword.value.value)
    if mode is None:
        return  # defaults to read
    if mode not in _ALLOWED_OPEN_MODES:
        raise SandboxViolation(
            f"open() with mode '{mode}' is not allowed; the sandbox is read-only. "
            "Return results with emit() instead of writing files."
        )


def _scrubbed_env(scrub_patterns: list[str], inputs: dict[str, str]) -> dict[str, str]:
    keep = {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "PYTHONHASHSEED"}
    env = {
        key: value
        for key, value in os.environ.items()
        if key in keep or not any(pattern in key.upper() for pattern in scrub_patterns)
    }
    # Belt and braces: drop anything that still looks like a credential even if
    for key in list(env):
        upper = key.upper()
        if any(word in upper for word in ("TOKEN", "SECRET", "PASSWORD", "KEY", "CREDENTIAL")):
            env.pop(key, None)
    env["MIMIR_INPUTS"] = json.dumps(inputs)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _limits(memory_mb: int, cpu_seconds: int):
    def apply() -> None:
        os.setsid()
        soft_bytes = memory_mb * 1024 * 1024
        for what, limit in (
            (resource.RLIMIT_AS, soft_bytes),
            (resource.RLIMIT_CPU, cpu_seconds),
            (resource.RLIMIT_FSIZE, 32 * 1024 * 1024),
            (resource.RLIMIT_NPROC, 64),
            (resource.RLIMIT_CORE, 0),
        ):
            try:
                _current_soft, current_hard = resource.getrlimit(what)
                hard = current_hard if current_hard != resource.RLIM_INFINITY else limit
                resource.setrlimit(what, (min(limit, hard), hard))
            except (ValueError, OSError):
                # macOS refuses RLIMIT_AS and RLIMIT_NPROC in some configurations.
                continue

    return apply


class RunPythonInput(BaseModel):
    code: str = Field(
        description=(
            "Python source. Use read_input(name)/read_lines(name) to read artifacts "
            "and emit(value) to return structured JSON. The standard library is "
            "available except for network, process, and dynamic-execution modules."
        )
    )
    inputs: dict[str, str] = Field(
        default_factory=dict,
        description="Map of a name to an artifact ref, made available to the script.",
    )
    timeout_s: float | None = Field(default=None, ge=1.0, le=300.0)
    description: str = Field(default="", description="What this script is for.")


@tool(
    "run_python",
    description=(
        "Run a short Python script in a restricted subprocess to parse, filter, diff, "
        "or summarise data. Use this instead of pulling a large artifact into context: "
        "read it with read_input(), reduce it, and emit() only the compact result. "
        "No network, no shell, no credentials, read-only filesystem access."
    ),
    capability=Capability.SANDBOX,
    risk=RiskClass.R1,
)
async def run_python(args: RunPythonInput, ctx: ToolContext) -> ToolResult:
    config = ctx.settings.sandbox
    if not config.enabled:
        raise ToolError("the sandbox is disabled in configuration", code="disabled")

    _static_check(args.code)

    artifacts = ctx.artifacts
    workdir = Path(tempfile.mkdtemp(prefix="mimir-sandbox-"))
    started = time.perf_counter()
    try:
        input_paths: dict[str, str] = {}
        for name, ref in args.inputs.items():
            if artifacts is None:
                raise ToolError("no artifact store is available", code="unavailable")
            artifact = artifacts.get(ref)
            if artifact is None:
                raise ToolError(f"unknown artifact ref: {ref}", code="not_found")
            # Copy rather than expose the store path, so a script cannot walk the
            target = workdir / f"{name}.dat"
            shutil.copyfile(artifact.path, target)
            input_paths[name] = str(target)

        script = workdir / "script.py"
        script.write_text(PRELUDE + "\n\n" + textwrap.dedent(args.code), encoding="utf-8")

        timeout = args.timeout_s or config.timeout_s
        env = _scrubbed_env(config.scrub_env_patterns, input_paths)

        try:
            completed = subprocess.run(
                [config.interpreter or sys.executable, str(script)],
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=_limits(config.memory_limit_mb, int(timeout) + 5),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                ok=False,
                tool="run_python",
                summary=f"script exceeded {timeout:.0f}s and was terminated",
                error=f"timeout after {timeout:.0f}s",
                error_code="timeout",
                duration_s=time.perf_counter() - started,
            )

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        cap = config.max_output_bytes
        truncated = len(stdout) > cap
        if truncated:
            stdout = stdout[:cap]

        emitted: Any = None
        printed_lines: list[str] = []
        for line in stdout.split("\n"):
            if line.startswith(_EMIT_MARKER):
                try:
                    emitted = json.loads(line[len(_EMIT_MARKER) :])
                except json.JSONDecodeError as exc:
                    printed_lines.append(f"[emit() payload was not valid JSON: {exc}]")
            else:
                printed_lines.append(line)
        printed = "\n".join(printed_lines)

        if completed.returncode != 0:
            return ToolResult(
                ok=False,
                tool="run_python",
                summary=f"script exited with code {completed.returncode}",
                error=(stderr or printed or "no error output").strip()[:2000],
                error_code="script_failed",
                data={"stdout": printed[:4000], "stderr": stderr[:4000]},
                duration_s=time.perf_counter() - started,
            )

        artifact_ref = None
        if artifacts is not None and len(printed) > 2000:
            artifact_ref = artifacts.put(
                printed,
                kind="sandbox_stdout",
                session_id=ctx.session_id,
                metadata={"description": args.description},
            ).ref

        summary = args.description or "script completed"
        if emitted is None and not printed.strip():
            summary += " (no output; call emit(value) to return structured data)"

        return ToolResult(
            ok=True,
            tool="run_python",
            summary=summary,
            data={
                "result": emitted,
                "stdout": printed[:2000] if not artifact_ref else printed[:500],
                "stderr": stderr[:1000],
            },
            artifact_ref=artifact_ref,
            truncated=truncated,
            duration_s=time.perf_counter() - started,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
