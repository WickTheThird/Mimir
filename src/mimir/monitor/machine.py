"""Host resource sampling."""

from __future__ import annotations

import contextlib
import os
import subprocess
import time
from dataclasses import dataclass, field

try:  # pragma: no cover - exercised by absence, not by tests
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

GIB = float(1 << 30)


@dataclass
class Reading:
    """A value that may not exist, carrying why when it does not."""

    value: float | None = None
    unavailable: str = ""

    @property
    def known(self) -> bool:
        return self.value is not None


@dataclass
class ProcessInfo:
    pid: int
    label: str
    cmdline: str
    cpu_percent: float
    rss_bytes: int
    started_at: float

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.started_at)


@dataclass
class MachineSample:
    cpu_percent: Reading = field(default_factory=Reading)
    cpu_count: int = 0
    per_core: list[float] = field(default_factory=list)
    ram_used_bytes: int = 0
    ram_total_bytes: int = 0
    ram_percent: Reading = field(default_factory=Reading)
    swap_used_bytes: int = 0
    load: tuple[float, float, float] | None = None
    thermal: str = ""
    thermal_detail: str = ""
    cpu_speed_limit: Reading = field(default_factory=Reading)
    processes: list[ProcessInfo] = field(default_factory=list)
    sampled_at: float = 0.0

    @property
    def load_per_core(self) -> float | None:
        if self.load is None or not self.cpu_count:
            return None
        return self.load[0] / self.cpu_count


_MIMIR_MARKERS = ("mimir", "uvicorn")
_RUNTIME_MARKERS = ("ollama", "llama-server", "mlx_lm", "llama.cpp")


def _classify(name: str, cmdline: list[str]) -> str | None:
    """Return a display label for processes worth showing, else None."""
    joined = " ".join(cmdline)
    lowered = name.lower()

    if lowered.startswith("ollama"):
        if len(cmdline) > 1 and cmdline[1] == "pull":
            return f"ollama pull {cmdline[2] if len(cmdline) > 2 else ''}".strip()
        if len(cmdline) > 1 and cmdline[1] == "runner":
            return "ollama runner"
        return f"ollama {cmdline[1]}" if len(cmdline) > 1 else "ollama"
    if any(lowered.startswith(m) for m in _RUNTIME_MARKERS):
        return lowered

    # A MIMIR process is identified by its entry point, not by the string
    for index, token in enumerate(cmdline[:2]):
        if " " in token:
            continue
        base = os.path.basename(token)
        if base in ("mimir", "mimir.exe"):
            verb = cmdline[index + 1] if len(cmdline) > index + 1 else ""
            return f"mimir {verb}".strip()
        if base.startswith("python") and "-m" in cmdline[:3]:
            try:
                module = cmdline[cmdline.index("-m") + 1]
            except (ValueError, IndexError):
                continue
            if module.startswith("mimir"):
                return f"python -m {module}"
    if "mimir.api" in joined and "uvicorn" in joined:
        return "mimir serve"
    return None


def _thermal() -> tuple[str, str, Reading]:
    """Read what macOS will actually tell an unprivileged process."""
    try:
        result = subprocess.run(
            ["pmset", "-g", "therm"], capture_output=True, text=True, timeout=3
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return "unavailable", f"pmset failed: {exc}", Reading(unavailable="pmset failed")
    text = result.stdout.strip()
    if not text:
        return "unavailable", "pmset returned nothing", Reading(unavailable="not reported")

    limit = Reading(unavailable="not reported")
    for line in text.splitlines():
        if "CPU_Speed_Limit" in line:
            with contextlib.suppress(IndexError, ValueError):
                limit = Reading(value=float(line.split("=")[1].strip()))
    if limit.known and limit.value is not None and limit.value < 100:
        return "throttled", f"CPU speed limit {limit.value:.0f}%", limit
    if "No thermal warning level has been recorded" in text:
        # Precisely what this means: nothing has been recorded.
        return "no warning recorded", "pmset has recorded no thermal warning", limit
    if "warning level" in text.lower():
        return "warning", text.splitlines()[0], limit
    return "no warning recorded", text.splitlines()[0][:80], limit


def sample(*, interval: float = 0.0) -> MachineSample:
    """Take one reading. ``interval`` blocks that long to measure CPU deltas."""
    out = MachineSample(sampled_at=time.time())

    try:
        out.load = os.getloadavg()
    except OSError:
        out.load = None

    out.thermal, out.thermal_detail, out.cpu_speed_limit = _thermal()

    if psutil is None:
        reason = "psutil not installed"
        out.cpu_percent = Reading(unavailable=reason)
        out.ram_percent = Reading(unavailable=reason)
        return out

    out.cpu_count = psutil.cpu_count() or 0
    out.cpu_percent = Reading(value=psutil.cpu_percent(interval=interval or None))
    try:
        out.per_core = psutil.cpu_percent(interval=None, percpu=True)
    except (OSError, RuntimeError):
        out.per_core = []

    virtual = psutil.virtual_memory()
    out.ram_used_bytes = virtual.total - virtual.available
    out.ram_total_bytes = virtual.total
    out.ram_percent = Reading(value=virtual.percent)
    try:
        out.swap_used_bytes = psutil.swap_memory().used
    except (OSError, RuntimeError):
        out.swap_used_bytes = 0

    for proc in psutil.process_iter(["name", "cmdline", "memory_info", "create_time"]):
        try:
            info = proc.info
            label = _classify(info.get("name") or "", info.get("cmdline") or [])
            if label is None:
                continue
            memory = info.get("memory_info")
            out.processes.append(
                ProcessInfo(
                    pid=proc.pid,
                    label=label,
                    cmdline=" ".join(info.get("cmdline") or [])[:160],
                    cpu_percent=proc.cpu_percent(interval=None),
                    rss_bytes=getattr(memory, "rss", 0) or 0,
                    started_at=info.get("create_time") or 0.0,
                )
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    out.processes.sort(key=lambda p: p.rss_bytes, reverse=True)
    return out
