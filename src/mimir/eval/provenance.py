"""Run provenance (ADR-002 section 5)."""

from __future__ import annotations

import hashlib
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from mimir.config import Settings, get_settings
from mimir.logging import get_logger

log = get_logger(__name__)


@dataclass
class ModelIdentity:
    alias: str
    name: str
    runtime: str
    base_url: str = ""
    digest: str = ""
    quantisation: str = ""
    parameter_size: str = ""
    family: str = ""
    context_window: int = 0
    """The configured window. What MIMIR asked for."""

    served_context: int = 0
    """The window the runtime is actually serving, read back from the runtime."""

    context_mismatch: bool = False
    temperature: float = 0.0
    seed: int | None = None
    resolved: bool = False
    """False when the runtime could not be queried, so digest and quantisation are unknown and any comparison involving this run is weaker."""


PROVENANCE_SCHEMA_VERSION = 2
"""Bumped when the shape changes."""

MANDATORY_FIELDS = (
    "source.commit",
    "evaluation.corpus_hash",
    "evaluation.prompts_hash",
    "evaluation.skills_hash",
    "evaluation.enabled_tools_hash",
    "runtime.version",
)
"""Fields without which two runs cannot be honestly compared."""


@dataclass
class Provenance:
    models: dict[str, ModelIdentity] = field(default_factory=dict)
    runtime_name: str = ""
    runtime_version: str = ""
    corpus_hash: str = ""
    corpus_files: list[str] = field(default_factory=list)
    corpus_case_count: int = 0
    prompts_hash: str = ""
    skills_hash: str = ""
    mimir_commit: str = ""
    mimir_dirty: bool = False
    mimir_diff_hash: str = ""
    """Hash of the uncommitted diff plus untracked file contents."""
    offline: bool = True
    settings_digest: str = ""
    enabled_tools: list[str] = field(default_factory=list)
    enabled_tools_hash: str = ""
    enabled_capabilities: list[str] = field(default_factory=list)
    external_calls: int = 0
    contaminated: bool = False
    contaminated_reason: str = ""
    captured_at: float = 0.0
    source_changed_during_run: bool = False
    """True when the working tree moved between the start and end snapshots."""

    def to_dict(self) -> dict[str, Any]:
        """Nested, versioned, and self-describing."""
        return {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "captured_at": self.captured_at,
            "source": {
                "commit": self.mimir_commit,
                "dirty": self.mimir_dirty,
                "diff_hash": self.mimir_diff_hash,
                "changed_during_run": self.source_changed_during_run,
            },
            "evaluation": {
                "corpus_hash": self.corpus_hash,
                "corpus_files": self.corpus_files,
                "corpus_case_count": self.corpus_case_count,
                "prompts_hash": self.prompts_hash,
                "skills_hash": self.skills_hash,
                "settings_digest": self.settings_digest,
                "offline": self.offline,
                "contaminated": self.contaminated,
                "contaminated_reason": self.contaminated_reason,
                "external_calls": self.external_calls,
                "enabled_tools": self.enabled_tools,
                "enabled_tools_hash": self.enabled_tools_hash,
                "enabled_capabilities": self.enabled_capabilities,
            },
            "runtime": {"name": self.runtime_name, "version": self.runtime_version},
            "models": {alias: asdict(identity) for alias, identity in self.models.items()},
        }

    @property
    def fully_resolved(self) -> bool:
        return bool(self.mimir_commit) and all(m.resolved for m in self.models.values())


def _sha256_of_files(paths: list[Path]) -> str:
    """Stable hash over file contents, ordered by path so it does not depend on filesystem enumeration order."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: str(p)):
        try:
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        except OSError:
            continue
    return digest.hexdigest()[:16]


def corpus_fingerprint(corpus_dir: Path | None = None) -> tuple[str, list[str], int]:
    """Hash the corpus so a score is always attributable to a known case set."""
    target = corpus_dir or (Path(__file__).parent / "corpus")
    if target.is_file():
        files = [target]
    else:
        files = sorted(target.glob("*.yaml")) if target.exists() else []
    if not files:
        return "", [], 0

    from mimir.eval.harness import EvalHarness

    count = len(EvalHarness.load_corpus(corpus_dir))
    return _sha256_of_files(files), [f.name for f in files], count


def prompts_fingerprint() -> str:
    """Hash the specialist prompts and the capability tables."""
    root = Path(__file__).resolve().parents[1]
    return _sha256_of_files(
        [root / "council" / "prompts.py", root / "council" / "specialists.py"]
    )


def skills_fingerprint(settings: Settings | None = None) -> str:
    active = settings or get_settings()
    files: list[Path] = []
    for root in active.skill_roots():
        if root.exists():
            files.extend(root.rglob("SKILL.md"))
    return _sha256_of_files(files)


def _git(root: Path, *args: str, timeout: float = 20.0) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout if result.returncode == 0 else ""


def git_commit(repo: Path | None = None) -> tuple[str, bool, str]:
    """Current commit, whether the tree is dirty, and a hash of the dirty state."""
    root = repo or Path(__file__).resolve().parents[3]
    commit = _git(root, "rev-parse", "HEAD").strip()
    status = _git(root, "status", "--porcelain")
    dirty = bool(status.strip())
    if not dirty:
        return commit[:12], False, ""

    digest = hashlib.sha256()
    digest.update(_git(root, "diff", "HEAD").encode("utf-8", "replace"))
    for line in status.splitlines():
        if not line.startswith("??"):
            continue
        candidate = root / line[3:].strip()
        if candidate.is_file():
            try:
                digest.update(line.encode())
                digest.update(candidate.read_bytes())
            except OSError:
                continue
    return commit[:12], True, digest.hexdigest()[:16]


def resolve_model(alias: str, settings: Settings | None = None) -> ModelIdentity:
    """Ask the runtime what it is actually serving."""
    active = settings or get_settings()
    profile = active.models.profiles.get(alias)
    if profile is None:
        return ModelIdentity(alias=alias, name="<unconfigured>", runtime="unknown")

    identity = ModelIdentity(
        alias=alias,
        name=profile.model,
        runtime=profile.runtime,
        base_url=profile.base_url,
        context_window=profile.context_window,
        temperature=profile.temperature,
    )
    if profile.runtime not in ("ollama", "openai_compat", "llamacpp", "litellm"):
        # An echo or MLX profile has no queryable digest; mark it resolved so a
        identity.resolved = profile.runtime == "echo"
        return identity

    root = profile.base_url.rsplit("/v1", 1)[0]
    try:
        response = httpx.post(
            f"{root}/api/show", json={"model": profile.model}, timeout=8.0
        )
        if response.status_code == 200:
            body = response.json()
            details = body.get("details") or {}
            identity.quantisation = details.get("quantization_level", "")
            identity.parameter_size = details.get("parameter_size", "")
            identity.family = details.get("family", "")
            identity.digest = (body.get("digest") or "")[:19]
            if not identity.digest:
                tags = httpx.get(f"{root}/api/tags", timeout=8.0)
                if tags.status_code == 200:
                    for entry in tags.json().get("models", []):
                        if entry.get("name") == profile.model:
                            identity.digest = (entry.get("digest") or "")[:19]
                            break
            identity.resolved = True

        # Read back what the runtime is actually serving.
        try:
            running = httpx.get(f"{root}/api/ps", timeout=8.0)
            if running.status_code == 200:
                for entry in running.json().get("models", []):
                    if entry.get("name") == profile.model or entry.get(
                        "model"
                    ) == profile.model:
                        identity.served_context = int(entry.get("context_length") or 0)
                        break
        except (httpx.HTTPError, ValueError):
            pass
        if identity.served_context and identity.context_window:
            identity.context_mismatch = identity.served_context != identity.context_window
            if identity.context_mismatch:
                log.warning(
                    "context_window_mismatch",
                    alias=alias,
                    configured=identity.context_window,
                    served=identity.served_context,
                    hint="set OLLAMA_CONTEXT_LENGTH; the OpenAI shim ignores num_ctx",
                )
    except (httpx.HTTPError, ValueError) as exc:
        log.info("provenance_model_unresolved", alias=alias, error=str(exc))
    return identity


def runtime_version(settings: Settings | None = None) -> tuple[str, str]:
    active = settings or get_settings()
    default_alias = active.models.routing.default
    profile = active.models.profiles.get(default_alias)
    if profile is None or profile.runtime == "echo":
        return (profile.runtime if profile else "unknown"), "n/a"
    root = profile.base_url.rsplit("/v1", 1)[0]
    try:
        response = httpx.get(f"{root}/api/version", timeout=8.0)
        if response.status_code == 200:
            return profile.runtime, str(response.json().get("version", ""))
    except (httpx.HTTPError, ValueError):
        pass
    return profile.runtime, ""


def collect(
    *,
    settings: Settings | None = None,
    corpus_dir: Path | None = None,
    offline: bool = True,
    aliases: list[str] | None = None,
) -> Provenance:
    """Gather everything needed to reproduce or fairly compare a run."""
    active = settings or get_settings()
    wanted = aliases or sorted(active.models.profiles)
    corpus_hash, corpus_files, case_count = corpus_fingerprint(corpus_dir)
    runtime_name, version = runtime_version(active)
    commit, dirty, diff_hash = git_commit()

    return Provenance(
        captured_at=time.time(),
        models={alias: resolve_model(alias, active) for alias in wanted},
        runtime_name=runtime_name,
        runtime_version=version,
        corpus_hash=corpus_hash,
        corpus_files=corpus_files,
        corpus_case_count=case_count,
        prompts_hash=prompts_fingerprint(),
        skills_hash=skills_fingerprint(active),
        mimir_commit=commit,
        mimir_dirty=dirty,
        mimir_diff_hash=diff_hash,
        offline=offline,
        settings_digest=hashlib.sha256(
            str(active.models.routing.model_dump()).encode()
        ).hexdigest()[:16],
    )


def _dig(run: dict[str, Any], path: str) -> Any:
    node: Any = run
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def _incomplete(run: dict[str, Any]) -> list[str]:
    """Mandatory fields that are missing or empty."""
    return [f for f in MANDATORY_FIELDS if not _dig(run, f)]


def comparable(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    """Reasons two runs cannot be honestly compared."""
    problems: list[str] = []
    if not left or not right:
        return ["one of the runs has no recorded provenance"]

    for side, run in (("baseline", left), ("candidate", right)):
        version = run.get("schema_version")
        if version != PROVENANCE_SCHEMA_VERSION:
            problems.append(
                f"{side} uses provenance schema {version or 1}, this build reads "
                f"{PROVENANCE_SCHEMA_VERSION}; it predates the containment and "
                "tool-surface fixes and cannot be treated as a controlled run"
            )
            continue
        missing = _incomplete(run)
        if missing:
            problems.append(
                f"{side} has incomplete provenance, missing: {', '.join(missing)}"
            )
    if problems:
        return problems

    # Contamination is decisive.
    for side, run in (("baseline", left), ("candidate", right)):
        evaluation = run.get("evaluation") or {}
        if evaluation.get("contaminated"):
            problems.append(
                f"{side} is CONTAMINATED and cannot serve as a controlled comparison: "
                f"{evaluation.get('contaminated_reason', 'reason not recorded')}"
            )
        calls = evaluation.get("external_calls") or 0
        if calls:
            problems.append(f"{side} made {calls} external network call(s)")
        if (run.get("source") or {}).get("changed_during_run"):
            problems.append(
                f"{side} had its source change while it was running; the recorded "
                "commit does not describe what executed"
            )

    for side, run in (("baseline", left), ("candidate", right)):
        for alias, identity in (run.get("models") or {}).items():
            if isinstance(identity, dict) and identity.get("context_mismatch"):
                problems.append(
                    f"{side} ran {alias} at a context of "
                    f"{identity.get('served_context')} while configured for "
                    f"{identity.get('context_window')}; the recorded setting is not "
                    "what the runtime served"
                )

    checks = [
        ("evaluation.enabled_tools_hash",
         "different tool sets were enabled ({a} vs {b}); the models were not "
         "offered the same capabilities"),
        ("evaluation.corpus_hash",
         "different corpus ({a} vs {b}); the case set changed between runs"),
        ("evaluation.prompts_hash",
         "different specialist prompts ({a} vs {b}); this is not a model comparison"),
        ("evaluation.skills_hash", "different skills on disk ({a} vs {b})"),
        ("evaluation.offline",
         "one run used live infrastructure and the other did not ({a} vs {b}); "
         "scores are not on the same footing"),
        ("runtime.version", "different runtime version ({a} vs {b})"),
        ("source.commit", "different MIMIR commit ({a} vs {b})"),
    ]
    for path, template in checks:
        a, b = _dig(left, path), _dig(right, path)
        if a != b:
            problems.append(template.format(a=a, b=b))

    left_dirty = _dig(left, "source.dirty")
    right_dirty = _dig(right, "source.dirty")
    if left_dirty or right_dirty:
        left_hash = _dig(left, "source.diff_hash") or "unknown"
        right_hash = _dig(right, "source.diff_hash") or "unknown"
        if left_dirty and right_dirty and left_hash == right_hash:
            problems.append(
                f"both runs were made from the same dirty tree ({left_hash}); "
                "comparable to each other but not reproducible from the commit"
            )
        else:
            problems.append(
                f"different working trees (diff {left_hash} vs {right_hash}); "
                "the source differed between runs"
            )
    return problems


def source_moved(start: Provenance, end: Provenance) -> bool:
    """Did the working tree change while the run was in flight?"""
    return (
        start.mimir_commit != end.mimir_commit
        or start.mimir_dirty != end.mimir_dirty
        or start.mimir_diff_hash != end.mimir_diff_hash
    )
