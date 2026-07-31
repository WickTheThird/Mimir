"""Run provenance (ADR-002 section 5).

A stored run is only reproducible if you can tell what produced it. "qwen2.5:32b"
is not an identity: the tag is mutable, the same tag can point at a different
digest or quantisation weeks later, and a comparison against an older run then
silently stops being a comparison.

So every run records the exact artifacts involved: model digest and
quantisation, runtime version, generation parameters, a hash of the corpus, a
hash of the prompts, and the MIMIR commit. If a field cannot be resolved it is
recorded as unknown rather than omitted, and :func:`comparable` reports why two
runs cannot be honestly compared.
"""

from __future__ import annotations

import hashlib
import subprocess
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
    temperature: float = 0.0
    seed: int | None = None
    resolved: bool = False
    """False when the runtime could not be queried, so digest and quantisation
    are unknown and any comparison involving this run is weaker."""


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
    """Hash of the uncommitted diff plus untracked file contents.

    A dirty run is not reproducible either way, but two dirty runs from
    different working states are not the same experiment. Without this they both
    record "dirty" and look interchangeable.
    """
    offline: bool = True
    settings_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            **{k: v for k, v in asdict(self).items() if k != "models"},
            "models": {alias: asdict(identity) for alias, identity in self.models.items()},
        }

    @property
    def fully_resolved(self) -> bool:
        return bool(self.mimir_commit) and all(m.resolved for m in self.models.values())


def _sha256_of_files(paths: list[Path]) -> str:
    """Stable hash over file contents, ordered by path so it does not depend on
    filesystem enumeration order."""
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
    """Hash the specialist prompts and the capability tables.

    A prompt edit changes behaviour as surely as a model swap does, and a
    comparison across a prompt change is not a model comparison.
    """
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
    """Current commit, whether the tree is dirty, and a hash of the dirty state.

    A dirty tree means the run cannot be reproduced from the commit alone. The
    diff hash does not fix that, but it distinguishes two different uncommitted
    states rather than labelling both "dirty" and treating them as equivalent.

    Untracked file contents are included, because a run whose behaviour depends
    on a file that was never added is exactly the case a bare `git diff` misses.
    """
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
    """Ask the runtime what it is actually serving.

    The configured tag is what was requested; the digest is what will run.
    """
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
        # deterministic test run is not reported as unreproducible.
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


def comparable(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    """Reasons two runs cannot be honestly compared.

    An empty list means the only deliberate difference is the thing under test.
    Anything else here is a confound, and reporting it is the difference between
    an experiment and a coincidence.
    """
    problems: list[str] = []
    if not left or not right:
        return ["one of the runs has no recorded provenance"]

    # A contaminated run is refused outright. Everything below compares how two
    # runs differ; none of it matters if one of them did not measure what it
    # claims to have measured.
    for side, run in (("baseline", left), ("candidate", right)):
        if run.get("contaminated"):
            problems.append(
                f"{side} is CONTAMINATED and cannot serve as a controlled "
                f"comparison: {run.get('contaminated_reason', 'reason not recorded')}"
            )

    left_tools = left.get("enabled_tools_hash")
    right_tools = right.get("enabled_tools_hash")
    if left_tools and right_tools and left_tools != right_tools:
        problems.append(
            f"different tool sets were enabled ({left_tools} vs {right_tools}); "
            "the models were not offered the same capabilities"
        )
    for side, run in (("baseline", left), ("candidate", right)):
        calls = run.get("external_calls")
        if calls:
            problems.append(f"{side} made {calls} external network call(s)")

    if left.get("corpus_hash") != right.get("corpus_hash"):
        problems.append(
            f"different corpus ({left.get('corpus_hash')} vs {right.get('corpus_hash')}); "
            "the case set changed between runs"
        )
    if left.get("prompts_hash") != right.get("prompts_hash"):
        problems.append(
            "different specialist prompts; this is not a model comparison"
        )
    if left.get("skills_hash") != right.get("skills_hash"):
        problems.append("different skills on disk")
    if left.get("offline") != right.get("offline"):
        problems.append(
            "one run used live infrastructure and the other did not; scores are "
            "not on the same footing"
        )
    if left.get("mimir_commit") != right.get("mimir_commit"):
        problems.append(
            f"different MIMIR commit ({left.get('mimir_commit') or 'unknown'} vs "
            f"{right.get('mimir_commit') or 'unknown'})"
        )
    left_dirty, right_dirty = left.get("mimir_dirty"), right.get("mimir_dirty")
    if left_dirty or right_dirty:
        left_hash = left.get("mimir_diff_hash") or "unknown"
        right_hash = right.get("mimir_diff_hash") or "unknown"
        if left_hash == right_hash and left_dirty and right_dirty:
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
