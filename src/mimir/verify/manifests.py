"""Does the repository agree with what is running? Computed, not judged.

ADR-001 G4's last question and plan step 9's first exact engine. A
Kubernetes manifest in the repository states a name, a namespace, a replica
count and images. The entity store holds what a listing last showed for the
same workload. The comparison is a set of equalities. A model reading both
and forming an opinion was the previous mechanism and is strictly worse.

Drift is reported with both sides and when the live side was seen, because
a stale observation is not a current disagreement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mimir.logging import get_logger

log = get_logger(__name__)

_WORKLOAD_KINDS = {"Deployment", "StatefulSet", "DaemonSet"}


@dataclass(slots=True)
class DeclaredWorkload:
    kind: str
    name: str
    namespace: str
    replicas: int | None
    images: list[str]
    path: str


@dataclass(slots=True)
class Drift:
    workload: str
    field: str
    declared: Any
    observed: Any
    seen_age_s: float
    path: str

    def render(self) -> str:
        age = f"{self.seen_age_s / 3600:.1f}h ago" if self.seen_age_s else "now"
        return (f"{self.workload}.{self.field}: repository says {self.declared!r}, "
                f"cluster showed {self.observed!r} ({age}) [{self.path}]")


def declared_workloads(root: Path, *, max_files: int = 400) -> list[DeclaredWorkload]:
    """Every workload manifest in the repository, by walking YAML files."""
    out: list[DeclaredWorkload] = []
    count = 0
    for path in sorted(root.rglob("*.y*ml")):
        if any(part in (".git", "node_modules", ".venv") for part in path.parts):
            continue
        count += 1
        if count > max_files:
            break
        try:
            docs = list(yaml.safe_load_all(path.read_text(encoding="utf-8", errors="replace")))
        except (yaml.YAMLError, OSError):
            continue
        for doc in docs:
            if not isinstance(doc, dict) or doc.get("kind") not in _WORKLOAD_KINDS:
                continue
            meta = doc.get("metadata") or {}
            spec = doc.get("spec") or {}
            containers = (((spec.get("template") or {}).get("spec") or {}).get("containers") or [])
            out.append(DeclaredWorkload(
                kind=str(doc["kind"]), name=str(meta.get("name") or ""),
                namespace=str(meta.get("namespace") or ""),
                replicas=int(spec["replicas"]) if isinstance(spec.get("replicas"), int) else None,
                images=[str(c.get("image")) for c in containers if isinstance(c, dict) and c.get("image")],
                path=str(path.relative_to(root)),
            ))
    return out


def compare(declared: list[DeclaredWorkload], store: Any, *, now: float | None = None) -> list[Drift]:
    """Declared against observed, for every declared workload the store has seen."""
    now = now or time.time()
    drifts: list[Drift] = []
    for d in declared:
        kinds = (d.kind.lower(),)
        hits = [e for e in store.candidates(d.name, kinds=kinds) if e.name == d.name
                and (not d.namespace or e.namespace == d.namespace)]
        if not hits:
            continue
        e = hits[0]
        age = e.age_s(now)
        label = f"{e.namespace}/{d.name}" if e.namespace else d.name
        observed_desired = e.attrs.get("desired")
        if d.replicas is not None and isinstance(observed_desired, int) and observed_desired != d.replicas:
            drifts.append(Drift(label, "replicas", d.replicas, observed_desired, age, d.path))
        observed_images = sorted(str(i) for i in (e.attrs.get("images") or []))
        if d.images and observed_images and sorted(d.images) != observed_images:
            drifts.append(Drift(label, "images", sorted(d.images), observed_images, age, d.path))
    return drifts


__all__ = ["DeclaredWorkload", "Drift", "compare", "declared_workloads"]
