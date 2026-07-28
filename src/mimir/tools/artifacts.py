"""Artifact store for large tool output (ADR 5.1 step 6, R7 context explosion).

Full command output, fetched web pages, and log dumps are written to disk and
referenced by an opaque ``input_ref``. Only compact summaries go into the model
context; helpers such as ``filter_logs`` operate on the stored artifact by
reference.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mimir.config import Settings, get_settings
from mimir.redaction import redact


@dataclass(slots=True)
class Artifact:
    ref: str
    kind: str
    path: Path
    size_bytes: int
    created_at: float
    session_id: str | None
    metadata: dict[str, Any]

    def read(self, limit: int | None = None) -> str:
        text = self.path.read_text(encoding="utf-8", errors="replace")
        return text[:limit] if limit else text

    def lines(self) -> list[str]:
        return self.read().splitlines()


class ArtifactStore:
    """Content-addressed-ish store under ``$MIMIR_HOME/artifacts``."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.root = self.settings.artifacts_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, Artifact] = {}
        self._load_index()

    # -- internals -------------------------------------------------------

    @property
    def _index_path(self) -> Path:
        return self.root / "index.jsonl"

    def _load_index(self) -> None:
        if not self._index_path.is_file():
            return
        for line in self._index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            path = Path(row["path"])
            if not path.is_file():
                continue
            self._index[row["ref"]] = Artifact(
                ref=row["ref"],
                kind=row.get("kind", "text"),
                path=path,
                size_bytes=row.get("size_bytes", 0),
                created_at=row.get("created_at", 0.0),
                session_id=row.get("session_id"),
                metadata=row.get("metadata", {}),
            )

    def _append_index(self, artifact: Artifact) -> None:
        row = {
            "ref": artifact.ref,
            "kind": artifact.kind,
            "path": str(artifact.path),
            "size_bytes": artifact.size_bytes,
            "created_at": artifact.created_at,
            "session_id": artifact.session_id,
            "metadata": artifact.metadata,
        }
        with self._index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")

    # -- api -------------------------------------------------------------

    def put(
        self,
        content: str,
        *,
        kind: str = "text",
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        redact_secrets: bool | None = None,
    ) -> Artifact:
        should_redact = (
            self.settings.safety.redact_secrets if redact_secrets is None else redact_secrets
        )
        body = redact(content, enabled=should_redact)
        digest = hashlib.sha256(body.encode("utf-8", "replace")).hexdigest()[:12]
        ref = f"art_{digest}_{uuid.uuid4().hex[:6]}"
        bucket = self.root / time.strftime("%Y-%m-%d")
        bucket.mkdir(parents=True, exist_ok=True)
        path = bucket / f"{ref}.txt"
        path.write_text(body, encoding="utf-8")
        artifact = Artifact(
            ref=ref,
            kind=kind,
            path=path,
            size_bytes=len(body.encode("utf-8", "replace")),
            created_at=time.time(),
            session_id=session_id,
            metadata=metadata or {},
        )
        self._index[ref] = artifact
        self._append_index(artifact)
        return artifact

    def get(self, ref: str) -> Artifact | None:
        artifact = self._index.get(ref)
        if artifact and artifact.path.is_file():
            return artifact
        return None

    def require(self, ref: str) -> Artifact:
        artifact = self.get(ref)
        if artifact is None:
            raise KeyError(f"unknown artifact reference: {ref}")
        return artifact

    def read(self, ref: str, limit: int | None = None) -> str:
        return self.require(ref).read(limit)

    def list(self, session_id: str | None = None, limit: int = 100) -> list[Artifact]:
        items = [
            a
            for a in self._index.values()
            if session_id is None or a.session_id == session_id
        ]
        items.sort(key=lambda a: a.created_at, reverse=True)
        return items[:limit]

    def head(self, ref: str, lines: int = 50) -> str:
        return "\n".join(self.require(ref).lines()[:lines])

    def prune(self, older_than_days: float) -> int:
        cutoff = time.time() - older_than_days * 86400
        removed = 0
        for ref, artifact in list(self._index.items()):
            if artifact.created_at < cutoff:
                artifact.path.unlink(missing_ok=True)
                self._index.pop(ref, None)
                removed += 1
        if removed:
            self._rewrite_index()
        return removed

    def _rewrite_index(self) -> None:
        with self._index_path.open("w", encoding="utf-8") as handle:
            for artifact in self._index.values():
                handle.write(
                    json.dumps(
                        {
                            "ref": artifact.ref,
                            "kind": artifact.kind,
                            "path": str(artifact.path),
                            "size_bytes": artifact.size_bytes,
                            "created_at": artifact.created_at,
                            "session_id": artifact.session_id,
                            "metadata": artifact.metadata,
                        },
                        default=str,
                    )
                    + "\n"
                )


_store: ArtifactStore | None = None


def get_artifact_store(settings: Settings | None = None) -> ArtifactStore:
    global _store
    if _store is None:
        _store = ArtifactStore(settings)
    return _store


def reset_artifact_store() -> None:
    global _store
    _store = None
