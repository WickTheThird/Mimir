"""What the system has seen, as things with names and edges between them.

ADR-003 phase 4 in its smallest useful form, built for three jobs:

* **Targeting.** "Restart the api deployment" names a workload and no
  namespace. If the store has seen exactly one `api`, that is a computed
  fact. If it has seen several, the candidate set is closed and small, which
  is what a decision model is for. Today every one of those links is
  re-derived by a generative model per question, and the operator's own
  first complaint (`messaging-whatsapp` in `messaging-squad`) was this.
* **Caller or callee.** A pod belongs to a workload in a namespace in a
  context. Walking those edges is a query, not a judgement.
* **Stale state.** Every fact carries ``seen_at``. A replica count seen
  fourteen days ago is not a current replica count.

The store grows only from observations tools already make. No crawler, and
nothing here calls a cluster. It is a derived cache: delete it and the next
investigation rebuilds it.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mimir.logging import get_logger

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id        TEXT PRIMARY KEY,
    kind      TEXT NOT NULL,
    name      TEXT NOT NULL,
    namespace TEXT NOT NULL DEFAULT '',
    context   TEXT NOT NULL DEFAULT '',
    attrs     TEXT NOT NULL DEFAULT '{}',
    seen_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS entities_kind ON entities(kind);
CREATE TABLE IF NOT EXISTS edges (
    src      TEXT NOT NULL,
    dst      TEXT NOT NULL,
    relation TEXT NOT NULL,
    seen_at  REAL NOT NULL,
    PRIMARY KEY (src, dst, relation)
);
"""


@dataclass(slots=True)
class Entity:
    id: str
    kind: str
    name: str
    namespace: str = ""
    context: str = ""
    attrs: dict[str, Any] = field(default_factory=dict)
    seen_at: float = 0.0

    @property
    def scope(self) -> str:
        """Where it lives, as one string a decision model can choose."""
        return "/".join(p for p in (self.context, self.namespace) if p) or "(unscoped)"

    def age_s(self, now: float | None = None) -> float:
        return max(0.0, (now or time.time()) - self.seen_at)


def entity_id(kind: str, name: str, namespace: str = "", context: str = "") -> str:
    return f"{kind}:{context}:{namespace}:{name}"


class EntityStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)

    # -- writing ---------------------------------------------------------

    def upsert(self, entity: Entity) -> None:
        self._db.execute(
            "INSERT INTO entities(id, kind, name, namespace, context, attrs, seen_at) "
            "VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET attrs=excluded.attrs, "
            "seen_at=excluded.seen_at",
            (entity.id, entity.kind, entity.name, entity.namespace, entity.context,
             json.dumps(entity.attrs, default=str), entity.seen_at or time.time()),
        )

    def link(self, src: str, dst: str, relation: str, seen_at: float | None = None) -> None:
        self._db.execute(
            "INSERT INTO edges(src, dst, relation, seen_at) VALUES(?,?,?,?) "
            "ON CONFLICT(src, dst, relation) DO UPDATE SET seen_at=excluded.seen_at",
            (src, dst, relation, seen_at or time.time()),
        )

    def commit(self) -> None:
        self._db.commit()

    def observe(self, tool: str, args: Any, result: Any) -> int:
        """Record what a tool result showed. Returns how many entities.

        Tolerant of shape on purpose: the tools put rows under different keys
        and this must never raise inside a tool call. An observation that
        cannot be read is logged and skipped, not a failure of the tool.
        """
        data = getattr(result, "data", None)
        if not getattr(result, "ok", False) or not isinstance(data, dict):
            return 0
        try:
            count = self._observe(tool, data)
            self.commit()
            return count
        except Exception as exc:  # noqa: BLE001 - a cache must not break a tool
            log.warning("entity_observe_failed", tool=tool, error=str(exc))
            return 0

    def _observe(self, tool: str, data: dict[str, Any]) -> int:
        now = time.time()
        count = 0
        context = str(data.get("context") or "")
        namespace = str(data.get("namespace") or "")

        if context:
            self.upsert(Entity(entity_id("context", context), "context", context, seen_at=now))
            count += 1
        for name in data.get("available_contexts") or []:
            self.upsert(Entity(entity_id("context", str(name)), "context", str(name), seen_at=now))
            count += 1
        for row in data.get("namespaces") or []:
            name = str(row.get("name") if isinstance(row, dict) else row)
            ns_id = entity_id("namespace", name, context=context)
            self.upsert(Entity(ns_id, "namespace", name, context=context,
                               attrs=row if isinstance(row, dict) else {}, seen_at=now))
            if context:
                self.link(ns_id, entity_id("context", context), "in", now)
            count += 1
        if namespace:
            ns_id = entity_id("namespace", namespace, context=context)
            self.upsert(Entity(ns_id, "namespace", namespace, context=context, seen_at=now))
            if context:
                self.link(ns_id, entity_id("context", context), "in", now)

        rows = [
            *(data.get("workloads") or []), *(data.get("matches") or []),
            *(data.get("found") or []), *(data.get("results") or []),
        ]
        for row in rows:
            if not isinstance(row, dict) or not row.get("name"):
                continue
            kind = str(row.get("kind") or "workload").lower()
            ns = str(row.get("namespace") or namespace)
            ctx = str(row.get("context") or context)
            wid = entity_id(kind, str(row["name"]), ns, ctx)
            attrs = {k: v for k, v in row.items() if k not in ("pods", "containers", "labels")}
            self.upsert(Entity(wid, kind, str(row["name"]), ns, ctx, attrs, now))
            count += 1
            if ns:
                ns_id = entity_id("namespace", ns, context=ctx)
                self.upsert(Entity(ns_id, "namespace", ns, context=ctx, seen_at=now))
                self.link(wid, ns_id, "in", now)
            for pod in row.get("pods") or []:
                pname = str(pod.get("name") if isinstance(pod, dict) else pod)
                pid = entity_id("pod", pname, ns, ctx)
                self.upsert(Entity(pid, "pod", pname, ns, ctx,
                                   pod if isinstance(pod, dict) else {}, now))
                self.link(pid, wid, "owned_by", now)
                count += 1
        for pod in data.get("pods") or []:
            if not isinstance(pod, dict) or not pod.get("name"):
                continue
            ns = str(pod.get("namespace") or namespace)
            pid = entity_id("pod", str(pod["name"]), ns, context)
            self.upsert(Entity(pid, "pod", str(pod["name"]), ns, context,
                               {k: v for k, v in pod.items() if k != "containers"}, now))
            count += 1
        return count

    # -- reading ---------------------------------------------------------

    def _row(self, r: sqlite3.Row) -> Entity:
        return Entity(r["id"], r["kind"], r["name"], r["namespace"], r["context"],
                      json.loads(r["attrs"] or "{}"), r["seen_at"])

    def candidates(self, fragment: str, *, kinds: tuple[str, ...] = (), limit: int = 26) -> list[Entity]:
        """Entities whose name contains the fragment, most recently seen first."""
        if not fragment.strip():
            return []
        sql = "SELECT * FROM entities WHERE lower(name) LIKE ?"
        params: list[Any] = [f"%{fragment.lower()}%"]
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            params += list(kinds)
        sql += " ORDER BY seen_at DESC LIMIT ?"
        params.append(limit)
        return [self._row(r) for r in self._db.execute(sql, params)]

    def get(self, id_: str) -> Entity | None:
        r = self._db.execute("SELECT * FROM entities WHERE id=?", (id_,)).fetchone()
        return self._row(r) if r else None

    def neighbours(self, id_: str, relation: str | None = None) -> list[tuple[str, Entity]]:
        sql = "SELECT e.*, g.relation FROM edges g JOIN entities e ON e.id = g.dst WHERE g.src=?"
        params: list[Any] = [id_]
        if relation:
            sql += " AND g.relation=?"; params.append(relation)
        return [(r["relation"], self._row(r)) for r in self._db.execute(sql, params)]

    def walk(self, id_: str, relation: str, depth: int = 4) -> list[Entity]:
        """Follow one relation upward: pod -> workload -> namespace -> context."""
        out: list[Entity] = []
        current = id_
        for _ in range(depth):
            nxt = self.neighbours(current, relation) or self.neighbours(current, "in")
            if not nxt:
                break
            _, ent = nxt[0]
            out.append(ent)
            current = ent.id
        return out

    def stale(self, older_than_s: float, *, kinds: tuple[str, ...] = ()) -> list[Entity]:
        sql = "SELECT * FROM entities WHERE seen_at < ?"
        params: list[Any] = [time.time() - older_than_s]
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"; params += list(kinds)
        return [self._row(r) for r in self._db.execute(sql, params)]

    def count(self) -> int:
        return int(self._db.execute("SELECT count(*) FROM entities").fetchone()[0])

    def close(self) -> None:
        self._db.close()


def get_entity_store(settings: Any) -> EntityStore:
    return EntityStore(Path(settings.home) / "entities.db")


__all__ = ["Entity", "EntityStore", "entity_id", "get_entity_store"]
