"""A map of a repository, built once and refreshed on change.

ADR-002 gap: a coding assistant that starts every task with a text search
is reading the codebase for the first time on every task. This is one
indexed pass per repository: where symbols are defined, what imports what,
which tests exercise which modules, and which files change most. The LSP
stays for precision inside a file; this is the altitude it lacks.

Derived cache under ``$MIMIR_HOME/cache``. Delete it and it rebuilds. Python
is parsed with ``ast``; other languages get a regex over definitions, which
is coarse and says so in the ``kind`` column.
"""

from __future__ import annotations

import ast
import hashlib
import re
import sqlite3
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mimir.logging import get_logger

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY, language TEXT, lines INTEGER, hash TEXT, is_test INTEGER, indexed_at REAL
);
CREATE TABLE IF NOT EXISTS symbols (
    name TEXT, kind TEXT, path TEXT, line INTEGER, parent TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS symbols_path ON symbols(path);
CREATE TABLE IF NOT EXISTS imports (
    path TEXT, module TEXT, resolved TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS imports_resolved ON imports(resolved);
CREATE TABLE IF NOT EXISTS churn (path TEXT PRIMARY KEY, commits INTEGER, last_change REAL);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

_LANG = {".py": "python", ".go": "go", ".ts": "typescript", ".tsx": "typescript",
         ".js": "javascript", ".rs": "rust", ".java": "java", ".kt": "kotlin", ".rb": "ruby"}

_DEF_RE = {
    "go": re.compile(r"^func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)|^type\s+([A-Za-z_]\w*)", re.M),
    "typescript": re.compile(r"^(?:export\s+)?(?:async\s+)?(?:function|class|interface|type|const|let)\s+([A-Za-z_$][\w$]*)", re.M),
    "javascript": re.compile(r"^(?:export\s+)?(?:async\s+)?(?:function|class|const|let)\s+([A-Za-z_$][\w$]*)", re.M),
    "rust": re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:fn|struct|enum|trait|type)\s+([A-Za-z_]\w*)", re.M),
    "java": re.compile(r"^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:class|interface|enum)\s+([A-Za-z_]\w*)", re.M),
    "kotlin": re.compile(r"^\s*(?:fun|class|object|interface)\s+([A-Za-z_]\w*)", re.M),
    "ruby": re.compile(r"^\s*(?:def|class|module)\s+([A-Za-z_]\w*)", re.M),
}
_IMPORT_RE = {
    "go": re.compile(r'^\s*"([^"]+)"\s*$', re.M),
    "typescript": re.compile(r"""from\s+['"]([^'"]+)['"]|require\(['"]([^'"]+)['"]\)""", re.M),
    "javascript": re.compile(r"""from\s+['"]([^'"]+)['"]|require\(['"]([^'"]+)['"]\)""", re.M),
    "rust": re.compile(r"^\s*use\s+([\w:]+)", re.M),
}
_TEST_PATH = re.compile(r"(^|/)(tests?|spec|__tests__)(/|$)|(_test|\.test|\.spec|test_)[\w.]*$")


@dataclass(slots=True)
class Symbol:
    name: str
    kind: str
    path: str
    line: int
    parent: str = ""


def _python(text: str, path: str) -> tuple[list[Symbol], list[str]]:
    symbols: list[Symbol] = []
    imports: list[str] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return symbols, imports
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(Symbol(node.name, "function", path, node.lineno))
        elif isinstance(node, ast.ClassDef):
            symbols.append(Symbol(node.name, "class", path, node.lineno))
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append(Symbol(sub.name, "method", path, sub.lineno, node.name))
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id.isupper():
                    symbols.append(Symbol(t.id, "constant", path, node.lineno))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(("." * node.level) + node.module)
    return symbols, imports


def _generic(text: str, path: str, language: str) -> tuple[list[Symbol], list[str]]:
    symbols: list[Symbol] = []
    pattern = _DEF_RE.get(language)
    if pattern:
        for m in pattern.finditer(text):
            name = next((g for g in m.groups() if g), None)
            if name:
                symbols.append(Symbol(name, "definition", path, text.count("\n", 0, m.start()) + 1))
    imports: list[str] = []
    ipat = _IMPORT_RE.get(language)
    if ipat:
        for m in ipat.finditer(text):
            mod = next((g for g in m.groups() if g), None)
            if mod:
                imports.append(mod)
    return symbols, imports


def _resolve_python(module: str, from_path: str, known: set[str]) -> str:
    """Map an import to a repository path when the repository contains it."""
    if module.startswith("."):
        base = Path(from_path).parent
        depth = len(module) - len(module.lstrip("."))
        for _ in range(depth - 1):
            base = base.parent
        tail = module.lstrip(".").replace(".", "/")
        candidates = [f"{base}/{tail}.py", f"{base}/{tail}/__init__.py"] if tail else [f"{base}/__init__.py"]
    else:
        tail = module.replace(".", "/")
        candidates = [f"{tail}.py", f"{tail}/__init__.py", f"src/{tail}.py", f"src/{tail}/__init__.py"]
    for c in candidates:
        c = c.lstrip("./")
        if c in known:
            return c
    # a package prefix: "mimir.verify.claims" resolves to the deepest known file
    parts = module.lstrip(".").split(".")
    while parts:
        for c in (f"{'/'.join(parts)}.py", f"src/{'/'.join(parts)}.py",
                  f"{'/'.join(parts)}/__init__.py", f"src/{'/'.join(parts)}/__init__.py"):
            if c in known:
                return c
        parts.pop()
    return ""


class RepoMap:
    def __init__(self, cache_dir: Path, root: Path) -> None:
        self.root = Path(root).resolve()
        digest = hashlib.sha1(str(self.root).encode()).hexdigest()[:12]
        self.path = Path(cache_dir) / f"repomap-{digest}.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)

    # -- building ---------------------------------------------------------

    def build(self, *, ignore_globs: tuple[str, ...] = (), max_file_bytes: int = 2_000_000,
              churn_commits: int = 500) -> dict[str, int]:
        """Index the tree. Unchanged files (by hash) are skipped."""
        from mimir.tools.repo import _walk_files

        started = time.time()
        seen: set[str] = set()
        known_before = {r["path"]: r["hash"] for r in self._db.execute("SELECT path, hash FROM files")}
        texts: dict[str, tuple[str, str]] = {}
        for path, rel in _walk_files(self.root, ignore_globs=ignore_globs, include_globs=()):
            lang = _LANG.get(path.suffix.lower())
            if not lang:
                continue
            seen.add(rel)
            try:
                if path.stat().st_size > max_file_bytes:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            digest = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
            if known_before.get(rel) == digest:
                continue
            texts[rel] = (lang, text)
        # drop files that vanished
        for gone in set(known_before) - seen:
            self._forget(gone)
        known_paths = seen | set(known_before)
        indexed = 0
        for rel, (lang, text) in texts.items():
            self._forget(rel)
            symbols, imports = _python(text, rel) if lang == "python" else _generic(text, rel, lang)
            self._db.execute(
                "INSERT INTO files(path, language, lines, hash, is_test, indexed_at) VALUES(?,?,?,?,?,?)",
                (rel, lang, text.count("\n") + 1,
                 hashlib.sha1(text.encode("utf-8", "replace")).hexdigest(),
                 int(bool(_TEST_PATH.search(rel))), time.time()),
            )
            self._db.executemany(
                "INSERT INTO symbols(name, kind, path, line, parent) VALUES(?,?,?,?,?)",
                [(s.name, s.kind, s.path, s.line, s.parent) for s in symbols],
            )
            self._db.executemany(
                "INSERT INTO imports(path, module, resolved) VALUES(?,?,?)",
                [(rel, m, _resolve_python(m, rel, known_paths) if lang == "python" else "")
                 for m in imports],
            )
            indexed += 1
        self._churn(churn_commits)
        self._db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('built_at', ?)", (str(time.time()),))
        self._db.commit()
        log.info("repomap_built", root=str(self.root), files=len(seen), indexed=indexed,
                 seconds=round(time.time() - started, 2))
        return {"files": len(seen), "indexed": indexed, "skipped": len(seen) - indexed}

    def _forget(self, rel: str) -> None:
        for table in ("files", "symbols", "imports"):
            self._db.execute(f"DELETE FROM {table} WHERE path=?", (rel,))

    def _churn(self, limit: int) -> None:
        try:
            out = subprocess.run(
                ["git", "-C", str(self.root), "log", f"-{limit}", "--name-only", "--format=%ct"],
                capture_output=True, text=True, timeout=30, check=False,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return
        counts: dict[str, int] = defaultdict(int)
        last: dict[str, float] = {}
        stamp = 0.0
        for line in out.splitlines():
            if line.strip().isdigit():
                stamp = float(line.strip())
            elif line.strip():
                counts[line.strip()] += 1
                last.setdefault(line.strip(), stamp)
        self._db.execute("DELETE FROM churn")
        self._db.executemany(
            "INSERT INTO churn(path, commits, last_change) VALUES(?,?,?)",
            [(p, n, last.get(p, 0.0)) for p, n in counts.items()],
        )

    # -- querying ---------------------------------------------------------

    def symbols(self, name: str, *, limit: int = 20) -> list[Symbol]:
        rows = self._db.execute(
            "SELECT * FROM symbols WHERE name=? OR lower(name) LIKE ? ORDER BY (name=?) DESC, path LIMIT ?",
            (name, f"%{name.lower()}%", name, limit),
        )
        return [Symbol(r["name"], r["kind"], r["path"], r["line"], r["parent"]) for r in rows]

    def importers_of(self, path: str) -> list[str]:
        return [r["path"] for r in self._db.execute(
            "SELECT DISTINCT path FROM imports WHERE resolved=?", (path,))]

    def tests_for(self, paths: list[str], *, depth: int = 2) -> list[str]:
        """Test files that import the changed modules, directly or through
        one intermediary. What run_worktree_tests should run first."""
        frontier = set(paths)
        seen: set[str] = set(paths)
        tests: set[str] = set()
        for _ in range(depth):
            nxt: set[str] = set()
            for p in frontier:
                for imp in self.importers_of(p):
                    if imp in seen:
                        continue
                    seen.add(imp)
                    if self._is_test(imp):
                        tests.add(imp)
                    else:
                        nxt.add(imp)
            frontier = nxt
        # a test file named after the module counts even without an import edge
        for p in paths:
            stem = Path(p).stem
            for r in self._db.execute("SELECT path FROM files WHERE is_test=1 AND path LIKE ?", (f"%{stem}%",)):
                tests.add(r["path"])
        return sorted(tests)

    def _is_test(self, path: str) -> bool:
        r = self._db.execute("SELECT is_test FROM files WHERE path=?", (path,)).fetchone()
        return bool(r and r["is_test"])

    def hot(self, n: int = 10) -> list[tuple[str, int]]:
        return [(r["path"], r["commits"]) for r in self._db.execute(
            "SELECT path, commits FROM churn ORDER BY commits DESC LIMIT ?", (n,))]

    def outline(self, path: str) -> list[Symbol]:
        return [Symbol(r["name"], r["kind"], r["path"], r["line"], r["parent"]) for r in
                self._db.execute("SELECT * FROM symbols WHERE path=? ORDER BY line", (path,))]

    def summary(self) -> dict[str, Any]:
        f = self._db.execute("SELECT count(*) c, sum(is_test) t FROM files").fetchone()
        s = self._db.execute("SELECT count(*) FROM symbols").fetchone()[0]
        built = self._db.execute("SELECT value FROM meta WHERE key='built_at'").fetchone()
        return {"files": f["c"], "tests": f["t"] or 0, "symbols": s,
                "built_at": float(built[0]) if built else None, "root": str(self.root)}

    def close(self) -> None:
        self._db.close()


def get_repo_map(settings: Any, root: Path) -> RepoMap:
    return RepoMap(Path(settings.home) / "cache", root)


__all__ = ["RepoMap", "Symbol", "get_repo_map"]
