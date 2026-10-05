"""A minimal, synchronous LSP client over stdio."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mimir.logging import get_logger
from mimir.lsp.servers import ServerSpec, server_for

log = get_logger(__name__)

DEFAULT_TIMEOUT = 20.0
INDEX_GRACE = 2.0
"""Servers answer before indexing finishes, often with nothing."""


class LspError(RuntimeError):
    """The server was reachable but the request failed."""


class LspUnavailable(LspError):
    """No server is installed for this language."""


@dataclass(frozen=True)
class LspLocation:
    path: str
    line: int
    """1-based, matching how the rest of MIMIR cites source."""

    column: int
    end_line: int = 0
    preview: str = ""

    def render(self) -> str:
        span = f"{self.line}" if self.end_line in (0, self.line) else f"{self.line}-{self.end_line}"
        return f"{self.path}:{span}"


@dataclass(frozen=True)
class LspSymbol:
    name: str
    kind: str
    location: LspLocation
    container: str = ""


SYMBOL_KINDS = {
    1: "file", 2: "module", 3: "namespace", 4: "package", 5: "class",
    6: "method", 7: "property", 8: "field", 9: "constructor", 10: "enum",
    11: "interface", 12: "function", 13: "variable", 14: "constant",
    15: "string", 16: "number", 17: "boolean", 18: "array", 19: "object",
    20: "key", 21: "null", 22: "enum member", 23: "struct", 24: "event",
    25: "operator", 26: "type parameter",
}

_SEVERITY = {1: "error", 2: "warning", 3: "information", 4: "hint"}


@dataclass
class _Pending:
    event: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Any = None


class LspClient:
    """One server process for one project root."""

    def __init__(self, spec: ServerSpec, root: Path, *, timeout: float = DEFAULT_TIMEOUT):
        self.spec = spec
        self.root = Path(root).resolve()
        self.timeout = timeout
        self._proc: subprocess.Popen[bytes] | None = None
        self._next_id = 1
        self._pending: dict[int, _Pending] = {}
        self._diagnostics: dict[str, list[dict[str, Any]]] = {}
        self._opened: set[str] = set()
        self._lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._alive = False

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        if self._alive:
            return
        if not self.spec.installed:
            raise LspUnavailable(
                f"no language server for {self.spec.language}: {self.spec.binary} is not "
                f"on PATH. Install with: {self.spec.install_hint}"
            )
        try:
            self._proc = subprocess.Popen(
                list(self.spec.command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=str(self.root),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except OSError as exc:
            raise LspUnavailable(f"could not start {self.spec.binary}: {exc}") from exc

        self._alive = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

        self._request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": self.root.as_uri(),
                "workspaceFolders": [
                    {"uri": self.root.as_uri(), "name": self.root.name}
                ],
                "capabilities": {
                    "textDocument": {
                        "definition": {"linkSupport": True},
                        "references": {},
                        "hover": {"contentFormat": ["plaintext", "markdown"]},
                        "documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
                        "publishDiagnostics": {},
                    },
                    "workspace": {"workspaceFolders": True},
                },
                "initializationOptions": self.spec.settings or {},
            },
            timeout=max(self.timeout, 30.0),
        )
        self._notify("initialized", {})
        log.info("lsp_started", server=self.spec.binary, root=str(self.root))

    def close(self) -> None:
        if not self._alive or self._proc is None:
            return
        self._alive = False
        try:
            self._notify("exit", {})
            self._proc.terminate()
            self._proc.wait(timeout=3)
        except (OSError, subprocess.SubprocessError):
            with contextlib.suppress(OSError):
                self._proc.kill()
        self._proc = None

    # -- transport --------------------------------------------------------

    def _send(self, payload: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise LspError("language server is not running")
        body = json.dumps(payload).encode()
        header = f"Content-Length: {len(body)}\r\n\r\n".encode()
        with self._lock:
            self._proc.stdin.write(header + body)
            self._proc.stdin.flush()

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _request(
        self, method: str, params: dict[str, Any], *, timeout: float | None = None
    ) -> Any:
        request_id = self._next_id
        self._next_id += 1
        pending = _Pending()
        self._pending[request_id] = pending
        self._send(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        )
        if not pending.event.wait(timeout or self.timeout):
            self._pending.pop(request_id, None)
            raise LspError(
                f"{self.spec.binary} did not answer {method} within "
                f"{timeout or self.timeout:.0f}s"
            )
        self._pending.pop(request_id, None)
        if pending.error is not None:
            raise LspError(f"{method} failed: {pending.error}")
        return pending.result

    def _read_loop(self) -> None:
        """Frame LSP messages off stdout and dispatch them."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        stream = proc.stdout
        try:
            while self._alive:
                length = 0
                while True:
                    line = stream.readline()
                    if not line:
                        return
                    if line in (b"\r\n", b"\n"):
                        break
                    if line.lower().startswith(b"content-length:"):
                        length = int(line.split(b":", 1)[1].strip())
                if length <= 0:
                    continue
                raw = stream.read(length)
                if not raw:
                    return
                message = json.loads(raw)
                self._dispatch(message)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            log.info("lsp_reader_stopped", server=self.spec.binary, error=str(exc))
        finally:
            self._alive = False
            for pending in self._pending.values():
                pending.error = "server closed the connection"
                pending.event.set()

    def _dispatch(self, message: dict[str, Any]) -> None:
        if "id" in message and ("result" in message or "error" in message):
            pending = self._pending.get(message["id"])
            if pending is not None:
                pending.result = message.get("result")
                pending.error = message.get("error")
                pending.event.set()
            return
        if message.get("method") == "textDocument/publishDiagnostics":
            params = message.get("params") or {}
            self._diagnostics[params.get("uri", "")] = params.get("diagnostics") or []

    # -- documents --------------------------------------------------------

    def open(self, path: Path) -> str:
        """Open a document, returning its URI. Idempotent."""
        path = Path(path).resolve()
        uri = path.as_uri()
        if uri in self._opened:
            return uri
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise LspError(f"cannot read {path}: {exc}") from exc
        self._notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": self.spec.id,
                    "version": 1,
                    "text": text,
                }
            },
        )
        self._opened.add(uri)
        time.sleep(INDEX_GRACE)
        return uri

    # -- queries ----------------------------------------------------------

    def definition(self, path: Path, line: int, column: int) -> list[LspLocation]:
        uri = self.open(path)
        result = self._request(
            "textDocument/definition",
            {
                "textDocument": {"uri": uri},
                "position": {"line": max(0, line - 1), "character": max(0, column)},
            },
        )
        return _locations(result)

    def references(
        self, path: Path, line: int, column: int, *, include_declaration: bool = True
    ) -> list[LspLocation]:
        uri = self.open(path)
        result = self._request(
            "textDocument/references",
            {
                "textDocument": {"uri": uri},
                "position": {"line": max(0, line - 1), "character": max(0, column)},
                "context": {"includeDeclaration": include_declaration},
            },
        )
        return _locations(result)

    def hover(self, path: Path, line: int, column: int) -> str:
        uri = self.open(path)
        result = self._request(
            "textDocument/hover",
            {
                "textDocument": {"uri": uri},
                "position": {"line": max(0, line - 1), "character": max(0, column)},
            },
        )
        if not result:
            return ""
        contents = result.get("contents")
        if isinstance(contents, dict):
            return str(contents.get("value", "")).strip()
        if isinstance(contents, list):
            parts = [
                c.get("value", "") if isinstance(c, dict) else str(c) for c in contents
            ]
            return "\n".join(p for p in parts if p).strip()
        return str(contents or "").strip()

    def document_symbols(self, path: Path) -> list[LspSymbol]:
        uri = self.open(path)
        result = self._request("textDocument/documentSymbol", {"textDocument": {"uri": uri}})
        return _symbols(result or [], str(path))

    def diagnostics(self, path: Path) -> list[dict[str, Any]]:
        """Diagnostics the server published for this file."""
        uri = self.open(path)
        deadline = time.time() + 3.0
        while uri not in self._diagnostics and time.time() < deadline:
            time.sleep(0.1)
        out = []
        for item in self._diagnostics.get(uri, []):
            start = (item.get("range") or {}).get("start") or {}
            out.append(
                {
                    "severity": _SEVERITY.get(item.get("severity", 1), "error"),
                    "line": int(start.get("line", 0)) + 1,
                    "message": str(item.get("message", "")).strip(),
                    "source": item.get("source", ""),
                }
            )
        return out


def _uri_to_path(uri: str) -> str:
    if uri.startswith("file://"):
        from urllib.parse import unquote, urlparse

        return unquote(urlparse(uri).path)
    return uri


def _locations(result: Any) -> list[LspLocation]:
    if not result:
        return []
    items = result if isinstance(result, list) else [result]
    out: list[LspLocation] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        # LocationLink (linkSupport) and Location have different shapes.
        uri = item.get("uri") or item.get("targetUri") or ""
        rng = item.get("range") or item.get("targetSelectionRange") or item.get(
            "targetRange"
        ) or {}
        start = rng.get("start") or {}
        end = rng.get("end") or {}
        if not uri:
            continue
        out.append(
            LspLocation(
                path=_uri_to_path(uri),
                line=int(start.get("line", 0)) + 1,
                column=int(start.get("character", 0)),
                end_line=int(end.get("line", 0)) + 1,
            )
        )
    return out


def _symbols(result: list[Any], path: str, container: str = "") -> list[LspSymbol]:
    out: list[LspSymbol] = []
    for item in result:
        if not isinstance(item, dict):
            continue
        rng = item.get("selectionRange") or item.get("range") or (
            item.get("location") or {}
        ).get("range") or {}
        start = rng.get("start") or {}
        location = LspLocation(
            path=_uri_to_path((item.get("location") or {}).get("uri", "")) or path,
            line=int(start.get("line", 0)) + 1,
            column=int(start.get("character", 0)),
        )
        out.append(
            LspSymbol(
                name=str(item.get("name", "")),
                kind=SYMBOL_KINDS.get(int(item.get("kind", 0)), "symbol"),
                location=location,
                container=container or str(item.get("containerName", "")),
            )
        )
        for child in item.get("children") or []:
            out.extend(_symbols([child], path, container=str(item.get("name", ""))))
    return out


_CLIENTS: dict[tuple[str, str], LspClient] = {}


def get_client(path: Path, root: Path) -> LspClient:
    """Cached client for the server that handles ``path`` rooted at ``root``."""
    spec = server_for(str(path))
    if spec is None:
        raise LspUnavailable(f"no language server is configured for {Path(path).name}")
    key = (spec.binary, str(Path(root).resolve()))
    client = _CLIENTS.get(key)
    if client is None or not client._alive:
        client = LspClient(spec, root)
        client.start()
        _CLIENTS[key] = client
    return client


def shutdown_all() -> None:
    for client in list(_CLIENTS.values()):
        client.close()
    _CLIENTS.clear()
