"""Which language server handles which language, and whether it is installed."""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path


def _resolve(binary: str) -> str | None:
    """Find a server binary on PATH, or in the interpreter's own bin directory."""
    found = shutil.which(binary)
    if found:
        return found
    candidate = Path(sys.executable).parent / binary
    return str(candidate) if candidate.is_file() else None


@dataclass(frozen=True)
class ServerSpec:
    language: str
    argv: tuple[str, ...]
    extensions: tuple[str, ...]
    install_hint: str
    language_id: str = ""
    """The LSP languageId. Defaults to ``language`` when they agree."""

    settings: dict[str, object] = field(default_factory=dict)

    @property
    def binary(self) -> str:
        return self.argv[0]

    @property
    def installed(self) -> bool:
        return _resolve(self.binary) is not None

    @property
    def command(self) -> tuple[str, ...]:
        """argv with the binary resolved to an absolute path where possible."""
        resolved = _resolve(self.binary)
        return (resolved or self.binary, *self.argv[1:])

    @property
    def id(self) -> str:
        return self.language_id or self.language


SERVERS: tuple[ServerSpec, ...] = (
    ServerSpec(
        language="python",
        argv=("pyright-langserver", "--stdio"),
        extensions=(".py", ".pyi"),
        install_hint="npm install -g pyright",
    ),
    ServerSpec(
        language="python",
        argv=("pylsp",),
        extensions=(".py", ".pyi"),
        install_hint="uv pip install python-lsp-server",
    ),
    ServerSpec(
        language="typescript",
        argv=("typescript-language-server", "--stdio"),
        extensions=(".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"),
        install_hint="npm install -g typescript-language-server typescript",
        language_id="typescript",
    ),
    ServerSpec(
        language="go",
        argv=("gopls",),
        extensions=(".go",),
        install_hint="go install golang.org/x/tools/gopls@latest",
    ),
    ServerSpec(
        language="rust",
        argv=("rust-analyzer",),
        extensions=(".rs",),
        install_hint="rustup component add rust-analyzer",
    ),
    ServerSpec(
        language="c",
        argv=("clangd",),
        extensions=(".c", ".h", ".cc", ".cpp", ".hpp"),
        install_hint="xcode-select --install",
        language_id="cpp",
    ),
)


def server_for(path: str) -> ServerSpec | None:
    """The first installed server claiming this file extension."""
    suffix = "." + path.rsplit(".", 1)[-1] if "." in path else ""
    if not suffix:
        return None
    candidates = [s for s in SERVERS if suffix in s.extensions]
    for spec in candidates:
        if spec.installed:
            return spec
    return candidates[0] if candidates else None


def available_servers() -> list[ServerSpec]:
    return [s for s in SERVERS if s.installed]
