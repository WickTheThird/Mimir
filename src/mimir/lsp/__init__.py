"""Language Server Protocol client.

A language server knows where a symbol is defined, every place it is used, and
what type an expression actually has. MIMIR currently answers those questions
with ripgrep plus a regex that guesses whether a line looks like a definition,
which is approximately right and occasionally confidently wrong.

This is the ADR-003 section 4.7 rule applied to the most common operation in
repository investigation: where an exact engine exists, use it rather than
asking a model, or a heuristic, to imitate one.
"""

from mimir.lsp.client import (
    LspClient,
    LspError,
    LspLocation,
    LspSymbol,
    LspUnavailable,
)
from mimir.lsp.servers import SERVERS, ServerSpec, available_servers, server_for

__all__ = [
    "SERVERS",
    "LspClient",
    "LspError",
    "LspLocation",
    "LspSymbol",
    "LspUnavailable",
    "ServerSpec",
    "available_servers",
    "server_for",
]
