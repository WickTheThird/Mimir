"""MIMIR capabilities as tools an outer agent invokes deliberately (ADR-001 §16.4)."""

from mimir.mcp.server import build_server, serve

__all__ = ["build_server", "serve"]
