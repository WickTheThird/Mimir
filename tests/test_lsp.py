"""Language server tests.

The theme: an exact engine is only worth having if its failures are
distinguishable from its answers. "No server installed" must never read as
"this symbol does not exist".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mimir.lsp.client import LspLocation, LspSymbol, LspUnavailable, _locations, _symbols
from mimir.lsp.servers import SERVERS, available_servers, server_for


class TestServerSelection:
    def test_extension_maps_to_a_server(self):
        assert server_for("main.go").binary == "gopls"
        assert server_for("lib.rs").binary == "rust-analyzer"
        assert server_for("a/b/c.py").language == "python"

    def test_unknown_extension_has_no_server(self):
        assert server_for("notes.txt") is None
        assert server_for("Makefile") is None

    def test_python_prefers_pyright_when_both_are_present(self):
        """Ordering within a language is preference order, not arbitrary."""
        python = [s for s in SERVERS if s.language == "python"]
        assert python[0].binary == "pyright-langserver"

    def test_a_server_in_the_venv_counts_as_installed(self):
        """MIMIR runs from a virtualenv, and `uv pip install python-lsp-server`
        puts pylsp beside the interpreter rather than on PATH. Checking only
        PATH reported a present server as missing."""
        assert any(s.language == "python" for s in available_servers())


class TestUnavailableIsNotEmpty:
    def test_missing_server_raises_rather_than_returning_nothing(self):
        from mimir.lsp.client import LspClient
        from mimir.lsp.servers import ServerSpec

        spec = ServerSpec(
            language="cobol", argv=("cobol-lsp-that-does-not-exist",),
            extensions=(".cbl",), install_hint="it does not exist",
        )
        with pytest.raises(LspUnavailable) as exc:
            LspClient(spec, Path.cwd()).start()
        assert "not on PATH" in str(exc.value)
        assert "it does not exist" in str(exc.value), "the fix must be in the error"


class TestResponseParsing:
    def test_location_and_locationlink_both_parse(self):
        """linkSupport changes the wire shape; both must resolve."""
        plain = _locations([{
            "uri": "file:///repo/a.py",
            "range": {"start": {"line": 9, "character": 4}, "end": {"line": 9, "character": 8}},
        }])
        link = _locations([{
            "targetUri": "file:///repo/a.py",
            "targetSelectionRange": {
                "start": {"line": 9, "character": 4}, "end": {"line": 9, "character": 8}
            },
        }])
        assert plain[0].path == "/repo/a.py" and link[0].path == "/repo/a.py"
        assert plain[0].line == link[0].line == 10, "LSP is 0-based, MIMIR cites 1-based"

    def test_a_single_result_is_accepted_as_well_as_a_list(self):
        assert len(_locations({"uri": "file:///a.py", "range":
                               {"start": {"line": 0, "character": 0}}})) == 1

    def test_empty_and_null_results_are_empty_not_errors(self):
        assert _locations(None) == []
        assert _locations([]) == []

    def test_nested_symbols_keep_their_container(self):
        found = _symbols([{
            "name": "Thing", "kind": 5,
            "range": {"start": {"line": 0, "character": 0}},
            "children": [{
                "name": "method", "kind": 6,
                "range": {"start": {"line": 4, "character": 4}},
            }],
        }], "a.py")
        assert [s.name for s in found] == ["Thing", "method"]
        assert found[1].container == "Thing"
        assert found[1].kind == "method"


class TestRendering:
    def test_single_and_multi_line_spans(self):
        assert LspLocation("a.py", 10, 0).render() == "a.py:10"
        assert LspLocation("a.py", 10, 0, end_line=12).render() == "a.py:10-12"

    def test_symbol_carries_a_citable_location(self):
        s = LspSymbol("f", "function", LspLocation("a.py", 3, 0))
        assert s.location.render() == "a.py:3"


class TestSupersession:
    """Never offer an approximate tool when an exact one is registered."""

    def test_lsp_hides_the_ripgrep_equivalents(self):
        """Registered, but not offered. The registry says what exists;
        selection says what a specialist is shown."""
        from mimir.eval.harness import EvalHarness

        registry = EvalHarness().offline_registry()
        assert "find_symbol" in registry.names(), "still registered"

        names = {s.name for s in registry.select()}
        assert "lsp_definition" in names and "lsp_references" in names
        assert "find_symbol" not in names, (
            "find_symbol is ripgrep plus a regex guessing at definitions; "
            "offering it beside lsp_definition costs schema tokens on every "
            "call and invites the model to pick the worse one"
        )
        assert "find_references" not in names

    def test_a_superseded_tool_is_still_callable_directly(self):
        """Hidden from selection, not removed. A caller that knows what it
        wants can still reach it."""
        from mimir.tools.base import load_all_tools

        assert load_all_tools().get("find_symbol") is not None

    def test_supersession_only_applies_when_the_replacement_exists(self):
        from mimir.tools.base import ToolRegistry, load_all_tools

        full = load_all_tools()
        lonely = ToolRegistry()
        lonely.register(full.get("find_symbol"))
        assert [s.name for s in lonely.select()] == ["find_symbol"], (
            "with no lsp_definition registered, find_symbol is the best available"
        )


class TestCodeIsNotRepository:
    def test_investigation_specialists_are_not_offered_mutation_tools(self):
        from mimir.council.specialists import Specialist
        from mimir.eval.harness import EvalHarness
        from mimir.models.specialist import SpecialistName

        registry = EvalHarness().offline_registry()
        offered = {
            s.name
            for s in Specialist(
                SpecialistName.REPOSITORY_EXPLORER, registry=registry
            ).available_tools()
        }
        for name in ("write_worktree_file", "create_task_worktree",
                     "discard_task_worktree"):
            assert name not in offered, f"{name} is code mutation, not investigation"
