# Language servers

```bash
mimir tools | grep lsp        # what is registered
```

Six tools: `lsp_status`, `lsp_definition`, `lsp_references`, `lsp_hover`,
`lsp_symbols`, `lsp_diagnostics`.

## Why

`find_symbol` and `find_references` are ripgrep plus a regex that decides
whether a matching line *looks like* a definition. That works until a name is
shadowed, re-exported, imported under an alias, defined inside a string, or
simply common. A language server answers the same questions from a parsed,
type-aware model of the project.

This is ADR-003 section 4.7 applied to the most frequent operation in
repository investigation: where an exact engine exists, use it rather than
asking a model - or a heuristic - to imitate one. Section 4.2 follows: a server
answer outranks anything the model believes about the same symbol.

## The chain is deterministic end to end

```
document symbols  ->  exact line of the name      (from the server's parse)
                  ->  column of the name in it    (offset within that line)
                  ->  server query
                  ->  location, signature, diagnostic
```

The column step exists because a server returning `SymbolInformation` reports
the range of the whole definition - column 0, the `def` or `class` keyword -
and a position query there resolves nothing. Locating the name inside a line
the server already identified keeps the answer anchored to the parse rather
than to a text search over the file.

## Unavailable is not empty

`LspUnavailable` is a distinct error from an empty result, and the message
carries the install command. Collapsing the two would let a missing toolchain
read as "this symbol does not exist", which is the confidently-wrong failure
this project exists to prevent.

Servers are resolved on PATH **and** in the running interpreter's `bin`
directory, because MIMIR usually runs from a virtualenv and
`uv pip install python-lsp-server` puts `pylsp` beside the interpreter rather
than on PATH. Checking only PATH reported a present server as missing.

| Language | Server | Install |
| --- | --- | --- |
| python | pyright-langserver, else pylsp | `npm i -g pyright` / `uv pip install python-lsp-server` |
| typescript | typescript-language-server | `npm i -g typescript-language-server typescript` |
| go | gopls | `go install golang.org/x/tools/gopls@latest` |
| rust | rust-analyzer | `rustup component add rust-analyzer` |
| c / c++ | clangd | `xcode-select --install` |

Python prefers pyright because it resolves types the others infer.

## Switchable, and why that matters

```yaml
lsp:
  enabled: true      # false reproduces the pre-LSP tool surface exactly
  timeout_s: 20.0
  index_grace_s: 2.0
```

Adding six tools changes `enabled_tools_hash` from `fc49c169e26e37bb` to
`dfe46ae658393e0c`, and a run with a different tool surface is not comparable
to one without it. The switch exists so an experiment can isolate one change at
a time: with `enabled: false` the fingerprint matches the A4-A6 baseline
byte for byte.

## Cost

A server costs seconds to start and longer to index, so clients are cached per
`(binary, repository root)` for the life of the process. Every request carries
a deadline; a hung server degrades to unavailable rather than wedging an
investigation.
