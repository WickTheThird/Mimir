---
name: repository-flow-tracing
version: 0.1.0
description: Trace how a request or event moves through a codebase, as ordered evidence.
when_to_use: Someone asks where something enters, what handles it, what is called next, or where state is written.
specialist: repository_explorer
max_risk: R1
author: mimir
updated_at: 2026-07-27
tags: [flow, trace, path, entrypoint, handler, call-path, where-does, how-does]
allowed_tools:
  - list_repositories
  - search_repository
  - find_symbol
  - find_references
  - read_file_range
  - build_import_graph
  - build_flow_evidence
  - locate_tests
inputs:
  - name: entrypoint
    description: A path, symbol, route, topic, or event name to start from.
    required: true
  - name: repo
    description: Repository to trace in.
outputs:
  - name: flow
    description: Ordered hops with file and line citations.
  - name: boundaries
    description: Where the flow leaves this repository.
tests:
  - name: separates-confirmed-from-inferred
    input: Where does an inbound checkout request enter the system?
    assertions:
      - "contains: inferred"
      - "max_risk_at_most: R1"

---

# Repository flow tracing

Answers the ADR 5.3 questions: where does this enter, which handler receives it,
what is called next, where is state persisted, what happens on timeout, which
branch does this flag select, and which service owns the next step.

## Method

1. **Find the entry.** A route table, a consumer registration, a CLI command, a
   cron entry, or a topic subscription. `search_repository` for the route string
   or topic name is usually faster than reading the framework setup.
2. **Use `build_flow_evidence`.** It walks imports from the entrypoint, classifies
   each file into a stage, and reports where timeout, retry, fallback, and
   feature-flag handling appear on the path. Start there rather than reading files
   one at a time.
3. **Confirm each hop.** The import graph proves a file is reachable, not that a
   specific function is called. Use `find_references` to confirm the call.
4. **Read narrow ranges.** Read the function, not the file. A 4000-line file will
   consume the whole context budget for one hop.
5. **Stop at the boundary.** When the flow leaves via an HTTP client, a queue
   publish, or an RPC stub, name the destination service and stop. Say clearly
   that the next hop is in another repository.

## Report shape

An ordered list, each hop with:

- the stage (entrypoint, routing, middleware, domain, outbound, persistence)
- the file and line range
- what it does in one sentence
- what happens there on failure or timeout

Then, separately: branches not taken and the condition that selects them, and any
hop you could not confirm.

## Honesty rules

Say which hops you confirmed by reading code and which you inferred from imports.
An inferred hop is a lead, not a fact. If a call is dynamic (reflection, a
registry, dependency injection), say so; the static path will be incomplete and
pretending otherwise produces a confident wrong map.
