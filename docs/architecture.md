# Architecture

Implements ADR-001 section 6. Read [ADR-001.md](ADR-001.md) for the reasoning;
this describes what was actually built.

## Topology

```
  CLI (mimir)            Web UI (React)          Warp
       |                       |                   |
       |                       v                   v
       |                  /api routes        /v1 facade
       |                  (loopback only)    (authenticated)
       |                       |                   |
       +-----------------------+-------------------+
                               |
                    InvestigationRunner
                               |
                    LangGraph orchestrator
                    /          |          \
          Council of    Typed helper     Knowledge,
          specialists   tools            skills, memory
                    \          |          /
                     Model runtime (local)
```

One runner backs all three interfaces, so they cannot drift apart in behaviour.

## Request lifecycle

1. **resolve_context** fills in cluster, namespace, and repositories from the
   shell and configuration. It never guesses a namespace; an unresolved one
   becomes a question rather than a default.
2. **recall_memory** retrieves curated notes, which enter as evidence carrying
   their real source type and freshness so the ADR 11.4 trust ladder ranks them.
3. **coordinate** classifies the request and produces a plan naming specialists,
   objectives, and skills. Failure here falls back to a keyword-routed single
   step rather than failing the investigation.
4. **select_skills** performs the level-2 skill load for the chosen skills only.
5. **dispatch** fans out one LangGraph `Send` per planned step, so specialists
   run concurrently with independent contexts.
6. **gather** folds the parallel channels back into the durable state.
7. **verify** attacks the conclusions. Contradictions lower confidence and are
   recorded rather than smoothed away.
8. **safety_review** reads proposed commands for intent mismatch and blast
   radius. Skipped when nothing was proposed.
9. **synthesise** produces the final answer with observed, inferred, and
   unverified separated structurally.
10. **curate_memory** proposes, never promotes.

## State

`InvestigationState` is the durable record and matches the conceptual state in
ADR 12 field for field. `GraphState` wraps it in LangGraph channels: `session`
has a single writer, while `reports`, `evidence`, `proposed_commands`, and
`memory_proposals` are append-only channels with reducers so concurrent
specialists do not clobber each other. Merging happens in exactly one place,
`merge_into_session`.

## Why approvals are brokered rather than graph interrupts

ADR 6.2 C2 lists human-in-the-loop interrupts as a reason to choose LangGraph.
The implementation uses an `ApprovalBroker` instead, for one reason: an approval
arises inside a tool call, several frames below any node boundary. A graph-level
interrupt would require every helper to unwind to a node before a human could be
asked, which would lose the tool's context and make partial work hard to resume.

The broker is an async rendezvous any interface can resolve. Durability still
comes from LangGraph checkpointing, so a run survives a restart either way.

## Concurrency

Read-only commands run in parallel through `CommandExecutor.run_many`. Anything
above the auto-execute ceiling is serialised deliberately, so approval prompts
never interleave and an operator is never asked to judge two commands at once.

## Extension points

| To add | Touch |
| --- | --- |
| A tool | `src/mimir/tools/`, decorate with `@tool` |
| A specialist | `SpecialistName`, prompt, capability and risk tables |
| A model runtime | `mimir/llm/openai_compat.py` `build_model` |
| A skill | a directory under `knowledge/skills/` |
| A search backend | `mimir/tools/web.py` `SearchBackend` |
| A risk rule | `mimir/safety/risk.py` tables |
