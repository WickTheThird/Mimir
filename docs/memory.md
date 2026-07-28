# Memory and knowledge

Implements ADR-001 section 11.

## Layers

| Layer | Directory | What belongs there |
| --- | --- | --- |
| M1 stable | `knowledge/stable/` | service ownership, repository mappings, namespace and SDM naming conventions, environment topology |
| M2 runbooks | `knowledge/runbooks/` | ordered, reviewed procedures |
| M3 history | `knowledge/history/` | incidents and past investigations |
| imports | `knowledge/imports/` | untrusted material from Claude, Codex, ChatGPT |

## Trust order

Retrieval ranks by the ADR 11.4 ladder, then freshness, then confidence:

1. live command output
2. current repository and configuration
3. current deployment state
4. a recently verified runbook
5. a verified historical incident
6. curated imported memory
7. general model knowledge

Live evidence beats a note that says otherwise. That ordering is encoded in
`TRUST_ORDER` and applied by the retriever and by synthesis.

## Freshness

Computed from `last_verified` and `expires_after` against
`knowledge.stale_after_days` (default 180).

Stale documents are returned and labelled stale, not hidden. Hiding them means
the operator never learns the note exists and never fixes it, which is how a
stale runbook survives for years. ADR R2's mitigation is visible metadata.

## Conflicts

When two notes disagree, or one supersedes another, both are surfaced with a
conflict marker. Nothing silently wins. Resolving the conflict is the operator's
call, and leaving it visible is better than picking wrong.

## Promotion

```
session finding -> proposal -> review -> approval -> stored
```

Nothing reaches `stable/` or `runbooks/` without an explicit approval flag, and
an unverified note may only land in `history/investigations`. That is ADR 11.6
and NG4, and it is enforced in `MemoryPromoter`, not by convention.

```bash
mimir memory search "pod restart"
mimir memory reindex
```

## Importing prior work

```bash
mimir memory import ~/Downloads/conversations.json
mimir memory import ~/.claude/projects/foo --tool claude
```

Imports are summarised rather than dumped, secrets are stripped, the original
date is recorded, and everything lands in `imports/` as unverified with low
confidence. Raw chat is source material, never trusted memory.

## Search

SQLite FTS5 for keyword search, plus optional embeddings for semantic search.
Embeddings come from whatever the `embed` profile points at; when no runtime is
reachable, retrieval degrades to a deterministic hashing embedder and keyword
search still works. `mimir doctor` reports which is in use.

## Metadata

```yaml
---
title: Pod restart investigation
category: runbooks/kubernetes
service: payments-api
environment: production
created_at: 2026-07-01
last_verified: 2026-07-20
source: incident INC-1234
confidence: high
owner: platform
supersedes: runbooks/kubernetes/old-note
expires_after: 180d
tags: [kubernetes, restarts, oom]
---
```

Missing frontmatter is tolerated: the title is inferred from the H1 and the
category from the directory. Get value from a note first and add metadata when
it earns it.
