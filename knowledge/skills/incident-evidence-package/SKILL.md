---
name: incident-evidence-package
version: 0.1.0
description: Assemble a compact handoff for a hosted coding agent or a colleague.
when_to_use: An investigation has concluded and the fix needs to be implemented elsewhere.
specialist: synthesis
max_risk: R0
author: mimir
updated_at: 2026-07-27
tags: [handoff, package, export, summary, escalate, delegate, claude, codex]
allowed_tools: []
inputs:
  - name: session
    description: The investigation to package.
    required: true
outputs:
  - name: package
    description: Markdown covering problem, evidence, cause, uncertainty, scope, non-goals.
tests:
  - name: includes-uncertainty-and-non-goals
    input: Package this investigation for Claude to implement the fix.
    assertions:
      - "contains: Uncertainty"
      - "contains: non-goals"
      - "max_risk_at_most: R0"

---

# Incident evidence package

The receiving agent has none of your context and will happily rediscover it at
great expense, or invent it. The package exists so it does neither.

## Sections, in this order

1. **Problem statement.** One paragraph. What is wrong, for whom, since when.
2. **Observed behaviour.** What was actually seen, not what was concluded.
3. **Evidence.** Each item with its source: file and line range, command and exit
   code, or URL and retrieval time. Excerpts, not whole files.
4. **Relevant paths.** The files a fix will touch, with line ranges.
5. **Commands executed.** So the reader can re-run them rather than guess.
6. **Likely cause.** Ranked, with likelihood, clearly labelled as inference.
7. **Uncertainty.** What is not established. This section is what stops the
   receiving agent from treating a hypothesis as a specification.
8. **Suggested scope.** What a fix should touch.
9. **Explicit non-goals.** What it must not touch.

## What makes a package useful

- **Excerpts, not dumps.** Twelve relevant lines beat a 900-line log.
- **Citations on every claim.** An uncited claim will be treated as fact.
- **Uncertainty stated plainly.** "We did not confirm the pool size in
  production" is more valuable than a confident guess.
- **Non-goals.** Without them a scoped fix becomes a refactor.

## What to leave out

Narrative of the investigation, dead ends that taught nothing, and raw transcript.
The reader needs the conclusion and its support, not the journey.

Secrets are redacted automatically, but read the package before sending it. The
redactor is good, not perfect.

Produce it with `mimir export <session-id> --format md`.
