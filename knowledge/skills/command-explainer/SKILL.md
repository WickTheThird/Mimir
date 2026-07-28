---
name: command-explainer
version: 0.1.0
description: Explain what a command does, what it touches, and how it could go wrong.
when_to_use: The operator pasted a command and wants to understand it before running it.
specialist: safety_reviewer
max_risk: R0
author: mimir
updated_at: 2026-07-27
tags: [explain, what-does, safe, review, understand, before-i-run]
allowed_tools: []
inputs:
  - name: command
    description: The command to explain.
    required: true
outputs:
  - name: explanation
    description: Effect, targets, blast radius, risks, and a safer alternative.
tests:
  - name: always-states-rollback
    input: What does `kubectl -n payments delete pod api-0` do?
    assertions:
      - "contains: rollback"
      - "max_risk_at_most: R0"
  - name: never-executes-anything
    input: Explain this command.
    assertions:
      - "contains: Nothing is executed"

---

# Command explainer

Nothing is executed. This is analysis only.

## Cover, in order

1. **What it does**, in one sentence, in plain language.
2. **Flag by flag.** Every flag that changes the target set or the behaviour. Do
   not skip the ones you are unsure of; say you are unsure.
3. **What it touches.** Cluster, namespace, resources, database, files. Be
   specific about how many objects match, and say when the count is unbounded.
4. **Whether it changes anything.** Read-only or mutating, stated plainly.
5. **How it could go wrong.** The realistic failure, not a theoretical one.
6. **Rollback.** The actual command that undoes it, or an explicit statement that
   nothing does.
7. **A safer alternative,** where one exists. A `--dry-run`, a narrower selector,
   or a read-only command that answers the same question.

## Things worth flagging every time

- `--all`, `-A`, `--all-namespaces`: the target set is everything in scope.
- `--force`, `--grace-period=0`: safety mechanisms are being bypassed.
- An empty or missing selector: matches everything, not nothing.
- `UPDATE` or `DELETE` with no `WHERE`: every row.
- A context or namespace that is not visible in the command: the operator may
  believe they are somewhere they are not.
- `| sh`, `curl ... | bash`: the content executed is not visible here.
- A wildcard in a path passed to a destructive command.

## Do not

Do not guess at a flag you do not know. Say it needs checking against
`--help`. A confident wrong explanation is worse than an incomplete one, because
it is the reason the operator will press enter.
