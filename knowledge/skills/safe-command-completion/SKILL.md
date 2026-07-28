---
name: safe-command-completion
version: 0.1.0
description: Turn a natural-language request into a correct, fully targeted command.
when_to_use: The operator describes what they want to do and wants the command for it.
specialist: kubernetes_investigator
max_risk: R1
author: mimir
updated_at: 2026-07-27
tags: [command, how-do-i, generate, construct, kubectl, cli, syntax]
allowed_tools:
  - get_current_context
  - list_namespaces
  - list_workloads
  - describe_resource
  - search_memory
inputs:
  - name: request
    description: What the operator wants to do, in their words.
    required: true
outputs:
  - name: command
    description: The command, with every target resolved.
  - name: explanation
    description: What it touches and whether it changes anything.
tests:
  - name: never-defaults-the-namespace
    input: Restart the api deployment.
    assertions:
      - "contains: Do not fall back to"
  - name: passes-context-explicitly
    input: Show me the pods.
    assertions:
      - "contains: --context"

---

# Safe command completion

## Resolve before constructing

A command is not correct until its target is unambiguous. Resolve, in this order:

1. What the operator said explicitly.
2. The session context (`/ns`, `/cluster`, or the flags they passed).
3. Configured defaults.
4. `get_current_context`, and say that is where it came from.

If the namespace is still unknown, **ask**. Do not fall back to `default`.
Guessing the namespace is the single most common way to touch the wrong thing,
and the guess is invisible in the output.

## Construct

- Pass `--context` and `-n` explicitly, even when they match the current context.
  The operator should be able to read the command and know what it hits without
  checking their shell state.
- Build an argument vector. No pipes, no redirection, no command substitution.
  If the work genuinely needs a pipeline, propose the steps separately and say
  why.
- Prefer the narrowest form that answers the question. `-l app=api` beats
  `--all`. A specific pod beats a selector when you know the pod.
- Prefer structured output (`-o json`, `-o jsonpath=...`) when the result will be
  parsed rather than read.

## Present

Show the command, then what it targets, then whether it changes anything. For
anything mutating, add the expected effect and the rollback.

Never claim to have run it. Say what it will do, and let the operator decide.

## Verify what you can

If a resource name was inferred rather than given, confirm it exists with a
read-only lookup before proposing a command that assumes it. A command targeting
a misspelled deployment fails loudly, which is fine; one targeting a real but
different deployment does not, which is not.
