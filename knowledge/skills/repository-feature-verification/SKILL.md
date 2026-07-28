---
name: repository-feature-verification
version: 0.1.0
description: Determine whether a claimed behaviour actually exists in the current code.
when_to_use: Someone asks whether a feature, fallback, flag, or safeguard exists or is enabled.
specialist: behaviour_verifier
max_risk: R1
author: mimir
updated_at: 2026-07-27
tags: [feature, behaviour, verify, exists, fallback, flag, fail-open, safeguard, does-it]
allowed_tools:
  - list_repositories
  - search_repository
  - find_symbol
  - find_references
  - read_file_range
  - locate_tests
  - build_flow_evidence
  - inspect_git_history
  - describe_resource
inputs:
  - name: claim
    description: The behaviour to verify, stated as a falsifiable sentence.
    required: true
  - name: repo
    description: Repository to search. May be discovered.
outputs:
  - name: verdict
    description: CONFIRMED, ABSENT, PARTIAL, or UNVERIFIABLE.
  - name: evidence
    description: File and line ranges for code, configuration, and tests.
tests:
  - name: states-a-verdict
    assert_contains: ["CONFIRMED", "ABSENT", "PARTIAL", "UNVERIFIABLE"]
---

# Repository feature verification

## Restate the claim as something falsifiable

"Does it fail open?" is not yet answerable. "When the auth client's call exceeds
its deadline, does Verify return nil rather than an error?" is. Do this first;
half the wrong answers come from verifying a vaguer claim than the one asked.

## Convergent evidence

A yes needs more than one of these, and says which it has:

1. **The code path.** The function, the branch, the return.
2. **The configuration.** The default value, and the deployed override if any.
3. **A test.** A test that pins the behaviour is the strongest single signal,
   because it fails if someone removes it.
4. **The deployed manifest**, where the behaviour depends on a deployed value.

## Be adversarial about your own answer

Before saying CONFIRMED, actively look for what would make it false:

- a guard clause or early return above the path you found
- a feature flag that defaults off
- an environment override in the deployed configuration
- the function existing but never being called (check `find_references`)
- a second implementation that shadows the one you found
- the behaviour having been removed recently (`inspect_git_history`)

Dead code that implements a behaviour perfectly is ABSENT, not CONFIRMED.

## When code and deployment disagree

That disagreement IS the finding. Report both sides with citations and do not
average them into a single answer. A repository that says the timeout is 2s and a
ConfigMap that says 30s is exactly the kind of thing this platform exists to
surface.

## Verdicts

| Verdict | Means |
| --- | --- |
| CONFIRMED | Code, configuration, and where available a test agree it exists and is reachable. |
| ABSENT | Searched thoroughly, found nothing, and can say where you looked. |
| PARTIAL | Exists but is gated, incomplete, unreachable, or disabled where it matters. |
| UNVERIFIABLE | Cannot be settled from the repository alone; say what would settle it. |

Answer with exact paths and line ranges. "It is handled in the middleware" is not
an answer; "src/mw/auth.go:112-140 returns nil on ctx.Err()" is.
