# Skills

Implements ADR-001 section 10, following the Agent Skills format used by Claude
Code.

## Layout

```
knowledge/skills/<name>/
  SKILL.md          instructions plus YAML frontmatter
  references/       loaded only when the body asks for a file by name
  scripts/          executed through the normal policy and approval gates
  tests/            declarative cases run by `mimir skills validate`
```

## Progressive disclosure

Three levels, and the API makes it hard to load everything by accident:

| Level | What loads | When |
| --- | --- | --- |
| 1 | name, description, when_to_use | always, for every skill |
| 2 | the SKILL.md body | only for a selected skill |
| 3 | a named reference or script | only when the body asks |

The level-1 catalogue is deliberately tiny (`mimir skills list` prints its token
cost). Without this, every task would carry every runbook, which is the failure
ADR 10.2 exists to prevent.

## Frontmatter

```yaml
name: pod-restart-investigation
version: 1.0.0
description: Diagnose why a pod is restarting.
when_to_use: >
  The operator reports restarts, CrashLoopBackOff, or OOMKilled.
specialist: kubernetes_investigator
max_risk: R1
allowed_tools: [list_workloads, get_events, get_logs, summarise_pod_health]
references:
  - name: exit-codes.md
    description: Container exit codes and what each rules in or out.
tests:
  - name: reads-previous-container-logs
    assert_contains: ["--previous"]
tags: [kubernetes, restarts]
```

## Permissions

`allowed_tools` can only narrow. The permitted set is the intersection of the
skill's list with what the specialist may call, computed from the same capability
and risk tables the council itself uses. A skill naming a Kubernetes tool inside
a web-research specialist gets nothing.

This is enforced in code from the frontmatter allowlist, not by the model reading
the prose. A skill body cannot grant itself a tool by asking.

A skill declaring a tool that is not registered fails validation with a message
naming it, so a typo is caught at load time rather than mid-incident.

## Writing one

Start from an investigation you have actually run. A good skill encodes the
order to check things in and what each result rules in or out. A bad one restates
what the tool descriptions already say.

Do not invent environment specifics. The seed skills for SDM, Kannel, Tankers,
and psql are deliberately stubs that say so: ADR 5.4 and 9.3 are explicit that
those workflows are environment specific and must come from approved local
documentation. Fill them in from what you actually run.

Validate before relying on them:

```bash
mimir skills validate
mimir skills show pod-restart-investigation
```

## Hooks

Lifecycle hooks (ADR 10.4) fire before and after tool execution and mutation, on
approval requests, on session completion, on memory promotion, and on web
ingestion. Built-in hooks add audit metadata, refuse a kubectl mutation with an
unresolved context or namespace, block protected namespaces, flag secrets that
survived redaction, flag injection-shaped web content, and require review before
an unverified note is promoted.

Add your own in `~/.mimir/hooks.yaml`:

```yaml
hooks:
  before_mutation:
    - name: notify-oncall
      command: /usr/local/bin/notify-oncall
      blocking: false
```

A blocking hook that exits non-zero denies the action. Hooks are awaited, never
fire-and-forget, because a hook that has not finished has not enforced anything.
