---
name: sdm-resource-investigation
version: 0.1.0
description: STUB. Investigate a resource reached through StrongDM. Fill in from approved local documentation.
when_to_use: Access to the target is mediated by SDM. Read the stub warning before relying on this.
specialist: sdm_investigator
max_risk: R1
author: mimir
updated_at: 2026-07-27
tags: [sdm, strongdm, resource, connect, access, tunnel]
allowed_tools:
  - get_sdm_status
  - list_sdm_resources
  - resolve_sdm_resource
  - search_memory
inputs:
  - name: resource
    description: SDM resource name or a partial name to resolve.
    required: true
outputs:
  - name: findings
    description: What was observed, with the commands that produced it.
tests:
  - name: declares-itself-a-stub
    input: Investigate the payments SDM resource.
    assertions:
      - "contains: STUB"
      - "contains: Never guess the name"
      - "max_risk_at_most: R1"

---

# SDM resource investigation (STUB)

**This skill is deliberately incomplete.**

ADR-001 sections 5.4 and 9.3 are explicit: the exact SDM commands, resource
naming, and access mechanisms are environment specific and "must be learned from
approved local documentation and curated skills. They are not invented in this
ADR."

So this file contains no invented resource names, no assumed connection
mechanics, and no guessed port conventions. Inventing them would produce commands
that look authoritative and are wrong, which is worse than having nothing.

## What is safe to rely on today

The generic surface, verified against the installed client:

- `sdm status` reports local connection state.
- `sdm ls` lists resources visible to the authenticated user.
- `sdm connect <resource>` establishes access (R2, gated).

`resolve_sdm_resource` fuzzy-matches a partial name against the real listing and
reports candidates when the match is ambiguous, so a name is discovered rather
than assumed.

## What you need to fill in

Replace this section with your own environment's facts:

- The resource naming convention, and how to tell environments apart.
- Which resources are databases, which are SSH targets, which are HTTP.
- The local port convention, if any, once connected.
- Which resources are production, so the safety rules can recognise them. Add
  those patterns to `sdm.denied_resource_patterns` or the production patterns in
  `~/.mimir/config.yaml`.
- The workflow your team actually follows, in the order they follow it.

## The workflow shape ADR 5.4 specifies

1. Identify or ask for the resource. Never guess the name.
2. Verify local SDM status first.
3. Connect with the existing authenticated client. MIMIR never handles
   credentials.
4. Identify the target container or service.
5. Run read-only inspection only.
6. Capture output and summarise the evidence.
7. Propose the next check.
8. Stop before any mutation.

Until this is filled in, ask the operator rather than proposing an SDM command
you cannot support.
