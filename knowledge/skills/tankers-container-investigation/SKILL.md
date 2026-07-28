---
name: tankers-container-investigation
version: 0.1.0
description: STUB. Investigate Tankers containers. Fill in from approved local documentation.
when_to_use: A question about Tankers behaviour or its containers. Read the stub warning first.
specialist: sdm_investigator
max_risk: R1
author: mimir
updated_at: 2026-07-27
tags: [tankers, container, service]
allowed_tools:
  - get_sdm_status
  - resolve_sdm_resource
  - list_remote_containers
  - inspect_remote_container
  - get_remote_container_logs
  - search_memory
inputs:
  - name: resource
    description: The SDM resource hosting the Tankers containers.
    required: true
outputs:
  - name: findings
    description: Observed state with the commands that produced it.
tests:
  - name: declares-itself-a-stub
    input: Why is Tankers unhealthy?
    assertions:
      - "contains: STUB"
      - "contains: deliberately incomplete"

---

# Tankers container investigation (STUB)

**This skill is deliberately incomplete, and more so than the others.**

ADR-001 5.4 and 9.3 require environment-specific workflows to come from approved
local documentation. "Tankers" is an internal system name. Unlike Kannel there is
no public project to describe generically, so this file contains nothing about
what it is or how it behaves. Anything written here without your input would be
fabrication.

## What to fill in

- What Tankers is and what it is responsible for, in two sentences.
- The container or service names, per environment.
- Which SDM resource reaches it.
- Its dependencies, and what breaks when each is unavailable.
- Log locations and format.
- What healthy looks like, concretely enough to compare against.
- The failures that actually recur, and how each is recognised.
- The read-only commands you run first when it misbehaves, in order.

## Until then

Observe with `list_remote_containers`, `inspect_remote_container`, and
`get_remote_container_logs`. Report what you see. Do not interpret a value as
healthy or unhealthy without something here to compare it against, and ask the
operator rather than inferring.
