---
name: kannel-container-investigation
version: 0.1.0
description: STUB. Investigate Kannel containers. Fill in from approved local documentation.
when_to_use: A question about Kannel behaviour or its containers. Read the stub warning first.
specialist: sdm_investigator
max_risk: R1
author: mimir
updated_at: 2026-07-27
tags: [kannel, sms, smsc, bearerbox, smsbox, gateway, container]
allowed_tools:
  - get_sdm_status
  - resolve_sdm_resource
  - list_remote_containers
  - inspect_remote_container
  - get_remote_container_logs
  - search_memory
inputs:
  - name: resource
    description: The SDM resource hosting the Kannel containers.
    required: true
outputs:
  - name: findings
    description: Observed state with the commands that produced it.
tests:
  - name: declares-itself-a-stub
    input: Why is Kannel not delivering messages?
    assertions:
      - "contains: STUB"
      - "contains: deliberately incomplete"
      - "max_risk_at_most: R1"

---

# Kannel container investigation (STUB)

**This skill is deliberately incomplete.** ADR-001 5.4 and 9.3 require these
workflows to come from approved local documentation rather than being invented.

## What is not filled in

Everything site specific: container names, the deployment layout, which host runs
what, config file locations, the SMSC links and their identifiers, log paths and
formats, and what the normal steady state looks like.

## Generic Kannel structure, for orientation only

Kannel upstream is split into `bearerbox` (SMSC connections and the message
store) and `smsbox` (the HTTP interface to applications), with `wapbox` where WAP
is used. Its admin interface exposes a status endpoint. That is public knowledge
about the project, not a claim about your deployment.

Whether your deployment matches that shape, how the parts are named, and how to
reach the admin interface are the things you must record here.

## What to fill in

- Container or process names, per environment.
- Where `kannel.conf` lives and which parts matter operationally.
- How to read SMSC link state, and what a healthy state looks like.
- The queue depth that counts as abnormal, with the number.
- Log locations and the format, so the log tools can parse them.
- The two or three failures that actually recur, and how each is recognised.

## Until then

Use `list_remote_containers` and `get_remote_container_logs` to observe, report
what you actually see, and do not assert what a value means unless this file says
so. Ask the operator instead of guessing.
