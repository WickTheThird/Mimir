---
name: timeout-investigation
version: 0.1.0
description: Determine which side of a call actually gave up, and why.
when_to_use: A service is timing out, latency spiked, or calls are failing with deadline exceeded.
specialist: log_analyst
max_risk: R2
author: mimir
updated_at: 2026-07-27
tags: [timeout, latency, deadline, slow, hang, "504", "499", retry, upstream, downstream]
allowed_tools:
  - ingest_logs
  - filter_logs
  - group_repeated_errors
  - extract_correlation_ids
  - correlate_logs
  - detect_timeout_patterns
  - summarise_log_volume
  - compare_before_after
  - get_logs
  - get_events
  - get_resource_usage
  - list_workloads
  - run_python
inputs:
  - name: service
    description: The service reporting the timeout.
    required: true
  - name: time_range
    description: When it started. Without this the search is unbounded and useless.
    required: true
outputs:
  - name: owner
    description: Which side gave up first, with the evidence for it.
  - name: hypotheses
    description: Ranked causes, each with the cheapest check that would settle it.
tests:
  - name: does-not-conclude-without-both-sides
    assert_contains: ["caller", "callee"]
---

# Timeout investigation

The mistake this skill exists to prevent: concluding that the slow service is the
broken one. Usually it is the victim.

## The one question that orders everything else

**Who gave up first?** A timeout is one party deciding to stop waiting. Find that
party before theorising about causes.

- If the caller's elapsed time clusters tightly on a round number (1s, 5s, 30s)
  and the callee kept working past it, the caller's configured timeout is the
  proximate cause. The callee being slow is the underlying cause, and the two
  need separate fixes.
- If the callee returned an error quickly and the caller reported a timeout, the
  timeout is a red herring and you are looking at the wrong error.

## Order of work

1. **Bound the window.** Get the time range from the operator. An unbounded log
   search returns noise.
2. **Find the first error, not the loudest.** Sort by time and read the earliest
   occurrence in the window. Downstream floods are symptoms.
3. **Group before reading.** `group_repeated_errors` collapses variable durations
   and ids into templates. Ten thousand lines usually become four distinct
   failures.
4. **Get correlation ids.** `extract_correlation_ids` finds them by key name and
   by value shape. Pick ids that appear on both sides.
5. **Correlate.** `correlate_logs` lines up caller and callee on that id. This is
   where "who gave up first" is answered, with timestamps.
6. **Check for retry amplification.** The same id repeating with growing backoff
   multiplies load downstream. A retry storm can be the whole incident.
7. **Rule things in and out explicitly.** State which of these you checked:
   caller timeout, callee latency, ingress or proxy timeout, database, connection
   pool exhaustion, DNS, resource starvation, deploy correlation.

## Signals and what they mean

| Observation | Rules in | Rules out |
| --- | --- | --- |
| Durations cluster within a few ms of a round number | a configured timeout | natural latency |
| Callee finished after the caller gave up | caller-side timeout too tight, or callee regression | callee returning errors |
| "connection pool exhausted", "timeout acquiring connection" | pool saturation, often caused by slow queries upstream of it | network |
| Same trace id N times with growing gaps | retry amplification | a single slow call |
| Errors start exactly at a rollout timestamp | deploy correlation | gradual degradation |
| CPU throttling present on the callee | limits set too low | application logic |
| Latency flat but error rate up | not a timeout at all | latency regression |

## What to report

Ranked hypotheses with a likelihood, the evidence for and against each, and for
each one the single cheapest command that would confirm or kill it. Keep the
hypotheses you rejected and say why. "I ruled out the database because query
latency was flat across the window" is often the most useful line in the report.

## Where this stops

Changing a timeout value, restarting a workload, or scaling anything is a
mutation. Prepare it, show the blast radius and the rollback, and stop.
