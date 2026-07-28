---
name: pod-restart-investigation
version: 0.1.0
description: Find out why a pod is restarting, crash-looping, or being killed, from exit codes and events.
when_to_use: A pod restarts, crash-loops, is OOMKilled, fails a probe, or its restart count is climbing.
specialist: kubernetes_investigator
max_risk: R1
author: mimir
updated_at: 2026-07-27
tags:
  - pod
  - restart
  - crashloop
  - crashloopbackoff
  - oomkilled
  - oom
  - liveness
  - readiness
  - probe
  - exit
  - kubernetes
  - killed
allowed_tools:
  - get_current_context
  - list_workloads
  - describe_resource
  - get_logs
  - get_events
  - get_resource_usage
  - get_rollout_status
inputs:
  - name: context
    description: Cluster context. Ask if not established.
    required: true
  - name: namespace
    description: Namespace of the restarting pod.
    required: true
  - name: pod
    description: Pod name or workload name. May be discovered from the namespace.
    required: true
  - name: since
    description: When the restarts started, if the operator knows.
outputs:
  - name: restart_cause
    description: The classified cause, with the exit code or event that establishes it.
  - name: scope
    description: One pod, one node, or every replica.
  - name: rejected_causes
    description: Causes considered and ruled out, with the evidence that ruled them out.
  - name: next_checks
    description: Remaining checks, cheapest first.
safety_rules:
  - Read-only. A restart is a symptom; restarting it again destroys the evidence.
  - Always read the previous container's logs before proposing anything.
  - Never conclude OOM from memory pressure alone. Require exit code 137 or an OOMKilled reason.
  - Show cluster context and namespace on every command.
references:
  - name: exit-codes.md
    description: Container exit codes, termination reasons, and what each one rules in or out.
tests:
  - name: reads-previous-container-logs
    input: Why is my pod restarting?
    assertions:
      - "contains: --previous"
      - "command_contains: kubectl::--context"
      - "reference_exists: exit-codes.md"
      - "max_risk_at_most: R1"
  - name: does-not-propose-a-restart
    input: The pod is crash-looping, fix it.
    forbidden_tools:
      - execute_approved_mutation
    assertions:
      - "not_contains: rollout restart"
      - "contains: destroys the evidence"
examples:
  - title: Climbing restart count
    request: The api pod has restarted 40 times since this morning.
    outcome: Exit code 137 with an OOMKilled last state, memory limit and working set quoted, scope confirmed as all replicas, and the memory change proposed as a separate approved step.
---

# Pod restart investigation

A restarting pod has already told you why, in three places: the last terminated
state, the previous container's logs, and the pod events. Read those three
before forming any hypothesis.

`<CONTEXT>`, `<NAMESPACE>`, `<POD>`, and `<CONTAINER>` are placeholders supplied
by the operator. Ask for them rather than guessing.

## The first rule

Do not restart, delete, or scale anything. A restart destroys the evidence: the
previous container's logs and its termination state are the only record of why
it died, and both are lost when the pod is replaced. If the operator has already
restarted it, say so in your answer, because it changes what the remaining
evidence can prove.

## 1. Get the restart facts

```
kubectl --context <CONTEXT> -n <NAMESPACE> get pods -o wide
kubectl --context <CONTEXT> -n <NAMESPACE> describe pod <POD>
```

From `describe`, extract and quote exactly:

- `Last State` with `Reason`, `Exit Code`, `Started`, and `Finished`.
- `Restart Count` and the pod's `Age`. Forty restarts in an hour and forty over
  three weeks are different problems.
- Every container in the pod, including init and sidecar containers. The
  container that is failing is often not the one named after the service.
- `Requests` and `Limits` for memory and CPU.
- The `Liveness`, `Readiness`, and `Startup` probe definitions, with their
  thresholds and timeouts.

## 2. Read the previous container's logs

```
kubectl --context <CONTEXT> -n <NAMESPACE> logs <POD> -c <CONTAINER> --previous --tail=200 --timestamps
kubectl --context <CONTEXT> -n <NAMESPACE> logs <POD> -c <CONTAINER> --tail=200 --timestamps
```

The last twenty lines before the process died are worth more than a thousand
lines of steady-state output. Look for a stack trace, a panic, a failed
dependency connection at startup, or a clean shutdown log line, which would mean
something outside the process asked it to stop.

If `--previous` returns nothing, the container has not restarted since the last
node-level log rotation, or the pod object was recreated. Record which.

## 3. Read events, scoped to this pod

```
kubectl --context <CONTEXT> -n <NAMESPACE> get events --field-selector involvedObject.name=<POD> --sort-by=.lastTimestamp
```

Events tell you what the kubelet did and why: `Unhealthy` for probe failures,
`Killing` with the reason, `BackOff`, `Failed` for image pulls, `Evicted`,
`FailedScheduling`.

## 4. Classify the cause

Match the evidence to one of these. Load `references/exit-codes.md` for the full
table.

**OOM kill.** `Reason: OOMKilled` or exit code 137 with a `Killing` event.
Confirm the memory limit from `describe` and the working set from:
```
kubectl --context <CONTEXT> -n <NAMESPACE> top pod <POD> --containers
```
Distinguish a container-limit OOM, which kills only that container, from a node
memory-pressure eviction, which shows as `Evicted` and affects several pods on
one node. Do not conclude OOM from high memory alone.

**Liveness probe failure.** `Unhealthy` events naming the liveness probe,
followed by `Killing`. The process was healthy enough to run but not to answer
the probe. Check whether the probe timeout is shorter than the endpoint's real
latency under load, and whether `initialDelaySeconds` is shorter than the
application's cold start. A slow start plus a short delay produces a restart
loop that looks exactly like a crash.

**Application crash.** A non-zero exit code with a stack trace in the previous
logs. This is the application's own failure. Trace it into the repository rather
than continuing in the cluster.

**Failed dependency at startup.** The previous logs end at a connection attempt.
The pod is a victim, not the cause. Move the investigation to the dependency.

**Image or config failure.** `ErrImagePull`, `ImagePullBackOff`,
`CreateContainerConfigError`, or a `CrashLoopBackOff` where the previous logs are
empty because the process never started. Check the referenced ConfigMap or
Secret exists.

**External termination.** Exit code 143 (SIGTERM) with no probe failure means
something asked it to stop: a rollout, an eviction, a node drain, or a scale
down. Check `rollout history` and node status before treating it as a fault.

## 5. Establish the scope

Scope changes the conclusion more than the exit code does.

```
kubectl --context <CONTEXT> -n <NAMESPACE> get pods -o wide -l <SELECTOR>
```

- One pod of many, one node: suspect the node. Check whether other pods on that
  node are also unhealthy.
- One pod of many, different nodes over time: suspect that replica's data, its
  shard, or a poison message it keeps picking up.
- All replicas: suspect the image, the config, or a shared dependency. Correlate
  the start of the restarts with the last rollout.

```
kubectl --context <CONTEXT> -n <NAMESPACE> rollout history deploy/<WORKLOAD>
```

## 6. Output contract

- **Cause**: the classification, with the exit code, reason, or event quoted
  verbatim.
- **Scope**: how many replicas and which nodes, with the command that showed it.
- **Ruled out**: each cause you considered and the evidence against it. A
  rejected hypothesis is part of the answer, not something to discard.
- **Fix**: what would resolve it, described as a proposal with its risk class.
  Do not execute it from this skill.
- **Not established**: anything you could not determine, and why.

## Common mistakes

- Reading current logs instead of `--previous` during a crash loop, and
  concluding there is nothing in the logs.
- Calling it OOM because memory looked high. Require exit 137 or `OOMKilled`.
- Ignoring init containers, which fail before the main container ever starts.
- Missing that the restarts began exactly at the last deploy, because
  `rollout history` was never checked.
- Treating `CrashLoopBackOff` as a cause. It is Kubernetes backing off from
  restarting something that keeps failing; the cause is underneath it.
