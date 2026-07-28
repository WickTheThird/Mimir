# Container exit codes and termination reasons

Loaded on demand by the `pod-restart-investigation` skill. Use it to turn a
`lastState.terminated` block into a shortlist of causes.

The field to read first is `.status.containerStatuses[].lastState.terminated`:

```
kubectl -n <ns> get pod <pod> -o jsonpath='{range .status.containerStatuses[*]}{.name}{"\t"}{.lastState.terminated.reason}{"\t"}{.lastState.terminated.exitCode}{"\t"}{.lastState.terminated.finishedAt}{"\n"}{end}'
```

## Exit codes

| Code | Meaning | What it rules IN | What it rules OUT |
| --- | --- | --- | --- |
| 0 | Clean exit | Process finished its work, or a job container completed. For a long-running service a restart with code 0 usually means the main process returned, for example a bad entrypoint or a CLI flag that makes it print and exit. | Crash, OOM |
| 1 | Generic application error | Unhandled exception, failed startup, bad configuration. Read the previous container's logs, the last 30 lines before exit. | Signal-based kill |
| 2 | Shell misuse | Malformed entrypoint or command in the manifest. | Application logic |
| 126 | Command found but not executable | Wrong permissions on the entrypoint, or a script without an interpreter line. | Application logic |
| 127 | Command not found | Wrong image, wrong path in `command`, or a missing binary after a base-image change. Compare the image tag with the previous rollout. | Application logic |
| 128 | Invalid exit argument | Rare, usually a wrapper script bug. | |
| 137 | SIGKILL (128 + 9) | OOMKilled, or a failed liveness probe whose termination grace period expired. Check `reason` to tell them apart. | Clean shutdown |
| 139 | SIGSEGV (128 + 11) | Segmentation fault. Native code, a bad CGO call, or a corrupted binary. | Configuration |
| 143 | SIGTERM (128 + 15) | Graceful shutdown requested. Normal during a rollout, an eviction, or a scale-down. Suspicious only if unexplained. | Crash |
| 255 | Exit status out of range | Often an application exiting with -1. Treat as a generic error. | |

## Termination reasons

| Reason | Meaning | First checks |
| --- | --- | --- |
| `OOMKilled` | The container exceeded its memory limit. The kernel killed it, the application never saw it coming, so the logs usually end mid-sentence with no error. | Compare `resources.limits.memory` with the working set from `kubectl top pod`. Look for a memory ramp before each restart, which points at a leak rather than a limit that is merely too tight. |
| `Error` | The process exited non-zero. | Previous container logs. |
| `Completed` | The process exited zero. | Whether this workload should be a Job rather than a Deployment. |
| `ContainerCannotRun` | Docker or containerd could not start the process. | The `message` field; usually a bad entrypoint. |
| `DeadlineExceeded` | The pod exceeded `activeDeadlineSeconds`. | Job specification. |
| `Evicted` | The node reclaimed resources. | Node pressure, `kubectl describe node`, and whether the pod set requests at all. |

## Waiting reasons that are not restarts

These appear in `state.waiting` and mean the container never started, which is a
different investigation from a crash loop:

- `CrashLoopBackOff`: the container did start and then died repeatedly. The
  backoff is a symptom; the cause is in the previous container's logs.
- `ImagePullBackOff` / `ErrImagePull`: registry auth, a wrong tag, or a network
  path to the registry. Check events, not logs.
- `CreateContainerConfigError`: a referenced ConfigMap or Secret key is missing.
  The event message names it.
- `ContainerCreating` for a long time: usually a volume that will not attach, or
  an admission webhook that is slow. Check events.

## The distinction that matters most

OOMKilled with a memory ramp before each kill points at a leak, and raising the
limit only buys time. OOMKilled with a flat memory profile that dies under load
points at a limit set below the real working set, and raising it is the fix.
Establish which one you have from `kubectl top pod` history or a dashboard
before proposing a limit change.

A liveness probe failure also produces SIGKILL and exit 137, which is easily
mistaken for OOM. The events will show `Liveness probe failed` and the reason
will not be `OOMKilled`. That distinction changes the fix from a memory limit to
a probe timeout or a slow-startup problem.
