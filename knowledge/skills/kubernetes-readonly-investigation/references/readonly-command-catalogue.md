# Read-only command catalogue

What each read-only verb proves, and what it does not. Load this when you need
to justify a command choice or when a result looks conclusive and you want to
check whether it actually is.

`<CONTEXT>` and `<NAMESPACE>` are operator-supplied placeholders throughout.

## config current-context

```
kubectl --context <CONTEXT> config current-context
```

Proves: which cluster the command you are about to run will hit.
Does not prove: that you have permission to do anything in it. Pair with
`auth can-i` before promising the operator a command will work.

## auth can-i

```
kubectl --context <CONTEXT> -n <NAMESPACE> auth can-i get pods
kubectl --context <CONTEXT> -n <NAMESPACE> auth can-i --list
```

Proves: whether the current identity is permitted a verb on a resource.
Does not prove: that the resource exists. Useful before proposing a chain of
commands, so a permission failure surfaces once rather than three commands in.

## get

```
kubectl --context <CONTEXT> -n <NAMESPACE> get pods -o wide
kubectl --context <CONTEXT> -n <NAMESPACE> get deploy <WORKLOAD> -o yaml
```

Proves: the current spec and status as recorded in the API server.
Does not prove: what the process inside the container is doing. `-o yaml` on a
single object is the cheapest way to see the full status conditions; avoid it on
a whole namespace, where it floods the context.

## describe

```
kubectl --context <CONTEXT> -n <NAMESPACE> describe pod <POD>
```

Proves: spec, status, container states, and the events still retained for that
object, in one read.
Does not prove: anything about a time window longer than the event retention
period, which is commonly one hour and is a cluster setting, not a constant.

Read specifically: `State`, `Last State`, `Reason`, `Exit Code`,
`Restart Count`, the probe definitions, and the resource requests and limits.

## get events

```
kubectl --context <CONTEXT> -n <NAMESPACE> get events --sort-by=.lastTimestamp
kubectl --context <CONTEXT> -n <NAMESPACE> get events --field-selector involvedObject.name=<POD>
```

Proves: control-plane-visible transitions such as scheduling failures, image
pull errors, probe failures, and evictions.
Does not prove: absence of a problem. Events expire and are also dropped under
load. Unsorted output is near-useless; always sort.

## logs

```
kubectl --context <CONTEXT> -n <NAMESPACE> logs <POD> -c <CONTAINER> --tail=200
kubectl --context <CONTEXT> -n <NAMESPACE> logs <POD> -c <CONTAINER> --previous
kubectl --context <CONTEXT> -n <NAMESPACE> logs <POD> --since=15m --timestamps
```

Proves: what the container wrote to stdout and stderr, for the log lines still
held by the node.
Does not prove: what happened before the last container restart unless you pass
`--previous`, and nothing at all if the application logs to a file inside the
container instead of stdout.

Always pass `--timestamps` when you intend to correlate with anything else.
`--since` is cheaper and more precise than a large `--tail`.

## top

```
kubectl --context <CONTEXT> -n <NAMESPACE> top pod --containers
```

Proves: instantaneous CPU and memory as reported by metrics-server.
Does not prove: a trend, a peak, or the value at the moment of an OOM kill. It
is a sample, not a time series. If metrics-server is absent the command fails
and that failure must be reported, not silently treated as healthy.

Memory here is working set. Compare it to the container limit, not to the node
capacity, when reasoning about OOM risk.

## rollout status and history

```
kubectl --context <CONTEXT> -n <NAMESPACE> rollout status deploy/<WORKLOAD> --timeout=10s
kubectl --context <CONTEXT> -n <NAMESPACE> rollout history deploy/<WORKLOAD>
```

Proves: whether the desired replica set reached availability, and the sequence
of recorded revisions.
Does not prove: what changed between revisions unless change-cause annotations
are actually populated, which is a deployment-pipeline convention rather than a
guarantee. Always pass a short `--timeout`; the default blocks indefinitely.

## Commands that are not read-only

These look harmless and are not. They belong to a mutation-planning skill with
its own approval, never to this one.

- `exec` opens a session inside a running container. Even a read-only command
  through `exec` is elevated inspection, not a read.
- `port-forward` opens a tunnel from the operator's machine into the cluster.
- `cp` writes into or reads out of a container filesystem.
- `edit`, `patch`, `apply`, `scale`, `rollout restart`, `delete`, `cordon`,
  `drain`, `annotate`, and `label` all mutate.
- `debug` creates an ephemeral container, which is a mutation of the pod spec.
