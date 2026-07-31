# Monitor

```bash
mimir monitor                        # live, refreshes every 2s
mimir monitor -n 5                   # slower refresh
mimir monitor --log /tmp/run.log     # tail a specific log
mimir monitor --once                 # one frame, for scripts and screenshots
```

Four panels: what MIMIR is doing, what the model runtime is holding, what the
machine is doing about it, and how the current evaluation is progressing.

## Read-only by construction

The monitor usually runs while a long benchmark is running, so it must not be
able to affect the thing it is watching.

- The database is opened `mode=ro`. A dashboard that can take a write lock on
  the store it observes can stall the run it is measuring.
- The runtime is polled with `/api/version`, `/api/ps` and `/api/tags`. None of
  those load a model or run inference. Nothing here ever issues a generate
  call, not even to check liveness.
- Corpus size is read once and cached. Re-parsing the corpus every second to
  render a denominator would make the monitor a measurable load on the machine
  it is supposed to be reporting on.

## Never display a number you cannot source

The same rule that governs evaluation governs the display. Die temperature, GPU
utilisation and per-core power need `sudo powermetrics`, which cannot run
unattended, so those rows say so rather than showing a plausible value.

`pmset -g therm` is readable without root and is what the thermal row reports.
When it says nothing has been recorded, the panel says **no warning recorded**,
not "nominal". Those are different claims and only the first is supported by
the output. A dashboard that prints a reassuring word it did not measure is
worse than one that prints nothing, because the operator believes it and stops
checking.

Where a value is genuinely missing, the reason travels with the absence
(`unavailable (psutil not installed)`).

## Evaluation progress

`eval_runs` gets one row when a run *finishes*, so for the hour a model suite
takes there is nothing in it to read. In-flight progress is counted instead from
the per-case sessions the harness writes as it goes, bounded to those created
after the `mimir evaluate` process started.

The denominator is the number of **model** cases, taken from the harness's own
`EvalCase.deterministic` predicate. Counting the whole corpus would leave the
bar stuck near 40% for the entire run, since deterministic cases open no
session.

No ETA is shown until two cases have finished. One sample is not a pace, and
extrapolating from it produces a confident estimate built on nothing.

A stall warning appears when the gap since the last completed case exceeds three
times the observed mean. A fixed threshold would either cry wolf on a slow model
or stay silent on a fast one.

## Telemetry

The telemetry panel shows measured cost per role over the last hour: call count,
mean latency, share of total inference time, and tokens in and out. Failures and
retries appear beside the token counts.

This panel could not exist before the telemetry repair. `model_calls` held zero
rows, so per-role latency and token cost were unknowable and the only available
number was wall clock for an entire investigation.

Health is assessed only over sessions created **after the first telemetry row**.
Sessions older than that predate the instrumentation, so their lack of model
calls is expected. Counting them reported "17/20 recent sessions recorded no
model calls" at a moment when every session since the fix was correctly
instrumented, which is history rendered as a present fault.

The evaluation panel adds:

- `telemetry` - persisted versus observed calls, and whether they agree
- `valid for` - quality reporting versus efficiency comparison, separately.
  Correctness may still be measurable when telemetry is missing; latency and
  token comparisons are not.
- `tool surface` - the fingerprint of the tools actually enabled. An empty
  fingerprint reads **not comparable**, never as a match with another empty
  fingerprint, because comparing those as equal is what silently disabled the
  capability check.
- `SOURCE CHANGED MID-RUN` when the working tree moved while the run was in
  flight.

## Contamination is visible

The evaluation panel shows offline containment status. A run with
`external_calls > 0`, or one annotated contaminated, is shown in red with the
reason. See [evaluation.md](evaluation.md) for why that annotation exists.

Runs stored under provenance schema 1 are shown with their version, so a reader
can tell they predate the containment and tool-surface fixes rather than
inferring it from which keys happen to be present.

## Audit gaps

The work panel prints raw counts for evidence, executions and model calls, and
warns when activity and telemetry disagree, for example sessions accumulating
while `model_calls` stays at zero.

This is not decoration. The executions table sat at zero across seventeen
sessions because nothing populated it, and the only symptom was a number nobody
was looking at.

## Downloads

An in-flight `ollama pull` is read from the partial blob files, using
`st_blocks` rather than `st_size`. Ollama pre-allocates blobs sparse, so
apparent size is the *target*: reading it reports 100% from the first second,
which is how a stalled download looks finished. Blobs under 1 MB are manifests
and are not listed.

A blob whose mtime has not moved for two minutes is flagged stalled.

## Requirements

`psutil` for CPU, memory and the process table. Without it those rows report
unavailable and the rest of the dashboard still works. Load average and thermal
state come from the standard library and `pmset`, and need nothing extra.
