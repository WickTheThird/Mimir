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

## Evaluation panels

There is no view flag. Which panels appear depends on what is happening, not on
what the operator remembered to type. Cases appear while an evaluation is in
flight; the series appears once there are repeats to compare; the events panel
takes the series slot when no series exists. A flag would make the interesting
state the one you have to know to ask for.

**Cases in flight.** Each case of the running evaluation as it completes, with
its case id, task type, confidence, evidence count, tool calls and duration.
Cases are matched to sessions by prompt, because the harness writes no
`eval_results` row until the whole run finishes.

There is deliberately **no pass or fail column**. Scoring happens in process and
is not on disk until the run ends, so a verdict shown here would be invented.
What is displayed is what was measured. The footer reports mean tool calls and
how many cases were answered with no tools at all, which is the signal that
separated a 7B run from a 1.5B run far more sharply than the pass count did.

**Run series.** Completed runs sharing a corpus, a commit **and** a model, with
mean, range, spread, standard deviation and stability rate. Grouping on all
three matters: averaging across a corpus or commit change produces the mean of
two different experiments.

Below that, only the cases that disagreed between runs, with each run's outcome
and the majority verdict:

```
unstable case                          1  2  3   majority
inv-001-locate-risk-classifier         P  F  F   F 67%
inv-009-ambiguous-namespace            F  P  F   F 67%
```

Stability rate is the fraction of cases with the same outcome in every run.
With three runs a case can only agree 3/3 or 2/3, and the display does not imply
more statistical resolution than that sample supports. Runs stored under
provenance schema 1, and contaminated runs, are excluded from series grouping
entirely.

## Motion

The dashboard animates in three places, and in each the movement carries
information that a static number cannot.

**Sparklines.** CPU, memory and throughput show a trend beside the current
value; the evaluation panel shows per-case durations across the run. An
instantaneous reading cannot distinguish load that is climbing from a spike that
has already passed, and a run that is slowing down looks identical to one that
is not until you can see the shape. The per-case trend is the display that would
have made a three-run decline visible while it was happening rather than three
runs later.

Sparklines scale to the observed range, not to a fixed ceiling, so a flat line
means genuinely flat rather than "too small to see". Fewer than two samples
draws the word `collecting`: one point is not a trend, and rendering it as a
full bar would imply a maximum that was never observed.

Machine trends come from the monitor's own sampling and reset when it restarts.
Per-case durations come from the database and survive a restart, because they
describe MIMIR rather than the display.

**The liveness pulse.** `◐◓◑◒` turns beside a specialist that is currently
issuing model calls, and beside the run header while cases are advancing.
Everything idle shows a static `·`. This is the difference between *slow* and
*hung*, which is otherwise invisible: a long case and a wedged process produce
identical static output.

A spinner that turns while nothing is happening would be an animation
pretending to be a status, so the pulse is driven by observed telemetry
recency rather than by the render loop.

## Council flow

```
  coordinator  44x   5.0s ░░░░░   -
  ├─ k8s        94x   7.9s █░░░░  78t
  ├─ logs       71x   6.9s █░░░░  82t
  ├─ behaviour  76x   5.4s █░░░░  64t
  ├─ repo       58x   5.6s █░░░░  53t
  ├─ safety     11x  11.7s ░░░░░   -
  └─ memory      4x   9.2s ░░░░░   3t
  synthesis    45x  13.7s █░░░░   -
  evidence memory_curato 307  search_reposi 178
  43/43 recent sessions instrumented   403 calls recorded
  never ran: web, sdm
```

**This is not a picture of the model.** Ollama exposes no weights, activations
or attention, so a diagram of neurons or attention heads would be decoration
presented as data. What MIMIR does expose is its own topology, and that is what
this draws.

Structure comes from the code: who may run, and in what order. Weights come from
the database: call count, mean latency, share of total inference time, tool
calls, failures, and evidence attributed to each producer. Neither half is
guessed.

A specialist currently issuing model calls is highlighted and marked `<`. That
is inferred from telemetry rather than from a liveness signal, so it lags by
about one call; the alternative is instrumenting the graph for the display's
benefit, which would let the display disagree with the audit trail.

The `never ran` line names specialists absent from the window entirely. A
specialist that never fires is either correctly unused for this workload or
quietly broken, and the graph is where that distinction becomes visible.

Labels are abbreviated rather than truncated (`k8s`, not `kubernete`): an
abbreviation reads as deliberate, a chopped word reads as a bug.

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
