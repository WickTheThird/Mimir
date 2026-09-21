# Deployment tiers, measured

Two machines run MIMIR: a 128GB laptop that travels, and a 16GB Mac mini
that is always on. They cannot run the same model. The question this
document answers is what an operator gives up by being away from the
workstation, and the answer turned out to be more interesting than the
question.

Corpus: 84 cases, of which 2 are decided but unbuilt and excluded, leaving
82. Twenty-four of the 82 are deterministic policy checks that a rule
answers identically every run; they are separated out below because they
enter every tier's headline figure as guaranteed marks and flatten the
difference. Fifteen of the cases are contrastive pairs: two prompts
identical but for one altered fact that flips the correct answer.

## The numbers

| tier | model | all | model-only | pair consistency | wall clock |
|---|---|---|---|---|---|
| laptop | qwen3-coder:30b | 63/82 | 39/58 | 0.40 | 3287s |
| mini | qwen2.5:7b | 64/82 | 40/58 | 0.47 | 2795s |

One case, one pair, and 15% of the wall clock separate a 30B from a 7B.

That gap must be read against the noise floor, which was measured rather
than assumed: two runs of the same model on the same corpus produced the
same aggregate (63/82 twice) while 16 of 84 individual cases flipped
verdict. Per-case churn is 19%. A one-case difference is inside it.

**The two tiers are indistinguishable on this corpus.** Not "the 7B is
better" - the measurement cannot support a claim that fine in either
direction.

## Why they are indistinguishable

The tempting reading is that the corpus is too easy. It is not: both tiers
answer under half the contrastive pairs correctly on both sides.

The per-pair breakdown says something else.

| pair | laptop | mini |
|---|---|---|
| con-approve, con-cite, con-conflict, con-intent, con-timeout | both | both |
| con-absence, con-complete, con-delete, con-fresh, con-ground, con-retry, con-route | split | split |
| con-restart, con-scope | split | both |
| con-support | both | split |

Twelve of fifteen pairs land the same way on both tiers. Inside the seven
pairs that fail on both, **all fourteen cases have identical verdicts on
both models.** Fourteen out of fourteen, against a 19% churn baseline, is
not sampling noise. These are deterministic failures.

A capacity-bound failure looks different. A 30B would clear some cases a 7B
misses, and the set of failures would be nested. Here the failure sets are
the same set. Quadrupling the parameters moved nothing, which means the
remaining errors are not the model failing to be smart enough.

## What is actually failing

The b-side of a pair is usually the altered-fact twin. Both tiers pass more
a-sides than b-sides (laptop 10/15 vs 7/15, mini 11/15 vs 8/15): they
anchor on the first reading and do not update when the fact flips.

Case by case, with the mechanism that would settle each one:

- **con-absence** - passes "searched three clusters, found no billing pod"
  and fails "searched three clusters, all timed out". It answers *no* both
  times. It does not distinguish a completed search that found nothing from
  a search that could not run. This is the most consequential failure in
  the corpus: on call, those two produce opposite actions. The tool result
  already knows which happened. Nothing infers it from prose.
- **con-ground** - names pods correctly from a real listing, and still
  names pods when the listing was empty. Whether a list is empty is a
  length check, not a judgement.
- **con-fresh** - trusts a note verified 3 days ago, and also trusts one
  never verified and written 14 months ago. The memory bank already stores
  the verification timestamp. A staleness threshold is arithmetic.
- **con-retry** - correctly reads gaps of 1s, 2s, 4s as a retry storm, then
  reads a single request plus an unrelated batch job as one too. Gap
  doubling is a pattern over timestamps.
- **con-route** - routes "why is api restarting" to investigation
  correctly, and routes "get the last 20 log lines from deployment/api" to
  investigation as well. `parse_request` already classifies the
  interrogative.
- **con-complete** and **con-delete** - fail on both sides on both tiers.
  Whether a diff used what it introduced is `verify/definitions.py`.
  Whether a diff deleted far beyond its instruction is the `guts_the_file`
  property in `verify/change.py`.

Six of the seven have a mechanism already written in this repository. It is
computed and then not carried into the answer path for these cases. The
corpus is not measuring how clever the model is. It is measuring which
facts got handed to it.

## Consequences

**The mini tier is approved for ops and investigation.** It costs nothing
measurable against the laptop, and it is the tier that is always on. The
scope of that claim is this corpus: ops triage, routing, grounding,
evidence handling. Coding was never in it. Coding is generative work with a
much larger output space, and it stays on the laptop until it has its own
measured number.

**Model upgrades are the wrong lever here.** The next unit of accuracy on
these seven pairs comes from wiring existing determinism into the answer
path, not from a larger model. That is the whole thesis of the project
stated as a measurement rather than a belief.

## A prediction recorded before the result

`gpt-oss:120b` is running the same corpus now. If the argument above is
right - that these failures are structural rather than capacity-bound -
then a model roughly seventeen times the mini's size should fail
substantially the same fourteen cases. Its aggregate may well be higher,
because the non-pair cases do reward capability. The pair set is the test.

If instead gpt-oss clears con-absence, con-ground and con-fresh, the
argument is wrong: those are capacity-bound after all, and the right
response is a bigger model on the mini rather than more plumbing.

Written before the run finished, so it can be checked rather than
rationalised.
