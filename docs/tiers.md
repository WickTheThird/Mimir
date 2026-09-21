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
| mini | qwen2.5:7b | 64/82 | 40/58 | **0.47** | 2795s |
| upgrade | gpt-oss:120b | 65/82 | 41/58 | 0.40 | 4420s |

One case, one pair, and 15% of the wall clock separate a 30B from a 7B.

Two cases out of 58 separate the 7B from a 117B. The 117B has the best case
rate in the table and the joint worst pair consistency, and it took 58%
longer than the 7B to produce it.

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

Twelve of fifteen pairs land the same way on both tiers.

### A correction, and the reason it matters

An earlier version of this document claimed that all fourteen cases inside
the seven both-failing pairs had identical verdicts on both models, and
argued from that that the failures could not be sampling noise. **That claim
was computed from two runs and does not replicate.** A third run was already
on disk - an earlier replicate of qwen3-coder:30b - and comparing against it
instead gives a different answer: three of the nine contrastive failures
(con-complete-partial-b, con-delete-unasked-b, con-route-direct-a) passed in
that run.

Pairwise agreement between runs, on the 58 model cases:

| comparison | agreement |
|---|---|
| qwen3-coder:30b run 1 vs run 2 (same model) | 42/58 (72%) |
| qwen3-coder:30b run 2 vs qwen2.5:7b | 49/58 (84%) |
| qwen3-coder:30b run 1 vs qwen2.5:7b | 37/58 (64%) |

The two cross-model figures are 64% and 84%. Which one you get depends
entirely on which run of the 30B you happen to compare against, and the
spread between them is wider than the gap between same-model and
cross-model. **At one run per tier, an agreement statistic cannot tell a
model difference from a sampling difference.** The 84% number that looked
like a finding was a draw.

This repository already had the rule that would have caught it - three
replicates before any comparison - written down after the baseline A work.
The rule was not applied to a statistic that arrived looking like a
conclusion.

### What survives three runs

Eight cases fail in every run of every model measured so far. Six of them
are contrastive:

- **con-absence-unreachable-b** - passes "searched three clusters, found no
  billing pod" and fails "searched three clusters, all timed out". It
  answers *no* both times. It does not distinguish a completed search that
  found nothing from a search that could not run. This is the most
  consequential failure in the corpus: on call, those two produce opposite
  actions. The tool result already knows which happened. Nothing infers it
  from prose.
- **con-ground-empty-b** - names pods correctly from a real listing, and
  still names pods when the listing was empty. Whether a list is empty is a
  length check, not a judgement.
- **con-fresh-stale-b** - trusts a note verified 3 days ago, and also trusts
  one never verified and written 14 months ago. The memory bank already
  stores the verification timestamp. A staleness threshold is arithmetic.
- **con-retry-innocent-b** - correctly reads gaps of 1s, 2s, 4s as a retry
  storm, then reads a single request plus an unrelated batch job as one too.
  Gap doubling is a pattern over timestamps.
- **con-complete-done-a** and **con-delete-asked-a** - whether a diff used
  what it introduced is `verify/definitions.py`. Whether a diff deleted far
  beyond its instruction is the `guts_the_file` property in
  `verify/change.py`.

Plus `inv-003-shell-spawning` and `inv-011-caller-side-timeout` from the
wider corpus. Three ambiguity cases - `ambiguous-namespace`,
`inv-009-ambiguous-namespace`, `reg-012-ambiguous-target` - fail on both
tiers in this sweep but passed in the earlier replicate, so they belong with
the unresolved group below rather than here.

Five of the six contrastive survivors have a mechanism already written in
this repository. It is computed and then not carried into the answer path.
That argument does not need the discredited statistic: a case that fails in
three consecutive runs across two model sizes is not waiting for a better
model.

The weaker cases - con-complete-partial-b, con-delete-unasked-b,
con-route-direct-a - flip between runs and are genuinely unresolved. They
may be capacity-bound, they may be prompt-sensitive, and one run each way
cannot say.

## Consequences

**The mini tier is approved for ops and investigation.** It costs nothing
measurable against the laptop, and it is the tier that is always on. The
scope of that claim is this corpus: ops triage, routing, grounding,
evidence handling. Coding was never in it. Coding is generative work with a
much larger output space, and it stays on the laptop until it has its own
measured number.

**Model upgrades are the wrong lever here.** Measured across 7B, 30B and
117B, the next unit of accuracy on these pairs does not come from
parameters. It comes from carrying facts the system already computes into
the answer path. That is the thesis of this project stated as a measurement
rather than a belief, and the four cases that survived every run are where
to start:

1. `con-absence` - the tool result knows whether the search errored or
   returned empty. Carry it.
2. `con-ground` - the listing knows its own length. Carry it.
3. `con-fresh` - the memory bank knows when the note was last verified.
   Threshold it.
4. `con-retry` - gap doubling over timestamps is a pattern match, not a
   judgement.

## A prediction recorded before the result

`gpt-oss:120b` is running the same corpus now. The test is the eight cases
that failed in all three runs so far, and specifically the five with an
existing mechanism: con-absence-unreachable-b, con-ground-empty-b,
con-fresh-stale-b, con-retry-innocent-b, con-complete-done-a.

If those fail again on a model roughly seventeen times the mini's size, they
are not capacity-bound and the fix is plumbing. Its aggregate may still be
higher, because the non-pair cases do reward capability.

If instead gpt-oss clears con-absence, con-ground and con-fresh, the
argument is wrong: those are capacity-bound after all, and the right
response is a bigger model on the mini rather than more plumbing.

One run of gpt-oss is one run, and the paragraphs above are what comes of
reading too much into one. It can only strengthen or weaken the case, not
settle it. Recorded before the result so it is checked rather than
rationalised.

## The result

Held for four of five.

| case | 30b run 1 | 30b run 2 | 7b | 120b |
|---|---|---|---|---|
| con-absence-unreachable-b | fail | fail | fail | fail |
| con-ground-empty-b | fail | fail | fail | fail |
| con-fresh-stale-b | fail | fail | fail | fail |
| con-retry-innocent-b | fail | fail | fail | fail |
| con-complete-done-a | fail | fail | fail | **pass** |

Seven cases now fail in all four runs across three model sizes spanning 7B
to 117B: the four above, plus `con-delete-asked-a`, `inv-003-shell-spawning`
and `inv-011-caller-side-timeout`.

Four cases surviving four runs and a seventeen-fold parameter increase is
the claim this document needed, and it is a claim about the system rather
than about any model. MIMIR cannot tell a search that completed and found
nothing from a search that could not run. It names pods from an empty
listing. It trusts a fourteen-month-old unverified note as readily as a
three-day-old verified one. It sees a retry storm in a single request. No
model in the range measured fixes any of it, and every one of those four
facts is already computed somewhere in this repository before the model is
asked anything.

`con-complete-done-a` is the honest exception. Whether a diff did what was
asked - the positive case, where both halves are present and the correct
answer is *yes* - was cleared by the 117B alone. That one looks at least
partly capacity-bound, and a claim that it is pure plumbing would be wrong.

## What the upgrade tier actually bought

Two model cases out of 58, for 58% more wall clock and 65GB resident. Pair
consistency went *down* against the 7B, 0.40 against 0.467.

That combination is the whole argument in one row. The case rate rewards
capability a little, because some cases are just hard. Pair consistency asks
whether the answer changes when the fact changes, and on that measure the
117B is no better than the 30B and worse than the 7B. Scale bought
competence at the cases and nothing at the discrimination.

The aggregate figures are trustworthy in a way the per-case ones are not:
two runs of qwen3-coder:30b both scored exactly 63/82 while 16 of 84
individual cases flipped. But 63, 64, 65 is a two-case spread over three
models with one run each, and the correction above is about exactly this
kind of reading. The safe statement is that all three are the same to within
what this corpus can measure, and nothing here justifies buying a larger
model for the mini.
