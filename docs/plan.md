# The plan, reframed around a decision model that actually exists

- Date: 22 September 2026
- Supersedes the ordering in `state.md` §"The order" and `gaps.md` §"ranked
  by leverage". Those documents stay as the diagnosis; this is the sequence.
- Why the reframe: `NimbleDecider` already targets the open System One
  reproduction (frozen 4B Qwen, one forward pass, a distribution per field).
  Only the `nimble` package and MLX are missing. That turns "stand up Kev"
  from a research step into an install, and it changes the order of
  everything after it.

---

## What the System One model is and is not, so the plan does not overclaim

- **Closed output set, guaranteed.** A choice from the options offered, a
  score, or yes/no. Nothing else can come back. This is the property every
  decision point needs and the property the generative model cannot give.
- **Low variance, not zero.** Jev's own report says identical inputs can
  differ. A local frozen model at argmax is deterministic for a fixed
  model; a served one may not be. Churn goes down, not to zero, and is
  measured the same way as everything else: replicates.
- **Calibration is trained, not free.** Jev's confidence comes from RLCD.
  A frozen open model gives a *distribution* and a *margin*, which is what
  makes `min_margin` usable, but its probability is not calibrated until we
  calibrate it. Until then every verdict carries `calibrated=False` and
  threshold gates do not apply. That is already how `Verdict` works.
- **It does not write.** Synthesis stays generative. The target is one or
  two generative calls per question, not zero.

## The sequence

Each step is measured on the failure population it targets before the next
begins. Categories, not pass rates, are the signal; nine cases in seventeen
minutes answered what eighty-two in an hour would have.

### 0. Install the decision backend. *An hour. No research.*

`nimble` + `mlx_lm`, a 4B Qwen at 4-bit (about 2.5GB), `decisions.enabled
= true`, `decisions.backend = nimble`. `LocalDecider` stays as the fallback
where MLX is absent.

Moved from fourth to first because doing it later means measuring every
decision twice, once through constrained generation and once through the
real backend. The protocol is the same; the numbers would not be.

Acceptance: `build_decider().available` is true; a verdict on a corpus
prompt returns a distribution over the offered options with a margin.

### 1. Retrieval outcome. *Hours.*

`outcome ∈ {observed, empty, failed}` over the operator's request plus
recorded tool errors, replacing the regex that read a runbook as an
incident. Targets: `con-absence-unreachable-b`, `con-ground-empty-b`.

### 2. Continue / replan / ask / stop. *Two days.*

The recurrence ADR-003 never got. One back edge from `gather` to a new
`assess` node whose whole job is this one four-way choice, decided by the
model over the evidence gathered so far plus a no-progress measure (new
evidence this round versus last). This is the second decision to wire, not
the fourth, because it is the one that turns a checklist into an
investigation, and because it is a closed-set choice with no prose in it.

Bounded by rounds as well as by the verdict, so a bad verdict cannot loop.
Targets: every case where the plan's first step could not have known what
the second step needed. `inv-011-caller-side-timeout` is the exemplar.

### 3. Targeting. *A day, plus the entity store.*

"Which of these workloads did the operator mean." A closed-set choice, but
the set has to come from somewhere: a small entity store populated from
what `kubectl` and repo metadata already return (pod, deployment,
namespace, cluster, repo, log stream). The store is Phase 4 of ADR-003 in
its smallest useful form. The decision model picks; the store supplies the
candidates. Eight corpus cases, and the original `messaging-whatsapp`
complaint.

### 4. Sufficiency and conflict. *A day.*

"Is this enough to answer" and "do these two reports disagree." Retires the
`verify` node's generative call. This is the step that moves pair
consistency, because it is where "does the answer change when the fact
changes" is actually decided.

### 5. Calibrate. *Half a day, once 1 to 4 have run.*

By now every decision has logged (context, options, verdict, later
outcome). Fit temperature scaling per decision type on those pairs with
grouped cross-validation, the same discipline that keeps
`FinalAnswer.probability` at None until earned. Flip `calibrated=True` only
for decision types that pass. Now `min_probability` routes an uncertain
verdict to the operator instead of a guess, which is the on-call property
worth having.

This is also the first concrete instance of the roadmap thesis. The
decision log is experience the organisation accumulates, and it is the
training set for anything smarter later. No completed task is wasted.

### 6. Coding: the selector already uses the decision model. Give it a corpus.

`agent/select.py` asks `satisfies ∈ {yes, no}` per candidate. It has been
the only wired call site all along, running against a backend that did not
exist. After step 0 it works. What it lacks is a number: contrastive pairs
over diffs, same instruction, one altered fact in the repo, the correct
change flips. Then the repository map (symbols, imports, test links, hot
spots) and test selection, both of which feed the selector better
candidates and the gate a smaller failure to read.

### 7. Predictions on hypotheses. *A day, after 2.*

A hypothesis states what the next round should observe. The `assess` node
compares. Residual moves confidence. Cheap once recurrence exists, and it
is the mechanism by which an investigation knows it is wrong before the
harness says so.

### 8. Deploy the mini. *A day.*

7B for prose, 4B Kev for decisions, about 7.5GB resident on a 16GB machine.
The facade behind its key, Warp pointed at it. After steps 1 to 4 the
generative work left is prose, which is what the tier-parity result says a
7B does as well as a 117B.

### 9. Collapse the specialists into a table. *After 1 to 4.*

With the decisions gone from them, each is a tool budget and an objective.

## What does not move, restated

Safety stays in code. Triage routing stays a rule. Tool choice stays under
constrained decoding at its measured 100%. The arithmetic gates stay. No
rule that is always right is replaced by a model that is usually right, and
the decision model is never asked to write.

## Targets, so the plan can be wrong

| | now | after 0-4 | after 5 | after 8 |
|---|---|---|---|---|
| iteration_limit | 0 | 0 | 0 | 0 |
| model-case pass | ~40/58 | 46-50 | same | same, on the 7B |
| pair consistency | 0.40 | > 0.7 | > 0.85 | same |
| generative calls / question | 15-25 | 6-10 | same | same |
| decisions with a usable threshold | 0 | 0 | most | most |
