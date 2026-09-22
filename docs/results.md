# Results: the plan, measured

- Date: 22 September 2026
- Model: qwen3-coder:30b throughout, so the only variable is the system
- Companion to `plan.md` (the sequence) and `tiers.md` (the baselines)

## The ops corpus, 82 cases, every step live

| run | all | model-only | pair consistency | iteration_limit failures |
|---|---|---|---|---|
| before, run 1 | 63/82 | 39/58 | 0.40 | 16 |
| before, run 2 | 63/82 | 39/58 | 0.40 | 16 |
| gates alone (21 Sep) | 62/82 | 38/58 | 0.40 | 17 |
| **the plan (22 Sep)** | **68/82** | **44/58** | **0.33** | **0** |

Five more cases than either baseline, on a corpus where the two baselines
agreed to the case. The dominant failure category is gone: not reduced,
gone. Eight cases newly pass and three newly fail. Four of the ten cases
that had failed in all three prior 30B runs now pass:
`con-complete-done-a`, `con-fresh-stale-b`, `con-retry-innocent-b`,
`inv-011-caller-side-timeout`. The last is the caller-or-callee timeout
that the entity store was built for.

Failure categories, before and after:

| category | before | after |
|---|---|---|
| iteration_limit | 16 | 0 |
| forbidden_content | 6 | 5 |
| missing_citation | 4 | 5 |
| incorrect_root_cause | 4 | 1 |
| missing_target_information | 3 | 1 |
| underconfident | 1 | 0 |
| (none recorded) | 0 | 4 |

The categories that remain are judgement and wording. The one that was
structure is gone.

## The part that went the wrong way

Pair consistency fell from 0.40 to 0.33: five pairs of fifteen answered
correctly on both sides instead of six. The plan's target was above 0.7,
and this number is the reason the plan existed. It has to be read
carefully rather than explained away.

Two pairs gained (`con-fresh`, `con-retry`), four split that were whole
(`con-approve`, `con-intent`, `con-support`, and the pair figure also lost
`con-absence` and `con-ground` which had gone 6/6 in the targeted run an
hour earlier). The stored session metadata says what happened to each:

- `con-absence-unreachable-b`: retrieval correctly classified as
  **failed** from the operator's statement. The model then wrote "there is
  insufficient evidence to confirm whether billing pods exist". No definite
  claim, so no overreach fired, so the word "unknown" never appeared. Right
  in substance, failed on the word.
- `con-ground-empty-b`: retrieval correctly **empty**; the model wrote
  "the listing returned no pods because only one cluster was queried".
  Correct, and not the word "none".
- `con-support-absent-b`: the answer refutes the claim by quoting it,
  "the claim that the queue waits five seconds between retries is not
  supported", and the case forbids the phrase. The answer is right; the
  assertion penalises quoting what is denied. Recorded as a corpus
  weakness, not changed, because changing an expectation after seeing the
  result is the thing this project refuses.
- `con-intent-read-a` and `con-approve-real-b`: one run each way on
  earlier baselines; churn until replicated.

The first two are the gate being right and not saying the word. Fixed
after the sweep by making the verdict word canonical under `failed` and
`empty` ("Unknown: ..." / "None: ..."), and those four cases re-run alone.


Re-run of the four absence and ground cases with the verdict word canonical: **4 of 4**.

So the honest statement about pair consistency: the mechanism that was
supposed to move it did fire on the cases it was built for, the answers
were substantively correct, and the metric did not credit them because it
keys on a word. That is partly a metric problem and partly a real one: an
operator wants the word too. Neither excuse makes 0.33 into 0.7. It stays
the number to move.

## The coding corpus, first number

| | cases | passed | pair consistency | time |
|---|---|---|---|---|
| qwen3-coder:30b | 8 | **8** | **1.0** | 140s |

First reported as 7 of 8. The one failure was a corpus needle that matched
the removal line every correct rename contains; the stored diff was a
complete rename. The needle now targets an added line calling the old
name, a test pins the distinction, and the stored diffs re-check to 8 of
8. Eight cases and one run: a direction, and the first coding number this
project has had. The contrastive twins that matter all landed: no edit when
the constant is already there, no edit when the test already passes, no
deletion when none was asked for, the test updated when it is the only
caller.

## What was learned that was not in the plan

- **Memory recall is not progress.** The recurrence looped to its cap on
  every prose case because recalled notes counted as new evidence. Only
  observed evidence counts now. This is why per-case time tripled in the
  first re-run and why the full sweep did not.
- **The operator's explicit words outrank the decider.** Kev-4B read "no
  listing was produced" as observed at p=0.58. The statement is not a
  judgement call; the decider is for the residual.
- **Uncalibrated verdicts get a margin floor.** An argmax at 0.58 over
  three options is close to noise, and acting on it because the closed
  set is guaranteed mistook the guarantee for a decision.
- **Two corpus assertions were wrong in the same way.** A needle that a
  correct answer necessarily contains. One was fixed because it was a
  logical impossibility; the other is recorded because it is arguable.

## What is not claimed

One run per configuration on one model. The aggregates on this corpus have
reproduced exactly across replicate runs before while a fifth of the cases
flipped, so 63 to 68 is likely real and the per-case lists are not to be
read individually. Pair consistency is below where it was. The mini tier
has not been re-measured with the plan live. Coding has eight cases. Kev's
verdicts are uncalibrated on the Qwen3 revision until the calibration
step has thirty samples per field, which it does not yet.
