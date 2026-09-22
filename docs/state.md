# MIMIR: where it is, and where it ought to be

- Date: 22 September 2026, after a read-only review of the whole tree
- Companion to ADR-004, which assigns decisions to mechanisms. This says what
  the system is, in numbers, and what it becomes when that assignment is done.

---

## Where we are

### The shape

50k lines. Two modes on one tool registry (80 tools) and one safety layer:
a 12-node LangGraph investigation pass for ops, and an agent loop for
coding. 651 tests, 28 files. An eval corpus of 84 cases, 32 of them
deterministic policy checks, 15 contrastive pairs.

### What is measured and holds

| property | value | mechanism |
|---|---|---|
| tool adherence | 100% at 7B, 30B, 117B (native: 80% falling to 33%) | constrained decoding |
| prefix cache | 18x (1.91s to 0.10s) | prompt layout |
| dangerous-command refusal | 14/14, every run | deterministic risk rules |
| risk classification | 12/12, every run | deterministic risk rules |
| aggregate reproducibility | 63/82 twice on one model | the harness |

These are the parts that do not change their mind between runs. Every one
of them is code, or a model held to a schema. None is a model trusted to
judge.

### What is measured and does not hold

| property | value | what it means |
|---|---|---|
| model-case pass rate | 39 to 41 of 58, all three tiers | a third of judgement cases fail |
| pair consistency | 0.40 / 0.47 / 0.40 across 7B / 30B / 117B | the answer tracks the question's shape, not its facts |
| per-case churn | 16 of 84 flip between identical runs | one question, two answers |
| scale sensitivity | 117B buys 2 cases over 7B for 58% more time | parameters are not the lever |
| generative calls per question | roughly 15 to 25 (median 12 tool calls, 60s) | every one is a chance to diverge |

### The one diagnosis under all of it

Every open failure is a decision that was left to the generative model when
it had a small, known answer set.

- *Continue or conclude* was never decided anywhere. It emerged from whether
  the model stopped emitting tool calls before a budget ran out. Sixteen of
  nineteen failures on the 30B. Fixed structurally last night, unmeasured.
- *Did the search fail or return empty* was a regex, which read a runbook
  about timeouts as a report of one. Wrong mechanism for a real judgement.
- *Which workload did they mean, is this evidence enough, do these reports
  conflict* are all still asked of the generative model in open prose, and
  pair consistency at 0.40 is what that costs.

The churn number and the pair number are the same fact seen twice. A
question routed through twenty stochastic calls, each of which can choose
differently, produces a different answer a fifth of the time and cannot
reliably notice when one fact flips.

### The decision layer

Built, tested, 591 lines, wired into one call site, backend disabled, Kev
never stood up. Given a local backend last night. This is the unused asset.

---

## Where we ought to be

### The principle

**MIMIR at full potential is a system in which the generative model writes
prose and nothing else.**

Every choice with a bounded answer set is made either by code, when it is
arithmetic, or by a closed-set decision model, when it is a judgement over
text. The generative model receives decisions already made and turns them
into an answer. It is never the thing that decides.

That is not a limitation on the model. It is what makes the system's
behaviour a property of the system rather than of a sample.

### What that looks like, concretely

**One question costs one generative call, not twenty.** The plan is a rule
over the parsed request (`parse_request` already does most of this). Tool
selection is constrained. Each specialist gathers under a budget and
concludes in one closing turn. Sufficiency, conflict and targeting are
decided by the closed-set model. Synthesis is the single open-ended call.
The harness will show this as per-case churn falling toward the
deterministic floor.

**Pair consistency above 0.85.** The contrastive pairs differ by one fact.
When the fact that differs is read by a decision model into a closed
verdict, and the prose is written from that verdict, the answer flips when
the fact flips. This is the number that says the system reads evidence.
It did not move with scale because scale was never what it measured.

**Kev standing behind every tier-2 decision.** Same protocol, same call
sites as the local backend, swapped by config. Cheaper per decision than a
generative call by an order of magnitude, calibrated by construction, so the
`min_probability` and `min_margin` gates finally mean something and an
uncertain verdict can be routed to a human instead of guessed.

**The mini tier is the production tier.** Tier parity already holds on the
corpus at 7B. Once judgement leaves the generative model, the remaining
generative work is prose, which a 7B writes acceptably. The 16GB machine
that is always on runs the ops mode. The 128GB machine runs coding, where
the output space is genuinely open and the bigger model earns its keep.

**Coding measured on its own corpus.** Nothing above transfers to coding
until it has its own contrastive cases. The best-of-k selector and the
change gate are the right shape; they have no number yet.

**The specialists become a library, not a council.** Twelve named
specialists each with their own prompt is twelve places for behaviour to
drift. With decisions removed from them, each is a tool budget plus an
objective, and the difference between them is which tools they may call.
That is a table, not twelve classes.

### The order

> Superseded on 22 September by `plan.md`, which reorders this around the
> decision backend that already exists in the tree. The analysis below stands.

Each moves alone, and the corpus is re-run after each. The four gates were
shipped together and it cost a session to find which one had regressed.

1. Measure the loop fix. **Done, 22 September.** The nine cases that hit
   `iteration_limit` in both replicate runs were re-run on the 30B after the
   fix: `iteration_limit` on 0 of 9. Three of the nine now pass, two of them
   (`con-fresh-stale-b`, `con-complete-done-a`) for the first time on this
   model. The six that still fail now fail on judgement (`missing 'unknown'`,
   `missing 'none'`, `incorrect_root_cause`), which is exactly the population
   steps 2 to 4 are for. Seventeen minutes on nine cases instead of an hour
   on eighty-two: the failure category was the stable signal, so the
   measurement could be narrowed to it.
2. Retrieval outcome to the decision model. First tier-2 decision, replacing
   the regex that failed on a real run.
3. Targeting to the decision model. Eight corpus cases, and the user's
   original complaint.
4. Sufficiency and conflict to the decision model. Retires the `verify`
   node's model call.
5. Stand up Kev. Flip the config. Re-run.
6. Coding corpus. Contrastive pairs over diffs.
7. Collapse the specialists into a table.

### What must not happen on the way

Safety stays in code. Tool choice stays constrained. A rule that is right
every time it has been measured is not replaced by a classifier that is
right most of the time. And no number from one run is a finding; three
replicates, or it is a hypothesis.
