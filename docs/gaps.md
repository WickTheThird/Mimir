# What is missing

- Date: 22 September 2026
- Method: every goal in ADR-001, every constraint in ADR-002, every phase and
  success criterion in ADR-003, checked against the tree by reading the code.
  No model was run. Companion to `state.md` (where we are) and ADR-004 (where
  decisions go); this one is the list of absences.

---

## ADR-001 goals

| goal | exists | gap |
|---|---|---|
| G1 command assistance | kubectl, sdm, psql, logs, shell tools; deterministic request parse; risk classes 26/26 | none material |
| G2 repository investigation | search, read ranges, LSP definition/references/diagnostics, locate tests | **no whole-repository map.** Every look-up is per file. Nothing knows a repo's modules, call graph or ownership before the first search |
| G3 operational investigation | kubeconfig, SDM, shell, tools bound to context | none material |
| G4 log and timeout diagnosis | log tools, retry-gap detector, staleness, sufficiency gates | **caller vs callee vs proxy is still a generative guess.** `inv-011-caller-side-timeout` fails on every model. Needs a topology to reason over (Phase 4) |
| G5 curated memory | Markdown store, chunk FTS5 + vector fusion, freshness, promotion, bank with decay | no graph, temporal, failure or case-similarity index (Phase 8). Memory is retrieved, not consulted for "have I seen this before" |
| G6 web research | web tool, source capture | works; untested by the corpus |
| G7 interfaces | CLI, REPL, web UI with timeline, OpenAI facade with key auth | **mini deployment not done.** Nothing runs as a service; Warp has nothing to point at |
| G8 safe assistance | deterministic policy engine, approvals, injection scan, worktree isolation | none material. This is the strongest part of the system |

## ADR-002 coding

Exists and is the right shape: task worktrees, bound read/write tools, a
change gate (syntax, rules, lint, LSP delta, formatter), dead-definition AST
check, best-of-k with a deterministic selector, run tests in the worktree.

Missing, in order of how much a coding assistant suffers without it:

1. **A coding corpus.** ADR-002 §5 said no figure may be cited until the
   harness produces it. Two months on, coding has no figure at all. The ops
   corpus does not transfer. Contrastive pairs over diffs, using the gate and
   selector that already exist.
2. **A repository map.** Same gap as G2. A coding assistant that starts every
   task with `search_repository` is reading the codebase for the first time
   on every task. One indexed pass per repo (symbols, imports, test-to-module
   links, recent change hot spots) is what turns "find where X is" from a
   model guess into a lookup.
3. **Test selection.** `run_worktree_tests` runs everything. The change gate
   knows which files changed; the map above would know which tests cover
   them. Running the affected subset is faster and, more importantly, gives
   the model a smaller failure to read.
4. **Multi-step plans with checkpoints.** The loop is one instruction, one
   worktree, N turns. A task that needs "add the model, then the migration,
   then the endpoint" has no representation, so the model either does it all
   in one diff or loses the thread.
5. **Repository memory.** `curate_memory` runs for ops sessions. Nothing a
   coding task learns about a repo (conventions, where things live, what the
   tests are strict about) is written anywhere for the next task.

## ADR-003 phases

| phase | status | what is there / what is not |
|---|---|---|
| 0 foundation | done | provenance, deterministic safety, comparable evals, typed tools, telemetry |
| 1 eliminate waste | mostly | repeat detection, budgets, gathering/concluding split (last night). **No no-progress measure** between rounds, because there are no rounds |
| 2 cognitive state | partial | `Hypothesis` exists and is promoted. **No Entity, Relation, Observation-as-object, Prediction, Transition.** Durable state is still the transcript plus a list |
| 3 close the loop | **absent** | the graph is single-pass. No back edge, no replan, no stop policy. The specialist loop iterates *within* a step; nothing iterates *across* steps on what was found. This is the architecture ADR-003 was written to introduce, and it does not exist |
| 4 world graph | **absent** | no entity resolution, no topology, nothing joins a pod to its deployment to its repo to its logs |
| 5 causal hypotheses | absent | hypotheses do not predict; nothing scores a prediction after the fact; confidence never moves on contradiction |
| 6 mixed substrate | partial | AST for definitions, deterministic request parse, retry-gap arithmetic, staleness arithmetic. No graph algorithms, no symbolic calculation, no config validators |
| 7 simulation-first SE | partial | worktrees, gate, best-of-k, artifact persistence. No test selection, no before/after comparison of runtime behaviour |
| 8 multi-index memory | partial | lexical + vector fusion. No graph, temporal, failure or case indexes |
| 9-11 | not started | correctly; they depend on 2 to 5 |

## ADR-003 §43 success criteria, honestly ticked

- select a later action from an earlier observation: **no** (single pass)
- stop a failing investigation early: **partly** (budgets, not policy)
- state without replaying the transcript: **no**
- competing hypotheses explicit: **yes**, weakly used
- predict before testing: **no**
- reduce confidence when predictions fail: **no**
- code, infra, logs, metrics on shared entities: **no**
- exact engines for formal tasks: **partly**
- verify code changes by isolated execution: **yes**
- safety preserved across model replacement: **yes**, measured
- improve action selection from history: **no**
- replicated gains under the harness: **yes**, with the discipline learned the hard way

Four of twelve. The four are the foundation. The eight are the architecture.

---

## What is actually missing, ranked by leverage

> Superseded on 22 September by `plan.md`, which reorders this around the
> decision backend that already exists in the tree. The analysis below stands.

Not by ADR order. By how much closer each one gets to an assistant you would
trust at 3am and hand a coding task to at 9.

### 1. Decisions out of the generative model (ADR-004, steps 1 to 4)

The accuracy lever. Already argued and measured; the loop fix was step 0 and
it landed. Everything below is worth less until this is done, because every
other component feeds its output through a judgement that is currently a
sample.

### 2. Recurrence with a stop policy (Phase 3)

The capability gap. An investigation that cannot act on what it just found
is a checklist, not an investigation. This is the single biggest difference
between MIMIR and the thing ADR-003 describes, and it is not large to build:
one back edge from `gather` to a new `assess` node that decides
continue / replan / ask / stop. With ADR-004 done, that decision is a
closed-set verdict, not a prompt. The no-progress measure from Phase 1
becomes the stop policy.

### 3. Entities and topology (Phase 4)

What makes G4 answerable and targeting deterministic. A pod belongs to a
deployment in a namespace in a cluster, built from a repo, logging to a
stream. Today every one of those links is re-derived by a model per
question. A small entity store, populated from `kubectl` and repo metadata
the system already reads, turns "is the timeout on the caller or the callee"
into a graph walk plus one decision, and turns "which workload did they
mean" into a lookup with a closed candidate set.

### 4. A repository map (G2, ADR-002 gaps 2 and 3)

What makes coding good on unfamiliar repos. Symbols, imports, test coverage
links, hot spots. One pass per repo, refreshed on change. Feeds test
selection for free. The LSP is the right primitive and the wrong altitude.

### 5. A coding corpus (ADR-002 §5)

Without it, every coding claim is aspirational by the project's own rule.
Contrastive pairs over diffs: same instruction, one altered fact in the
repo, the correct change flips.

### 6. Predictions on hypotheses (Phase 5)

Cheap once 2 exists. A hypothesis states what the next observation should
show; the observation arrives; the residual moves confidence. This is what
makes "reduce confidence when predictions fail" true, and it is the
mechanism by which an investigation *knows* it is wrong rather than being
told by the harness afterwards.

### 7. Memory that learns from tasks (the roadmap thesis)

`curate_memory` writes ops findings. Nothing writes what a coding task
learned about a repo, nothing records which specialist or tool was useful
for which question shape, and no completed task becomes a corpus case
without a person doing it. The thesis says MIMIR improves because its
organisation accumulates experience. Today it accumulates sessions.

### 8. Always-on deployment (G7)

The mini as a service, the facade behind a key, Warp pointed at it. Purely
engineering, no research, and it is the difference between a project and a
tool that gets used.

---

## What is not missing

Worth stating so this reads as a list and not a verdict.

Safety is complete and measured. Tool adherence is solved. The evaluation
apparatus is better than most production systems have. The write surface for
coding is correct. The memory store is sound. Model portability is real:
three sizes were swapped through one config line this week. None of those
needs work; all of them need protecting while the eight above are built.
