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

### 2b. Warp reaches the real MIMIR. *Two days. The step the goal depends on.*

The facade is a model gateway. Warp keeps its own agent loop and calls
MIMIR as if it were a model, so from Warp none of the graph, the gates,
the decision layer or the safety engine runs. Every accuracy gain in this
plan is behind the graph, and Warp never enters it.

`docs/warp.md` names the fix and says it is unbuilt: MIMIR capabilities as
MCP tools Warp invokes deliberately, with privileged execution staying
local. Three tools, each running the full path with every gate:

- `construct_command(request)`: the deterministic parse, the risk class,
  the argument vector, the context and namespace it targets. Already the
  strongest part of the system; 25 of the 37 command cases are decided by
  rules that pass every run. This one can ship before steps 1 to 4.
- `investigate(question)`: the graph, with recurrence once step 2 lands.
  Returns the answer with its evidence and its unverified list.
- `code_task(instruction, repo)`: the coding loop, worktree, gate and
  selector. Returns the diff and the test result, never applies it.

One agent loop stays in charge, Warp's, and MIMIR is a set of things it
can call that are right for the reasons the harness measured. Nesting the
graph behind the model endpoint stays the thing to avoid.

Acceptance: a command asked for in Warp arrives through `construct_command`
with the same risk class the CLI gives it, and a mutation is never run
without the approval the policy engine requires.

### 3. Targeting, and the entity store it needs. *Two to three days.*

"Which of these workloads did the operator mean" is a closed-set choice,
but the set has to come from somewhere. A small entity store populated
from what `kubectl` and repo metadata already return: pod, deployment,
namespace, cluster, repo, log stream, with the edges between them and a
`seen_at` on every fact. This is ADR-003 Phase 4 in its smallest useful
form, and it does three jobs, not one:

- **Targeting.** The store supplies candidates; the decision model picks.
  Eight corpus cases, and the original `messaging-whatsapp` complaint.
- **Caller or callee.** "Is the timeout on the caller, the callee, the
  ingress or the database" becomes a walk along the edges plus one
  decision at the end, instead of a generative guess over log text.
  `inv-011-caller-side-timeout` has failed on every model; this is why.
- **Stale state.** A fact with an old `seen_at` is not a current fact. The
  currency gate already does this for notes; the store does it for
  topology, which is where "the deployment has 6 replicas" actually lives.

Grows only from observations the system already makes. No crawler.

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

### 6. Coding, in three parts.

`agent/select.py` asks `satisfies ∈ {yes, no}` per candidate. It has been
the only wired decision all along, running against a backend that did not
exist. After step 0 it works. Everything else coding needs is below.

**6a. A corpus.** *Two days.* Contrastive pairs over diffs: same
instruction, one altered fact in the repository, the correct change flips.
Until this exists every coding claim is aspirational by ADR-002 §5, and the
selector's decisions have nothing to be calibrated against.

**6b. A repository map, and test selection on top of it.** *Three days.*
One indexed pass per repo: symbols, imports, which tests import which
modules, recent change hot spots. Refreshed on change, stored beside the
memory index. Then `run_worktree_tests` runs the tests that cover the files
the diff touched, which is faster and gives the model a smaller failure to
read. The LSP stays for precision; the map is the altitude it lacks. This
is what makes coding good on a repo it has never seen.

**6c. Multi-step tasks and repository memory.** *Three days.* A task that
needs "the model, then the migration, then the endpoint" gets a plan with
checkpoints, each a worktree commit the gate has passed, so the loop can
lose a step without losing the task. And what a task learns about a repo,
conventions, where things live, what its tests are strict about, is
written to the memory store under the repo's name. Today `curate_memory`
runs for ops sessions only. The next task on that repo starts from what
the last one found.

**6d. Acceptance checks before generation, and generation as one option.**
*Three days.* From a review of non-LLM program synthesis (AlphaDev, cvc5
and Rosette, DreamCoder, STOKE, AlphaEvolve). Most of its recommendation
is already the loop's shape: sandbox, gate, deterministic selector, the
model never judging its own diff. Three things it names that are missing:

- **Checks first.** Turn the instruction into executable acceptance checks
  before any candidate exists: a test that must pass, an assertion, a
  grep that must become true. The gate then decides mechanically, the
  selector has a real objective instead of "smallest passing diff", and
  every task yields a contrastive corpus case by construction.
- **Iterate on the gate's findings.** Best-of-k is one round. The gate's
  output becomes the next round's constraint. This is the cheap form of
  generate-search-evaluate, built from parts that exist.
- **Validated transformations as candidates.** A routine that solved a
  task, kept with its validation evidence, offered as a candidate before
  the model is asked. Step 8's memory applied to code; DreamCoder's idea
  without DreamCoder.

Declined, on the review's own evidence: symbolic synthesis as a generator
(fits step 9 when a formal spec exists, never for feature work), algorithm
superoptimizers (no overlap), diffusion models (a model choice, and the
runtime already makes that boring).

### 7. Cognitive state, then predictions on it. *Three days, after 2.*

ADR-003 Phase 2 and 5 together, because the second needs the first.

**State without the transcript.** Round to round, the `assess` node needs
to know what changed, not re-read everything. Observation, Claim,
Hypothesis and Action become records with links, so that "new evidence
this round" is a count and "which claims does this contradict" is a query.
Durable state stops being the transcript plus a list. Transcript size no
longer bounds what the system can hold across rounds.

**Predictions.** A hypothesis states what the next round should observe.
`assess` compares. The residual moves confidence, and a contradicted
hypothesis loses rank without a model being asked to notice. This is the
mechanism by which an investigation knows it is wrong before the harness
says so. Cheap once the records exist and the loop exists.

### 8. Memory that learns from tasks. *Ongoing from step 5.*

The roadmap thesis: MIMIR improves because its organisation accumulates
experience. Four concrete things get written, none of which are today:

- Every decision, with its outcome: the calibration set from step 5, and
  the fine-tuning set for a trained decision model later.
- Every completed session that ended with a verified answer: a corpus case
  draft, with its expected nouns, for a person to accept or reject. The
  corpus grows from use instead of by hand.
- Which tools and which specialist produced the evidence the answer cited,
  per question shape: routing statistics, so the plan in step 2 has history
  to prefer and the specialists table in step 10 has data behind it.
- What a coding task learned about its repo (6c).

Plus the retrieval indexes Phase 8 named and the store lacks: failure
cases ("this shape of question went wrong before, this way") and case
similarity ("the last three times a pod restarted in this namespace it
was this"). Both are lookups over the records above once they exist.

### 9. Exact engines where a model is still doing arithmetic. *Ongoing.*

ADR-003 Phase 6. Each one replaces a generative judgement with a
computation, evaluated independently: a config validator that checks the
repo's manifest against the deployed state (G4's last question, "does the
repository agree with what is running"), a deterministic command
constructor for the families G1 lists, and unit checks over the numbers
in log lines before a model is allowed to reason about them. The rule for
admission is the same as for tier 1 in ADR-004: if it is computable, it
is computed.

### 10. Deploy the mini. *A day.*

7B for prose, 4B Kev for decisions, about 7.5GB resident on a 16GB machine.
The facade behind its key, Warp pointed at it. After steps 1 to 4 the
generative work left is prose, which is what the tier-parity result says a
7B does as well as a 117B.

### 11. Collapse the specialists into a table. *After 1 to 4 and 8.*

With the decisions gone from them and routing statistics behind them,
each is a tool budget, an objective, and a row.

## What "high accuracy" means, in numbers, so it is a finish line

Defined now because the goal is a tool that gets used from Warp, and
"accurate" has to be checkable before it is trusted.

| surface | corpus | target | today |
|---|---|---|---|
| bash and command construction | a command corpus of 50+, half contrastive (same request, one flag or target changed) | 0.95, every run | 37 mixed cases; the 25 rule-decided ones pass every run |
| ops investigation | the 84-case corpus | 0.85 on model cases, pair consistency 0.85 | 0.69 and 0.40 |
| coding | 6a's corpus | 0.85 pass@k through the gate and selector | unmeasured |

Each on the model that will actually serve it: the 7B for command and ops
once the decisions are out of it, the 30B for coding. Three replicates
before any of them is called met.

## What does not move, restated

Safety stays in code. Triage routing stays a rule. Tool choice stays under
constrained decoding at its measured 100%. The arithmetic gates stay. No
rule that is always right is replaced by a model that is usually right, and
the decision model is never asked to write.

## Targets, so the plan can be wrong

| | now | after 0-4 | after 5 | after 6 | after 7-8 |
|---|---|---|---|---|---|
| iteration_limit | 0 | 0 | 0 | 0 | 0 |
| model-case pass | ~40/58 | 46-50 | same | same | 50+ |
| pair consistency | 0.40 | > 0.7 | > 0.85 | same | same |
| generative calls / question | 15-25 | 6-10 | same | same | 4-8 |
| decisions with a usable threshold | 0 | 0 | most | most | all |
| coding pass rate | unmeasured | unmeasured | unmeasured | a number | rising |
| ADR-003 §43 criteria met | 4/12 | 6/12 | 7/12 | 8/12 | 11/12 |

The twelfth criterion, action selection improved from verified transition
history, is what step 8's records make possible and what a later trained
model would deliver. It stays unmet until it is measured.
