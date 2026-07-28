# Roadmap

The governing constraint, and the reason this document exists:

> The biggest risk is no longer that MIMIR cannot do enough. It is becoming a
> large system whose improvements nobody can explain.

Every phase below is gated on being able to answer *why* a version outperformed
the one before it. Features that cannot be measured wait.

## Phase 1: scientific foundation

**Goal:** every improvement is measurable. This phase ends when "why did 0.8
beat 0.7?" has an answer backed by a stored run.

| Item | State |
| --- | --- |
| Evaluation harness | done |
| Regression corpus | done |
| Failure taxonomy | done |
| Offline evaluation by default | done |
| Stored run ids | done |
| Run provenance | done |
| Run comparison with confound detection | done |
| Held-out corpus support | done |
| Capability matrix | built, never run |
| Model benchmarking | pending Baseline B |
| Claim-level unsupported-claim scoring | deferred |

Claim-level scoring is deliberately deferred. The current metric is
session-level, which is enough to compare models and enough for a first matrix.
Per-claim attribution is worth building once there are enough runs to justify
the implementation cost, and not before.

**Do not add specialists in this phase.** Every specialist added from here
should justify itself by moving a benchmark number.

## Phase 2: design department

The pipeline today is roughly question, plan, act. It should become:

```
requirements
  -> current system reconstruction
  -> architecture options
  -> tradeoff analysis
  -> implementation DAG
  -> execution
```

The point is not better code generation. It is making bad architecture hard to
reach before trying to make implementation better.

## Phase 3: repository knowledge as organisational memory

Not a repository visualiser. A graph that connects requirement to workflow to
files to symbols to tests to evidence to ADR to deployment, so that a claim
about the system can always be traced to the thing that supports it.

The target: know a repository better after six months than a new senior engineer
joining the team.

## Phase 4: organisational structure

Departments, project management, technical lead, specialists, QA, release. Built
only once there are measurements, so the specialists solve observed problems
rather than imagined ones.

## Phase 5: model and assignment optimisation

Role-to-model assignment driven by the capability matrix. Assignments, not
models in isolation, because a strong planner paired with a weak synthesiser can
lose to two mediocre models that agree on format.

Scaffolding exists (`mimir eval matrix`). It stays unrun until Baseline B gives
it something to compare against.

## Phase 6: organisational learning

Only after hundreds of evaluated runs. Technical director, staffing
optimisation, specialist retirement, routing improvements, prompt evolution.

The distinguishing idea: optimise the *process*, not the answer. A report that
says "migrations required repair 31% of the time, root cause is that the
migration specialist skips lock analysis, recommend inserting a lock analysis
step" is redesigning the organisation rather than tuning a prompt.

Before enough data exists, such a report is a confident guess, which is the
failure mode this whole system is built to avoid.

## Version milestones

| Version | Contents |
| --- | --- |
| v0.5 | Evaluation complete, regression corpus, offline mode, run ids |
| v0.6 | Design department, architecture generation, system reconstruction |
| v0.7 | Repository knowledge graph, evidence graph, workflow graph |
| v0.8 | Company structure: PM, tech lead, specialists |
| v0.9 | Autonomous implementation, git worktrees, review, release candidate |
| v1.0 | The first version worth trusting on a real repository |

v0.5 is complete except for the capability matrix run.

## Architecture direction

Components plug into a core rather than into the graph directly:

```
Core        LangGraph, policy engine, evaluation, memory,
            repository knowledge, tool runtime
Company     proposal, planning, engineering, QA, security, release, delivery
Models      local, remote, router, reviewer
```

## What to avoid

**Chasing models.** If MIMIR is coupled to whichever model is fashionable this
month, the work becomes rewriting adapters instead of improving the product.
Swapping a model should be boring. The runtime layer is already model-agnostic
(ADR-001 section 25 keeps it that way); the discipline is to keep it so.

**Adding capability faster than measurement.** A specialist that cannot be shown
to improve a benchmark is a liability, because it enlarges the system while
making its behaviour harder to explain.

## The thesis

Foundation models improve because vendors retrain them. MIMIR should improve
because its organisation accumulates experience: every completed task leaves
behind a regression case, routing statistics, repository knowledge, a better
procedure, or a corpus entry.

That is a different source of progress, and it is the part worth protecting.
