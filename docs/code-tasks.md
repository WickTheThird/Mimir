# Code tasks

Six tools implement ADR-002 section 3: `create_task_worktree`,
`write_worktree_file`, `diff_task_worktree`, `run_worktree_tests`,
`list_task_worktrees`, `discard_task_worktree`.

## MIMIR never writes to your checkout

Every change happens in a git worktree created for one task, living under
`$MIMIR_HOME/worktrees`. That buys three properties without any cleverness:

- every change is reversible by deleting a directory;
- every change is reviewable as a diff against a known base commit;
- an abandoned task leaves the operator's checkout untouched.

```
create_task_worktree   -> mimir/fix-retry-bounds at ~/.mimir/worktrees/repo-fix-retry-bounds
write_worktree_file    -> created src/queue/retry.py (24 lines)
diff_task_worktree     -> 1 file changed, 24 insertions
run_worktree_tests     -> passed (exit 0): pytest tests/test_retry.py -q
discard_task_worktree  -> discarded mimir/fix-retry-bounds and everything in it
```

## Containment is enforced by resolution, not by convention

`resolve_inside` resolves the path **and then** checks it is under the worktree
root. Symlinks are followed first, because a boundary compared as a string
prefix is crossed by a symlink pointing outward, and that is not theoretical in
a repository MIMIR did not write.

```
'../../../tmp/escape.txt' resolves to /Users/x/tmp/escape.txt,
outside the task worktree at /Users/x/.mimir/worktrees/repo-task.
Writing outside the worktree requires explicit approval (ADR-002 3.3).
```

## Running tests is an escalation

| Operation | Risk |
| --- | --- |
| `list_task_worktrees` | R0 |
| `create_task_worktree`, `write_worktree_file`, `diff_task_worktree`, `discard_task_worktree` | R1 |
| `run_worktree_tests` | **R2** |

Writing a file inside the worktree is R1. *Running* the repository's tests is
R2, because it executes arbitrary code from that repository including whatever
MIMIR just wrote, and anything a dependency's fixtures decide to do. ADR-002
section 4.1 calls this elevated inspection rather than a read, and the risk
table encodes it literally.

## Scrubbed, not stripped

Test processes inherit the environment minus credentials, matched both by name
(`KUBECONFIG`, `AWS_SECRET_ACCESS_KEY`, `GH_TOKEN`, `SSH_AUTH_SOCK`, ...) and
by shape (any variable whose name contains `TOKEN`, `SECRET`, `PASSWORD`,
`CREDENTIAL`, `API_KEY`). Verified: zero credential variables are visible to a
test process.

The first version rebuilt `PATH` from scratch instead, which also removed the
project's own toolchain and made every test command exit 127. The environment
was perfectly safe and completely useless. Credentials are a shape; the
toolchain is not.

## Not yet permitted

Per ADR-002 sections 3.3 and 3.4, and deliberately unimplemented: committing,
pushing, history rewriting, writing outside a worktree, touching CI config or
lock files, adding dependencies. Those need approval paths that do not exist
yet, and absent code cannot be talked into running.


## Rules

Every write is checked before it is allowed to stand. A model writing code has
to hold four things at once: the rules of the language, the rules of the
framework, the rules of this codebase, and the task. A small local model gets
one of them wrong regularly, and the first three do not need a model to check.

Three checks, in order:

1. **Syntax.** Does the file parse. Python and JSON in-process, instantly. This
   is the only check that reverts: a file that does not parse is not a partial
   change, it is a broken one, and every later read of it returns nonsense.
2. **Project rules.** Invariants no compiler knows: side effects must be
   awaited in this runtime, money crosses the wire in minor units, this column
   is NOT NULL. Written as a pattern with a reason, in
   `~/.mimir/rules/*.yaml` or `<repo>/.mimir/rules/*.yaml`.
3. **The language server.** New errors only, measured against the diagnostics
   before the change, so a file that was already failing is not blamed on the
   edit that touched it.

Anything but a syntax error is reported and left in place. An edit that
introduces a type error may be the first half of a change the next step
completes, and reverting it would stop the loop working in two steps.

The gate can only block on evidence. No language server for a file means no
diagnostics check, not a failed one.

A rule is data, not code:

```yaml
- id: awaited-side-effects
  title: A side effect in a Workers handler must be awaited
  why: >
    Workers terminate the instant fetch() returns, so an un-awaited notify is
    aborted mid-flight.
  paths: ["*.js", "*.ts"]
  forbid: '^(?!.*await).*\b(sendEmail|notify)\s*\('
  severity: error
```

Recording what an incident taught should not need a code change and a release,
which is the reliable way to ensure nobody records it.


## Sampling, and what the rules can decide

Measured on two tasks at temperature 0, four attempts each.

Adding a method: 4 of 4 correct, three distinct but equivalent diffs. There is
no variance to exploit and sampling more than once buys nothing.

Introducing a constant and making a method use it: 1 of 4 correct. The others
defined the constant twice, defined it four times, or added it and never used
it. Every one of them passed syntax, the linter and the tests, so before the
definition check the gate scored 4 of 4 and could not tell them apart. Worse,
the incomplete attempts have the smallest diffs, so a selector preferring small
diffs would have chosen a wrong one deliberately.

With the definition and deletion checks in place the gate marks exactly one
attempt clean, and it is the correct one. That is the case for sampling: not
that attempts fail outright, which on an easy task they do not, but that on a
task with any depth most of them finish incompletely in ways that look fine.

The checks that separate them are all facts, not judgements: a module level name
defined and never read did not connect to anything; a name defined twice is one
definition too many; a change that removes three times what it adds is either a
refactor that was asked for or an accident.
