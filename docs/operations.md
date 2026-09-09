# Operating MIMIR

## Daily use

```bash
mimir                                   # interactive
mimir investigate "why is checkout timing out against auth?"
mimir command "show pods in payments that restarted in the last hour"
mimir repo ask "does this fail open when auth times out?" --repo billing
mimir repo flow "HandleCheckout" --repo billing
mimir k8s investigate payments-api -n payments -c staging
mimir sdm investigate <resource> --target <container>
kubectl -n payments logs deploy/api --since 1h | mimir logs
mimir research "what changed in the Go 1.23 http client timeouts"
```

Sessions:

```bash
mimir session list
mimir session show <id>
mimir session resume <id>
mimir export <id> --format md -o handoff.md
```

## Interactive commands

`/help` `/context` `/ns` `/cluster` `/repo` `/evidence` `/commands`
`/hypotheses` `/skills` `/session` `/export` `/new` `/quit`

Code and cluster work:

`/code <task> [repo]` opens a task worktree and works in it: reading, editing
and running tests in a loop until the task is done. `/diff` shows what it
changed, `/why` how it got there, `/done` leaves. A request that already names
its target and its action, such as "get the logs for deployment/api in
namespace payments", runs the same kind of loop against the cluster without
being asked: there is nothing to investigate in an instruction.

While a loop runs, the trail of what it looked for appears beside the
transcript. The live region cannot scroll, so it shows the last screenful and
the whole transcript is printed again underneath when the turn ends. `/panel`
turns it off; terminals under 104 columns do not get it at all.

`/terms` shows what your words have resolved to before, `/sessions` lists past
conversations and `/open <id>` reads one back.

What is loaded right now, read from live state rather than from config:

`/status` tools offered, model and the context actually served, language
servers, skills. `/tools [capability]` the tool surface with its risk class.
`/lsp` per language: ready, or the exact command that installs it. `/model`
every routing role with its digest and served context, in red when the
runtime is serving a different context than the one configured. `/worktree`
task worktrees, and `/worktree diff <task>` for what a task has changed.

Follow-up questions reuse the evidence already gathered, so asking "and what
about the callee side?" does not re-run the same commands.

## Approvals

Read-only work runs without prompting. Anything above the ceiling stops and
shows the ADR 13.3 display. Choose approve, reject, edit, or explain. `explain`
prints which policy rules fired and why, which is worth using the first few times
so the classifier's behaviour is not a mystery.

In a pipeline with no TTY, gated commands are refused rather than run unattended.
An approval nobody can answer must not become a yes.

## Configuration

`~/.mimir/config.yaml`. `mimir init` writes a commented starting point.

The settings worth understanding:

- `safety.auto_execute_max_risk` (default R1). Raising it to R3 or R4 means
  MIMIR changes live state without asking. `mimir doctor` reports that as a
  blocking problem, not a warning.
- `safety.production_context_patterns`. Anything matching in a context,
  namespace, or resource name never auto-executes.
- `kubernetes.allowed_contexts` / `denied_contexts`. Empty allow list means all.
- `models.routing`. Point a task class at a faster profile if command completion
  feels slow (ADR R6).

## Health

```bash
mimir doctor
```

Reports config, binaries, repositories, every model profile, tool count,
knowledge and skills, persistence including WAL mode, the safety ceiling, and
network exposure. A missing `kubectl` is reported as a skipped capability, not a
failure; an unreachable model is a blocking failure.

## Troubleshooting

**Model unreachable.** `mimir doctor` names the profile and URL. Start the
runtime, pull the model, or repoint with `mimir models set deep --model X`.

**Structured output keeps failing.** Small models struggle to emit valid JSON.
The router already repairs fenced blocks, trailing commas, and Python literals,
and retries once with the validation error. If it still fails, route
`classification` and `final_synthesis` at a larger profile.

**Everything asks for approval.** The command's target is probably unresolved.
Pass `-n` and `--context` explicitly, or set defaults in config. A mutating
command with no resolved target is R4 by design.

**Repository search is slow.** Install ripgrep. Without it the helpers fall back
to a Python walk, which works but is markedly slower on large trees.

**"database is locked".** Should not happen; WAL is enabled at connect time.
`mimir doctor` shows the journal mode. If it says `delete`, something replaced
the connection setup.

## Data on disk

```
~/.mimir/
  config.yaml          configuration
  mimir.db             sessions, commands, evidence, audit
  checkpoints.sqlite   LangGraph checkpoints
  artifacts/           full command output and fetched documents
  knowledge/           the memory base
  logs/                when running under launchd
```

Command output lives in `artifacts/`, not in the database, so the audit trail
stays queryable while large outputs stay on disk. Retention defaults to keeping
everything; ADR 19.3 leaves the policy open.
