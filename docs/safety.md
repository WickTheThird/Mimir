# Safety model

Implements ADR-001 section 13.

## The core rule

The model proposes. Deterministic policy code decides. Nothing in
`mimir/safety/risk.py` or `mimir/safety/policy.py` consults a model, and a
model's opinion about how risky its own command is never enters the decision.

## Risk classes

| Class | Meaning | Default handling |
| --- | --- | --- |
| R0 | Pure analysis, nothing executes | runs |
| R1 | Read-only (`get`, `describe`, `logs`, `rg`, `git log`) | runs |
| R2 | Elevated inspection (`exec`, port-forward, db session) | asks |
| R3 | Reversible mutation (`rollout restart`, `scale`, bounded `UPDATE`) | asks |
| R4 | High risk (broad delete, `apply`, `DROP`, unbounded `UPDATE`, drain) | asks |

`safety.auto_execute_max_risk` sets the ceiling and defaults to R1.

Classification escalates, never de-escalates, on: production-looking targets,
protected namespaces, wildcard flags (`--all`, `--all-namespaces`, `--force`,
`--grace-period=0`), unresolved targets on a mutating command, and destructive
exec payloads.

## Things that are easy to get wrong, and how they are handled

**Flag values read as subcommands.** `kubectl -n payments get pods` must
classify on `get`. A parser without a table of value-taking flags reads
`payments` as the verb, fails to recognise it, and gates every namespaced read.

**Exec payloads.** `kubectl exec -- rm -rf /data` is not an R2 inspection. The
payload after `--` is classified in its own right, and the binary deny list
applies to it as well as to the outer command.

**Shell wrappers.** `sh -c "..."` is unwrapped and the inner command inspected.

**Shell operators.** No shell is ever spawned. An argv carrying `;`, `|`, `>`,
or backticks is a mistake or an injection attempt, and is refused.

**SQL in flags.** `psql -d billing -c "drop table x"` extracts the statement
from `-c` rather than joining the argv, which would otherwise match no known
prefix and land a `DROP` in the too-low "unrecognised write" branch.

**Unbounded writes.** `UPDATE` or `DELETE` with no `WHERE` is R4, not R3. The
`WHERE` clause is the difference between one row and every row.

**Approval edits.** An operator may edit a command before approving it. The
edited argv is re-classified from scratch, and a result riskier than what was
reviewed is refused rather than executed.

## What is shown before a gated action

Exact command, resolved binary, cluster context, namespace, SDM resource,
database, target objects, purpose, expected effect, risk class with the reasons
that produced it, and the rollback plan (or an explicit statement that there is
none). This is ADR 13.3 and it is rendered identically in the CLI and the UI.

## Secrets

Redaction runs before output is stored, displayed, embedded in a prompt, or
logged. It covers private keys, JWTs, cloud and provider tokens, basic-auth and
database URLs, `key=value` assignments, base64 blobs in Kubernetes secret
output, and the prose form ("my password is ...") that dominates imported chat
transcripts.

The prose rule requires the value to look like a credential, so "the password is
wrong" is left alone. A redactor that mangles ordinary prose is one people
switch off.

MIMIR is not a credential vault (ADR NG3). It uses SDM, kubeconfig, `PGPASSFILE`,
and the OS keychain, and never asks for or stores a password.

## Prompt injection

Repository files, logs, notes, and web pages are data. `wrap_untrusted` fences
them with a restated rule, neutralises forged fences and marker phrases, and
attaches a warning when the injection scanner fires. Retrieved content cannot
grant tools, approve a command, change a classification, or override policy,
because permissions live in policy code that never reads model output.

The scanner is advisory. It is tuned to leave ordinary operational text alone: a
false positive on normal log output would make the warning worthless.

## Network posture

The internet-facing surface is the `/v1` inference facade and `/api/health`.
Every other route refuses non-loopback callers regardless of credentials, so a
valid API key never becomes a remote shell. `X-Forwarded-For` is ignored, since
trusting it would let a remote caller claim to be loopback.

The web helper resolves hostnames before fetching and blocks loopback, private,
link-local, and metadata addresses on every redirect hop. MIMIR runs on a machine
with VPN access to internal systems, so a malicious search result must not be
able to make it fetch an internal endpoint.

## Sandbox

The code runner enforces a separate process, a scrubbed environment with no
inherited credentials, POSIX rlimits, a wall-clock timeout, and a temporary
working directory. The AST check that refuses `socket`, `subprocess`, `eval`,
and write-mode `open` is a guardrail against accidents, not a boundary against a
determined adversary. `src/mimir/tools/sandbox.py` says so in its docstring
rather than implying isolation it does not have.
