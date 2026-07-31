# MIMIR

Local-first operations investigation and on-call assistance platform.

MIMIR runs on your own machine, uses a local model runtime, and helps with the
terminal-centric parts of on-call work: constructing commands, searching
repositories, verifying that a behaviour actually exists, inspecting Kubernetes
and SDM-mediated environments, reading logs, and diagnosing timeouts with
evidence rather than vibes.

It is not primarily a coding agent. See [docs/ADR-001.md](docs/ADR-001.md) for
the architecture decision record this implementation follows.

## Quick start

```bash
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -e ".[dev,search]"

# One-time setup: writes ~/.mimir/config.yaml and seeds the knowledge base.
mimir init

# Point MIMIR at a local model runtime (Ollama shown here).
ollama serve &
ollama pull qwen2.5:32b
mimir models set deep --model qwen2.5:32b

# Check that everything the ADR needs is reachable.
mimir doctor

# Ask something.
mimir command "show pods in payments that restarted in the last hour"
mimir investigate "why is checkout timing out against the auth service?"
mimir repo ask "does this service fail open when auth times out?" --repo billing
```

Start the API and web UI:

```bash
mimir serve                 # FastAPI on 127.0.0.1:8756
cd web && npm install && npm run dev   # Vite UI on 127.0.0.1:5173
```

## What is in here

| Area | Path | ADR section |
| --- | --- | --- |
| Configuration | `src/mimir/config.py` | 18, 19, 22, 25 |
| Domain models, evidence, state | `src/mimir/models/` | 12 |
| Model runtimes and routing | `src/mimir/llm/` | 6.2 C3, 18 |
| Safety, risk classes, approvals | `src/mimir/safety/` | 13 |
| Typed helper tools | `src/mimir/tools/` | 8, 9 |
| Knowledge and memory | `src/mimir/knowledge/` | 11 |
| Skills (Agent Skills format) | `src/mimir/skills/` | 10 |
| Lifecycle hooks | `src/mimir/hooks/` | 10.4 |
| LangGraph orchestration | `src/mimir/graph/` | 6.2 C2 |
| Council of specialists | `src/mimir/council/` | 7 |
| Persistence | `src/mimir/persistence/` | 19 |
| HTTP API and OpenAI facade | `src/mimir/api/` | 6.2 C1/C9, 16 |
| CLI | `src/mimir/cli/` | 14 |
| Web UI | `web/` | 15 |
| Evaluation harness | `src/mimir/eval/` | 21 |
| Deployment assets | `deploy/` | 22 |

Longer documentation lives in [docs/](docs/):

- [docs/architecture.md](docs/architecture.md)
- [docs/safety.md](docs/safety.md)
- [docs/skills.md](docs/skills.md)
- [docs/memory.md](docs/memory.md)
- [docs/warp.md](docs/warp.md)
- [docs/operations.md](docs/operations.md)
- [docs/evaluation.md](docs/evaluation.md)
- [docs/monitor.md](docs/monitor.md)

## Safety posture

- The model proposes; deterministic policy code decides (ADR 13.1).
- Read-only work (risk class R1) runs without prompting. Anything that mutates
  state stops for an explicit approval with a rendered impact summary.
- Retrieved content (files, logs, web pages) is wrapped as untrusted data and
  cannot grant tools, approve commands, or change risk classification.
- The internet-facing surface is the authenticated OpenAI-compatible inference
  facade only. Shell, Kubernetes, SDM, database, and filesystem helpers stay
  bound to loopback.

## Status

Phases 0 through 7 of the ADR are implemented. Section 25 open decisions are
kept as configuration rather than baked in: model, runtime, database, search
provider, and browser implementation are all swappable.
