# MIMIR on the Mac mini

The always-on ops tier (plan step 10). Three services under launchd, one
config, reachable from Warp over the tunnel described in
`../cloudflared/README.md`.

| service | what | port | memory |
|---|---|---|---|
| `com.mimir.kev` | Kev-0.8B decision model, Qwen3 revision, bf16 | 8009 | ~1.6GB |
| `com.mimir.api` | OpenAI-compatible facade, key-authenticated | 8000 | shares Ollama |
| `com.mimir.mcp` | MIMIR's capabilities as MCP tools for Warp (streamable HTTP) | 8010 | shares Ollama |

Ollama holds `qwen2.5:7b` (4.7GB, 8k context) and `nomic-embed-text`. Total
resident about 11GB on 16GB. Kev-4B is ~8GB in bf16 and does not fit beside
the 7B on this machine; 0.8B costs about 0.05 accuracy on Kev's own suite,
and computed facts and the margin floor come before it everywhere.

## Install

```bash
# once: the model tier
ollama pull qwen2.5:7b && ollama pull nomic-embed-text
git clone https://github.com/jaredpalmer/kev.git ~/Documents/PERS/kev && (cd ~/Documents/PERS/kev && uv sync --extra serve)

# config and keys
mkdir -p ~/.mimir/logs
cp deploy/mini/config.yaml ~/.mimir/config.yaml
mimir keys add warp

# services
for s in kev api mcp; do
  cp deploy/mini/com.mimir.$s.plist ~/Library/LaunchAgents/
  launchctl load ~/Library/LaunchAgents/com.mimir.$s.plist
done
```

Check: `curl -s localhost:8009/v1/models`, `curl -s localhost:8000/v1/models -H "Authorization: Bearer <key>"`.

## Point Warp at it

The facade gives Warp a model. The MCP server gives Warp MIMIR. Configure
both: the facade as the BYOK model per `docs/warp.md`, and the MCP server
as an MCP source at `http://<mini>:8010/mcp` (through the tunnel when away
from home). `construct_command` is the one to try first.

## What does not run here

Coding. The coding corpus is measured on the laptop's 30B and has no number
on a 7B; until it does, `code_task` on the mini is a preview.
