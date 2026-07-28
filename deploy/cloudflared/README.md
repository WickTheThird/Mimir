# Exposing MIMIR to Warp (ADR 16)

Warp's inference path runs server side, so the endpoint has to be reachable from
the internet. A Cloudflare Tunnel does that without opening a port on the Mac or
exposing anything but the inference facade.

## What is exposed, and what is not

Exposed: `POST /v1/chat/completions`, `GET /v1/models`, `GET /api/health`.

Not exposed, and refused in code as well as at the edge (ADR 16.5, NG5):
shell execution, SDM, Kubernetes, database access, repository filesystem access,
and the LangGraph administrative surface. A valid API key buys model responses.
It does not buy a remote shell on your machine. That separation is enforced by
`mimir.api.auth`, which rejects the privileged routes for any non-loopback
caller regardless of credentials, so a misconfigured ingress rule cannot open
them.

## Setup

```bash
brew install cloudflared
cloudflared tunnel login
cloudflared tunnel create mimir
# note the tunnel id it prints

cloudflared tunnel route dns mimir mimir.example.com

# fill in the placeholders
sed -e "s|CHANGE_ME_TUNNEL_ID|<id>|g" -e "s|CHANGE_ME|$USER|g" \
    deploy/cloudflared/config.yml > ~/.cloudflared/config.yml

cloudflared tunnel run mimir
```

Generate a key and check it end to end:

```bash
mimir keys create --label warp
curl -s https://mimir.example.com/v1/models -H "Authorization: Bearer <key>"
```

## Configure Warp

In Warp's settings, add a custom OpenAI-compatible inference endpoint:

| Field | Value |
| --- | --- |
| Endpoint name | MIMIR |
| Endpoint URL | `https://mimir.example.com/v1` |
| API key | the key from `mimir keys create` |
| Model name | `mimir-local-ops` |
| Model alias | MIMIR Local Ops |

Warp then executes commands locally on your machine, so the commands it runs
still use your VPN, internal DNS, kubeconfig, SDM authentication, and shell
environment (ADR 16.3). The endpoint only supplies the model's text.

## Why the endpoint is a gateway, not an agent

ADR 16.4. Warp already owns the outer agent loop. Putting MIMIR's LangGraph
agent behind Warp's model endpoint would duplicate planning, conflict on tool
schemas, make approvals ambiguous, and risk the same command being proposed or
executed twice. So the facade adds MIMIR's terminal conventions and, optionally,
read-only memory retrieval, and stops there.

To pull real MIMIR workflows into Warp later, expose specialists as explicit MCP
tools rather than nesting the graph. Privileged tools stay local; Warp invokes a
specialist deliberately.

`mimir_memory: true` in the request body turns on memory retrieval for a single
call if you want to try it without changing configuration.

## Hardening

Cloudflare Access in front of the hostname adds a second authentication factor
ahead of the API key, and is worth enabling if the hostname is guessable. Rate
limiting is applied per key inside MIMIR (`api.facade_rate_limit_per_minute`),
and a Cloudflare rate-limiting rule on the hostname adds an edge-level cap so
abusive traffic never reaches the Mac at all (ADR R8).
