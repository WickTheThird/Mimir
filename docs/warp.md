# Warp integration

Implements ADR-001 section 16. Setup lives in
[../deploy/cloudflared/README.md](../deploy/cloudflared/README.md); this covers
what the integration is and is not.

## What it is

Warp is configured with MIMIR's authenticated OpenAI-compatible HTTPS endpoint as
a custom BYOK model. Warp keeps its own terminal UX and its own agent loop, and
gets its model responses from the local machine.

The commands Warp then runs execute locally, so they use your VPN, internal DNS,
kubeconfig, SDM authentication, shell environment, and repository checkouts
(ADR 16.3). The endpoint only supplies text.

## What it is not

The facade is a model gateway, not a nested agent. ADR 16.4 gives the reasons and
they are all real: Warp already owns the outer loop, a nested graph duplicates
planning, tool schemas conflict, approvals become ambiguous, latency and context
use grow, and the same command can be proposed or executed twice.

So the facade adds MIMIR's terminal conventions to the system prompt and,
optionally, read-only memory retrieval. It stops there.

## What the facade adds

The appended system prompt asks the model to build commands as argument vectors,
state the cluster context and namespace a command targets, never assume a default
namespace, say plainly when a command changes state and what the rollback is, and
admit uncertainty about a flag rather than inventing it.

Memory retrieval is off by default. Send `"mimir_memory": true` in the request
body to try it on a single call.

## Security

A valid API key buys model responses and nothing else. Every privileged route
refuses non-loopback callers regardless of credentials, and `X-Forwarded-For` is
ignored so it cannot be used to claim loopback. Rate limiting is per key inside
MIMIR, with an optional Cloudflare rule in front.

If you want a second factor ahead of the key, put Cloudflare Access on the
hostname.

## Going further

The ADR's preferred extension is exposing specialist workflows as explicit MCP
tools that Warp invokes deliberately, with privileged tools staying local. That
keeps one agent loop in charge while making real MIMIR capability reachable. It
is not built yet, and nesting the graph behind the model endpoint is the thing to
avoid in the meantime.

## Caveats

Availability depends on the Mac being awake and on the network, the tunnel
running, and enough free memory for the model (ADR 22.3). There is no SLA. Warp
cloud-agent runs have different limitations from interactive use; interactive use
is what this targets.
