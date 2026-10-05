# ai.bumbuindustries.com

The concrete plan for this domain, on top of `README.md` in this folder.

## Shape

```
Warp / Hermes / browser
   -> https://ai.bumbuindustries.com   (Cloudflare edge: TLS, rate limit, optional Access)
   -> cloudflared on the Mac mini      (outbound tunnel; no inbound port)
      /v1/*   -> 127.0.0.1:8756  MIMIR facade (model gateway, key-authenticated)
      /mcp/*  -> 127.0.0.1:8010  MIMIR MCP server (three tools, key-authenticated)
      /api/health -> 127.0.0.1:8756 (no key)
      everything else -> 404 at the edge
```

The mini already runs a tunnel for `api.bumbuindustries.com`. Either add
`ai.` as a second hostname on that tunnel, or create a second tunnel; both
work. One tunnel with two hostnames is simpler to keep alive.

## On the mini, once

```bash
# services (see ../mini/README.md): Ollama with qwen2.5:7b, Kev-4B, facade, MCP
# a key for Warp; it is shown once
mimir keys create --label warp

# the hostname
cloudflared tunnel route dns <existing-tunnel-name> ai.bumbuindustries.com
```

Then merge the `ingress` block from `config.yml` into the mini's existing
`~/.cloudflared/config.yml` with `hostname: ai.bumbuindustries.com` on the
three MIMIR rules, keeping the `api.` rules and the final 404 as they are.
`cloudflared tunnel ingress validate`, then restart cloudflared.

## Check from anywhere

```bash
curl -s https://ai.bumbuindustries.com/api/health
curl -s https://ai.bumbuindustries.com/v1/models -H "Authorization: Bearer <key>"
curl -s https://ai.bumbuindustries.com/mcp -H "Authorization: Bearer <key>"   # 401 without the key
```

## Warp, two things

1. **Model**: Warp settings, custom OpenAI-compatible endpoint,
   `https://ai.bumbuindustries.com/v1`, the key, model `mimir-local-ops`.
   This gives Warp the 7B with MIMIR's terminal conventions. Nothing more.
2. **Tools**: Warp MCP source, `https://ai.bumbuindustries.com/mcp`, same
   key as a Bearer header. This gives Warp `construct_command`,
   `investigate` and `code_task`, each running the full path with every
   gate. This is the one that makes it MIMIR rather than a model.

## What a leaked key can and cannot do

A key reaches the facade and the three MCP tools. The MCP tools can run
read-only investigations and make changes in task worktrees on the mini.
They cannot execute a mutating command (the policy engine requires an
operator's approval, which a remote caller cannot give), cannot push, and
cannot reach shell, SDM, Kubernetes or database helpers directly: those
refuse non-loopback callers in code regardless of credentials. Rotate with
`mimir keys create` and remove the old key from `~/.mimir/config.yaml`.

Put Cloudflare Access in front of the hostname if you want a second factor;
the hostname is guessable.
