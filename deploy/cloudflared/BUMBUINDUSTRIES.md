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

## Keys

Two keys, created on the machine that serves, shown once, stored as hashes:

```bash
mimir keys create --label warp-cloud    # lives only in Warp's secret store
mimir keys create --label wick-local    # lives only in your password manager
mimir keys list
mimir keys revoke warp-cloud            # the other keeps working
```

Each is 32 random bytes. Both the facade and the MCP server check the same
store, on every route including streams; a revoked key fails everywhere at
once. Requests over 1MB are refused and each key is rate limited.

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

## The split: inference on the host, everything else in the UTM VM

Of the exposed tools, `code_task` runs a repository's tests, which is
arbitrary code on whichever machine hosts MIMIR. That machine is the VM.

```
internet -> Cloudflare -> cloudflared (VM)
                            -> MIMIR facade :8756 + MCP :8010 + tools   (VM)
                                 -> Ollama :11434   (macOS host, Metal)
                                 -> Kev    :8009    (macOS host, Metal)
```

The host runs the two inference servers and nothing else of MIMIR's, bound
to the UTM virtual network, not the LAN. Text in, text out; neither
executes anything. Tools, worktrees, test execution and the tunnel live in
the VM, which can be snapshotted, capped and discarded.

### Host (macOS)

```bash
# Ollama listens on the UTM shared-network gateway only (not the LAN)
launchctl setenv OLLAMA_HOST 192.168.64.1:11434 && brew services restart ollama
# Kev the same
KEV_DTYPE=bf16 uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b@qwen3 --host 192.168.64.1 --port 8009
```

Memory: 7B + Kev-0.8B + macOS, about 11GB. The VM takes the rest.

### VM (Debian, 3GB, no models)

MIMIR installed as on any Linux box, with `~/.mimir/config.yaml` from
`../mini/vm-config.yaml`: every profile's `base_url` and `decisions.base_url`
point at the host gateway. The existing cloudflared adds the three
`ai.bumbuindustries.com` hostnames pointing at `localhost` inside the VM.
`investigate` sees whatever kubeconfig and VPN the VM has; `code_task`
works on repositories cloned into the VM.

### What a compromised VM reaches

Ollama and Kev as text APIs, and nothing else on the host. Put Cloudflare
Access in front of the hostname for a second factor, and keep the VM's
snapshot current.
