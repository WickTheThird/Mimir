# Running MIMIR under launchd (ADR 22.2)

The ADR leaves process management open and asks for "the simplest reliable
approach after the runtime is proven". On a single Apple Silicon Mac that is
launchd: it starts at login, restarts on crash, and needs no daemon of its own.

## Install

```bash
mkdir -p ~/.mimir/logs
sed "s|CHANGE_ME|$USER|g" deploy/launchd/com.mimir.api.plist > ~/Library/LaunchAgents/com.mimir.api.plist
launchctl load ~/Library/LaunchAgents/com.mimir.api.plist
```

Check it came up:

```bash
launchctl list | grep mimir
curl -s http://127.0.0.1:8756/api/health
```

## The model runtime

MIMIR does not manage the model runtime, because the ADR keeps the runtime
swappable (section 25). Run yours however you prefer. For Ollama:

```bash
brew services start ollama
```

For an MLX server, write a second plist wrapping:

```bash
python -m mlx_lm.server --model <hf-repo> --port 8080
```

then point a profile at `http://127.0.0.1:8080/v1` with `mimir models set`.

## Uninstall

```bash
launchctl unload ~/Library/LaunchAgents/com.mimir.api.plist
rm ~/Library/LaunchAgents/com.mimir.api.plist
```

## Availability

ADR 22.3 is honest that Warp integration depends on the Mac being on, the
network being up, the tunnel running, and enough free memory. There is no
availability SLA. If the Mac sleeps, Warp requests fail until it wakes. To keep
it awake while on mains power:

```bash
sudo pmset -c sleep 0
```
