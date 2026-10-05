#!/bin/zsh
# macOS host on the mini: inference only, bound to the UTM shared-network gateway.
set -euo pipefail
GATEWAY="${GATEWAY:-192.168.64.1}"
echo "== Ollama (bound to $GATEWAY, not the LAN)"
brew list ollama >/dev/null 2>&1 || brew install ollama
launchctl setenv OLLAMA_HOST "$GATEWAY:11434"
brew services restart ollama
sleep 3
OLLAMA_HOST="$GATEWAY:11434" ollama pull qwen2.5:7b
OLLAMA_HOST="$GATEWAY:11434" ollama pull nomic-embed-text
echo "== Kev-0.8B (bf16, ~1.6GB; the 4B does not fit beside the 7B in 16GB)"
[ -d "$HOME/kev" ] || git clone -q https://github.com/jaredpalmer/kev.git "$HOME/kev"
(cd "$HOME/kev" && uv sync --extra serve)
sed -e "s|GATEWAY|$GATEWAY|g" -e "s|HOME_DIR|$HOME|g" "$(dirname "$0")/com.mimir.kev.plist" > "$HOME/Library/LaunchAgents/com.mimir.kev.plist"
mkdir -p "$HOME/.mimir/logs"
launchctl unload "$HOME/Library/LaunchAgents/com.mimir.kev.plist" 2>/dev/null || true
launchctl load "$HOME/Library/LaunchAgents/com.mimir.kev.plist"
echo "== check from the host"
sleep 20
curl -s "http://$GATEWAY:11434/api/tags" | head -c 120; echo
curl -s "http://$GATEWAY:8009/v1/models" | head -c 120; echo
echo "host ready: Ollama and Kev on $GATEWAY. Nothing else of MIMIR runs here."
