#!/bin/bash
# Debian VM on the mini: MIMIR, its tools, and the tunnel. No models here.
set -euo pipefail
GATEWAY="${GATEWAY:-$(ip route | awk '/default/ {print $3; exit}')}"
REPO="${REPO:-https://github.com/WickTheThird/Mimir.git}"
echo "== deps"
sudo apt-get update -qq && sudo apt-get install -y -qq git ripgrep curl python3 python3-venv >/dev/null
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH")
echo "== MIMIR"
[ -d "$HOME/mimir" ] || git clone -q "$REPO" "$HOME/mimir"
cd "$HOME/mimir" && uv sync -q
mkdir -p "$HOME/.mimir/logs"
sed "s|192.168.64.1|$GATEWAY|g" deploy/mini/vm-config.yaml > "$HOME/.mimir/config.yaml"
echo "== check the host is reachable at $GATEWAY"
curl -sf "http://$GATEWAY:11434/api/tags" >/dev/null && echo "  ollama ok" || echo "  OLLAMA NOT REACHABLE at $GATEWAY (bind it on the host first)"
curl -sf "http://$GATEWAY:8009/v1/models" >/dev/null && echo "  kev ok" || echo "  KEV NOT REACHABLE at $GATEWAY"
echo "== services"
for u in mimir-api mimir-mcp; do
  sed "s|HOME_DIR|$HOME|g" "deploy/mini/vm/$u.service" | sudo tee "/etc/systemd/system/$u.service" >/dev/null
done
sudo systemctl daemon-reload && sudo systemctl enable --now mimir-api mimir-mcp
sleep 4
curl -s http://127.0.0.1:8756/api/health; echo
echo
echo "VM ready. Now, on this VM:"
echo "  ~/mimir/.venv/bin/mimir keys create --label warp-cloud    # paste into Warp's secret store"
echo "  ~/mimir/.venv/bin/mimir keys create --label wick-local    # your password manager"
echo "  sudo systemctl restart mimir-api mimir-mcp               # keys are read at start"
echo "then add the three ai.bumbuindustries.com hostnames to the tunnel (localhost:8756 /v1*, localhost:8010 /mcp*, localhost:8756 /api/health)."
