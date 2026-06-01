#!/usr/bin/env bash
# Start the brain server + ngrok tunnel for cloud deployment.
#
# Usage:
#   bash start_tunnel.sh                    # auto port
#   bash start_tunnel.sh --port 8766        # custom port
#
# What it does:
#   1. Sets HF_HOME to project-local cache (model_cache/ — persists forever)
#   2. Starts brain_server.py with TRIBE v2 + Phi-3-mini
#   3. If ngrok is installed, opens a tunnel and prints the URL
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUX_DIR="$SCRIPT_DIR/auxiliary"
VENV_PYTHON="/home/elinzi/Coding/Underlie/mesosphere/bin/python"
PORT=8766

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

# Project-local HF cache — model lives here forever
export HF_HOME="$AUX_DIR/model_cache"
export TRANSFORMERS_CACHE="$AUX_DIR/model_cache/hub"
mkdir -p "$TRANSFORMERS_CACHE"

echo "=== Starting brain server on port $PORT ==="
echo "  Model cache: $HF_HOME"
echo "  Text encoder: microsoft/Phi-3-mini-4k-instruct (downloads ~7GB on first run)"
echo ""

cd "$AUX_DIR"
$VENV_PYTHON brain_server.py --port $PORT --host 127.0.0.1 &
BRAIN_PID=$!
echo "Brain server PID: $BRAIN_PID"

sleep 3
if ! kill -0 $BRAIN_PID 2>/dev/null; then
  echo "Brain server failed to start."
  exit 1
fi

if curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
  echo "Brain server health: OK"
fi

NGROK_URL=""
if command -v ngrok &> /dev/null; then
  echo ""
  echo "=== Starting ngrok tunnel ==="
  ngrok http $PORT --log=stdout > /dev/null &
  NGROK_PID=$!
  sleep 3
  NGROK_URL=$(curl -sf http://127.0.0.1:4040/api/tunnels 2>/dev/null | \
    python3 -c "import sys,json; data=json.load(sys.stdin); print([t['public_url'] for t in data['tunnels'] if t['proto']=='https'][0])" 2>/dev/null || true)
fi

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Brain Activation Server is RUNNING              ║"
echo "║                                                  ║"
echo "║  Sync API:  POST http://127.0.0.1:$PORT/compare_sync  ║"
if [ -n "$NGROK_URL" ]; then
echo "║  Public:    $NGROK_URL  ║"
echo "║                                                  ║"
echo "║  Set: export BRAIN_API_URL=$NGROK_URL  ║"
fi
echo "║                                                  ║"
echo "║  Then run mesocosm:                              ║"
echo "║    BRAIN_API_URL=<url> mesocosm run local ...     ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

cleanup() {
  echo "Shutting down..."
  kill $BRAIN_PID 2>/dev/null || true
  [ -n "$NGROK_PID" ] && kill $NGROK_PID 2>/dev/null || true
  exit 0
}
trap cleanup SIGINT SIGTERM

wait $BRAIN_PID
