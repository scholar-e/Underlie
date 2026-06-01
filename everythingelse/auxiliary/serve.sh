#!/usr/bin/env bash
# Brain Activation Matching — UI + direct text comparison
# Usage: bash serve.sh [port]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MESO_DIR="$SCRIPT_DIR/../mesocosm/auxiliary"
VENV_PYTHON="/home/elinzi/Coding/Underlie/mesosphere/bin/python"
PORT=${1:-8080}

export PYTHONPATH="$MESO_DIR"
export HF_HOME="$MESO_DIR/model_cache"

echo "╔══════════════════════════════════════════╗"
echo "║  Brain Activation Matching               ║"
echo "║  http://127.0.0.1:$PORT                   ║"
echo "║                                           ║"
echo "║  Compare two texts directly on the        ║"
echo "║  'Compare' tab — no setup needed.         ║"
echo "║                                           ║"
echo "║  For brain server (GPU recommended):      ║"
echo "║    python brain_server.py --port 8766     ║"
echo "║    Then enter http://127.0.0.1:8766       ║"
echo "║    in the Compare tab's Brain API URL.    ║"
echo "╚══════════════════════════════════════════╝"
echo ""

exec "$VENV_PYTHON" -m ui --port "$PORT" --host "127.0.0.1"
