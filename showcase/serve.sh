#!/usr/bin/env bash
# Serve the showcase locally
cd "$(dirname "$0")"
PORT=${1:-8080}
echo "Showcase: http://localhost:$PORT"
python3 -m http.server "$PORT"
