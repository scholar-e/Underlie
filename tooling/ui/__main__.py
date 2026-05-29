#!/usr/bin/env python3
"""Local Testing UI — python -m tooling.ui"""

import argparse
import os
import socket
import sys

# Ensure both tooling/ and tooling/ui/ are importable
_ui_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ui_dir, ".."))
sys.path.insert(0, _ui_dir)

from server import app
import uvicorn


def find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main():
    parser = argparse.ArgumentParser(description="Local Testing UI for Kaggle Prediction Benchmark")
    parser.add_argument("--port", type=int, default=0, help="Port for the UI server (default: auto-select)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    args = parser.parse_args()

    port = args.port or find_free_port()

    print("╔══════════════════════════════════════════╗")
    print("║   Local Testing UI                       ║")
    print(f"║   http://{args.host}:{port}                  ║")
    print("║   Kaggle Prediction Benchmark            ║")
    print("╚══════════════════════════════════════════╝")
    print()
    print("Open the URL in your browser to configure and run the benchmark.")
    print("Close with Ctrl+C.\n")

    uvicorn.run(app, host=args.host, port=port, log_level="info")


if __name__ == "__main__":
    main()
