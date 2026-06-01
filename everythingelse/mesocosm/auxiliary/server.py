#!/usr/bin/env python3
"""Standalone server for local development.
Runs the Brain Activation Matching adapter on your machine.

Usage:
    python server.py
    python server.py --port 8765 --host 127.0.0.1
"""

import argparse
import os
import sys

# Ensure the auxiliary directory is on the path
_aux_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _aux_dir)

from adapter import serve
from env import MyEnv

HF_HELP = """
TRIBE v2 needs access to meta-llama/Llama-3.2-3B on HuggingFace.
Before running, set up your HF token:

    export HF_TOKEN=hf_your_token_here

Or login interactively:

    hf auth login

You also need to accept the LLaMA license at:
    https://huggingface.co/meta-llama/Llama-3.2-3B
"""


def main():
    parser = argparse.ArgumentParser(description="Brain Activation Matching — Local Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("WARNING: No HF_TOKEN set." + HF_HELP)

    print(f"Starting Brain Activation Matching server on {args.host}:{args.port}")
    print(f"  Health: http://{args.host}:{args.port}/health")
    print()
    serve(MyEnv, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
