"""
HTTP adapter — exposes your env via the BenchAnything four-endpoint protocol.

Local dev:
    python adapter.py
    python adapter.py --port 9000
"""

import argparse
import os
import sys

# Import MyEnv from tooling/env.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tooling"))
from env import MyEnv  # noqa: E402

from bench_common.env_sdk import serve  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print(f"MyEnv adapter → http://{args.host}:{args.port}")
    serve(MyEnv, host=args.host, port=args.port)
