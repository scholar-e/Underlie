"""Quick test for the tribev2 API server."""
import subprocess, time, json, urllib.request, sys, os, socket

os.environ["CUDA_VISIBLE_DEVICES"] = ""

s = socket.socket()
s.bind(("", 0))
port = s.getsockname()[1]
s.close()

api_dir = os.path.dirname(os.path.abspath(__file__))
tribev2_dir = os.path.join(api_dir, "..", "everythingelse", "tribev2")

p = subprocess.Popen(
    [sys.executable, "-c", f"""
import sys
sys.path.insert(0, {api_dir!r})
sys.path.insert(0, {tribev2_dir!r})
import os
os.environ["TQDM_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import uvicorn
from tribev2_api import app
uvicorn.run(app, host="127.0.0.1", port={port}, log_level="warning")
"""],
    stderr=subprocess.PIPE,
    stdout=subprocess.PIPE,
)

time.sleep(15)

body = json.dumps({"sentence_a": "Hi.", "sentence_b": "Bye."}).encode()
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/compare",
    data=body,
    headers={"Content-Type": "application/json"},
)
try:
    resp = urllib.request.urlopen(req, timeout=600)
    print("RESULT:", resp.read().decode())
except urllib.error.HTTPError as e:
    print("HTTP ERROR:", e.code, e.read().decode())
    err = p.stderr.read(65536)
    print("STDERR:", err.decode("utf-8", errors="replace")[:5000])
except Exception as e:
    print("FAILED:", e)
finally:
    if p.poll() is None:
        p.terminate()
        p.wait()
