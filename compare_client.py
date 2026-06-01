"""Client for the TRIBE v2 Brain Comparison API.
Usage: python compare_client.py <url> "<sentence_a>" "<sentence_b>"

Example:
    python compare_client.py https://removing-hatred-aggregate.ngrok-free.dev "Hello world." "Goodbye world."
"""

import sys, time, json, urllib.request, urllib.error

BASE_URL = sys.argv[1].rstrip("/")
SENTENCE_A = sys.argv[2]
SENTENCE_B = sys.argv[3]

body = json.dumps({"sentence_a": SENTENCE_A, "sentence_b": SENTENCE_B}).encode()
req = urllib.request.Request(
    f"{BASE_URL}/compare",
    data=body,
    headers={"Content-Type": "application/json"},
)
resp = urllib.request.urlopen(req)
data = json.loads(resp.read())
job_id = data["job_id"]
print(f"Job submitted: {job_id}")
print("Waiting for results", end="")

while True:
    time.sleep(5)
    resp = urllib.request.urlopen(f"{BASE_URL}/status/{job_id}")
    status = json.loads(resp.read())
    if status["status"] == "done":
        print("\n\nResults:")
        print(json.dumps(status["result"], indent=2))
        break
    elif status["status"] == "error":
        print(f"\nError: {status['error']}")
        break
    else:
        print(".", end="", flush=True)
