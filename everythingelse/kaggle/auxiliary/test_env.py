#!/usr/bin/env python3
"""Test the Kaggle Prediction Benchmark env via the running adapter.

Usage:
    python3 auxiliary/test_env.py
    python3 auxiliary/test_env.py --url http://localhost:8765
"""

import argparse
import json
import sys
import urllib.request
import urllib.error

BASE = "http://localhost:8765"
PASS = 0
FAIL = 0


def post(endpoint: str, body: dict) -> dict:
    url = f"{BASE}{endpoint}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode()
        try:
            return json.loads(detail)
        except json.JSONDecodeError:
            return {"error": detail, "status_code": e.code}
    except Exception as e:
        return {"error": str(e)}


def get(endpoint: str) -> dict:
    url = f"{BASE}{endpoint}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode()
        try:
            return json.loads(detail)
        except json.JSONDecodeError:
            return {"error": detail, "status_code": e.code}
    except Exception as e:
        return {"error": str(e)}


def check(name: str, got: dict, key: str = None, expected=None, cond=None):
    global PASS, FAIL
    ok = False
    val = None
    if cond:
        ok = cond(got)
    elif key is not None and expected is not None:
        val = got.get(key)
        ok = val == expected
    elif key is not None:
        ok = key in got
        val = got.get(key, "MISSING")
    else:
        ok = bool(got)

    status = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1

    detail = f"  → got {val!r}" if not ok else ""
    print(f"  [{status}] {name}{detail}")
    return ok


# -- Helpers to extract observation from reset vs step responses -----------

def reset_obs(resp: dict) -> dict:
    return resp.get("data", {})


def step_obs(resp: dict) -> dict:
    obs = resp.get("observation", {})
    if isinstance(obs, dict) and "data" in obs:
        return obs["data"]
    return obs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health():
    print("\n=== Health ===")
    resp = get("/health")
    check("adapter is running", resp, "status", "ok")


def test_reset_toy_data():
    print("\n=== Reset with toy data ===")
    resp = post("/reset", {
        "episode_id": "test-reset-1",
        "seed": 42,
        "scenario_params": {"toy_data": "1"},
    })
    obs = reset_obs(resp)
    check("returned data_dir", obs, "data_dir")
    check("returned work_dir", obs, "work_dir")
    check("returned target_column", obs, "target_column")
    check("returned feature_columns", obs, "feature_columns")
    check("returned data_preview", obs, "data_preview")
    check("returned is_classification", obs, "is_classification")
    check("step == 0", obs, "step", 0)
    check("max_steps == 5", obs, "max_steps", 5)
    check("best_score == 0.0", obs, "best_score", 0.0)
    check("feature count == 2", obs, "feature_columns",
          cond=lambda d: len(d.get("feature_columns", [])) == 2)
    check("is_classification is False (regression)", obs, "is_classification", False)
    return obs


def test_valid_program(episode_id: str):
    print("\n=== Valid program (OLS regression) ===")
    program = """
import pandas as pd
import numpy as np

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

Xt = np.c_[np.ones(train.shape[0]), train[["x1", "x2"]].values]
yt = train["target"].values
Xte = np.c_[np.ones(test.shape[0]), test[["x1", "x2"]].values]

theta = np.linalg.lstsq(Xt, yt, rcond=None)[0]

def predict():
    return Xte @ theta
"""
    resp = post("/step", {"episode_id": episode_id, "action": program})
    obs = step_obs(resp)
    check("returned score", obs, "score")
    check("score >= 0.75 (good fit)", obs, "score",
          cond=lambda d: d.get("score", 0) >= 0.75)
    check("best_score updated", obs, "best_score",
          cond=lambda d: d.get("best_score", 0) >= 0.75)
    check("step incremented to 1", obs, "step", 1)
    check("not terminated", resp, "terminated", False)
    check("reward > 0", resp, "reward", cond=lambda r: r.get("reward", 0) > 0)
    return obs


def test_syntax_error_free_retry(episode_id: str):
    print("\n=== Syntax error (free retry — no fail consumed) ===")
    program = "this is not valid python @@@"
    resp = post("/step", {"episode_id": episode_id, "action": program})
    obs = step_obs(resp)
    check("returned error message", obs, "error")
    check("fail_count STILL 0 (free retry)", obs, "fail_count", 0)
    check("step NOT incremented (still 1)", obs, "step", 1)
    check("not terminated", resp, "terminated", False)
    check("reward is 0", resp, "reward", 0.0)
    return obs


def test_no_predict_function(episode_id: str):
    print("\n=== No predict function (runtime failure 1/3) ===")
    program = "x = 1"
    resp = post("/step", {"episode_id": episode_id, "action": program})
    obs = step_obs(resp)
    check("returned error about predict()", obs, "error",
          cond=lambda d: "predict" in str(d.get("error", "")).lower())
    check("fail_count == 1", obs, "fail_count", 1)
    check("step NOT incremented (still 1)", obs, "step", 1)
    check("NOT terminated", resp, "terminated", False)
    return obs


WRONG_COUNT_PROG = """
def predict():
    return [1.0, 2.0, 3.0]
"""


def test_runtime_failure_1(episode_id: str):
    print("\n=== Wrong prediction count (runtime failure 2/3) ===")
    resp = post("/step", {"episode_id": episode_id, "action": WRONG_COUNT_PROG})
    obs = step_obs(resp)
    check("returned error about expected count", obs, "error",
          cond=lambda d: "expected" in str(d.get("error", "")).lower())
    check("fail_count == 2", obs, "fail_count", 2)
    check("step NOT incremented (still 1)", obs, "step", 1)
    check("NOT terminated yet", resp, "terminated", False)
    return obs


def test_runtime_failure_2_and_terminate(episode_id: str):
    print("\n=== Third runtime failure (max fails reached) ===")
    resp = post("/step", {"episode_id": episode_id, "action": WRONG_COUNT_PROG})
    obs = step_obs(resp)
    check("fail_count == 3", obs, "fail_count", 3)
    check("terminated", resp, "terminated", True)
    check("truncated", resp, "truncated", True)
    return obs


def test_multi_step_chaining():
    print("\n=== Multi-step chaining with work_dir ===")
    eid = "test-chaining"

    post("/reset", {
        "episode_id": eid,
        "seed": 99,
        "scenario_params": {"toy_data": "1", "max_steps": 3},
    })

    program1 = """
import pandas as pd
import joblib
import numpy as np

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

Xt = np.c_[np.ones(train.shape[0]), train[["x1", "x2"]].values]
yt = train["target"].values
Xte = np.c_[np.ones(test.shape[0]), test[["x1", "x2"]].values]

theta = np.linalg.lstsq(Xt, yt, rcond=None)[0]

joblib.dump(theta, "work_dir/theta.pkl")
joblib.dump(Xte, "work_dir/Xte.pkl")

def predict():
    return Xte @ theta
"""
    resp = post("/step", {"episode_id": eid, "action": program1})
    s1_obs = step_obs(resp)
    check("step 1 score >= 0.75", s1_obs, "score",
          cond=lambda d: d.get("score", 0) >= 0.75)

    program2 = """
import joblib

theta = joblib.load("work_dir/theta.pkl")
Xte = joblib.load("work_dir/Xte.pkl")

def predict():
    return Xte @ theta
"""
    resp = post("/step", {"episode_id": eid, "action": program2})
    s2_obs = step_obs(resp)
    check("step 2 loaded from work_dir", s2_obs, "score",
          cond=lambda d: d.get("score", 0) >= 0.75)
    check("step incremented to 2", s2_obs, "step", 2)

    post("/close", {"episode_id": eid})
    print("  [INFO] chaining episode closed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global BASE, PASS, FAIL
    parser = argparse.ArgumentParser(description="Test the Kaggle Prediction Benchmark env")
    parser.add_argument("--url", default=BASE, help="Adapter URL")
    args = parser.parse_args()
    BASE = args.url

    try:
        resp = get("/health")
        if resp.get("status") != "ok":
            print(f"Adapter not healthy: {resp}")
            sys.exit(1)
        print(f"Adapter healthy — env: {resp.get('env', '?')}")
    except Exception as e:
        print(f"Cannot reach adapter at {BASE}: {e}")
        print("Start with: python3 auxiliary/adapter.py")
        sys.exit(1)

    test_health()
    test_reset_toy_data()

    eid = "test-suite-1"
    post("/reset", {
        "episode_id": eid,
        "seed": 42,
        "scenario_params": {"toy_data": "1"},
    })
    test_valid_program(eid)
    test_syntax_error_free_retry(eid)
    test_no_predict_function(eid)
    test_runtime_failure_1(eid)
    test_runtime_failure_2_and_terminate(eid)
    post("/close", {"episode_id": eid})
    print("  [INFO] test-suite-1 closed")

    test_multi_step_chaining()

    total = PASS + FAIL
    pc = PASS / total * 100 if total else 0
    print(f"\n{'=' * 40}")
    print(f"Results: {PASS}/{total} passed ({pc:.0f}%), {FAIL}/{total} failed")
    if FAIL > 0:
        sys.exit(1)
    print("All tests passed!")


if __name__ == "__main__":
    main()
