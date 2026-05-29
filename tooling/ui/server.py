"""FastAPI server for the Local Testing UI."""

import json
import os
import re
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

# --- Paths ---

UI_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLING_DIR = os.path.dirname(UI_DIR)
REPO_ROOT = os.path.dirname(TOOLING_DIR)
CONFIG_PATH = os.path.join(TOOLING_DIR, "run_config.json")
TRIALS_DIR = os.path.join(TOOLING_DIR, "trials")
TEST_RESULTS_DIR = os.path.join(TOOLING_DIR, "test_results")
ADAPTER_PATH = os.path.join(REPO_ROOT, "adapter.py")
MANIFEST_PATH = os.path.join(TOOLING_DIR, "benchanything.json")

# Ensure tooling is importable
sys.path.insert(0, TOOLING_DIR)

# --- FastAPI ---

app = FastAPI(title="Kaggle Prediction Benchmark — Local UI")

static_dir = os.path.join(UI_DIR, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

templates = Jinja2Templates(directory=os.path.join(UI_DIR, "templates"))


# --- Config ---

@dataclass
class RunConfig:
    dataset: str = "uciml/iris"
    target_column: str = ""
    test_size: float = 0.3
    max_steps: int = 5
    max_fails: int = 3
    toy_data: bool = False
    model: str = "ollama/llama3.2"
    episodes: int = 1
    mode: str = "local"
    domain_id: str = ""
    vow_version: str = "1.0.0"

    def to_env(self) -> dict[str, str]:
        env = {}
        if not self.toy_data:
            env["KAGGLE_DATASET"] = self.dataset
        if self.target_column:
            env["TARGET_COLUMN"] = self.target_column
        env["TEST_SIZE"] = str(self.test_size)
        env["MAX_STEPS"] = str(self.max_steps)
        env["MAX_FAILS"] = str(self.max_fails)
        if self.toy_data:
            env["KAGGLE_TOY_DATA"] = "1"
        return env


def load_config() -> RunConfig:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                data = json.load(f)
            valid = set(RunConfig.__dataclass_fields__)
            return RunConfig(**{k: v for k, v in data.items() if k in valid})
        except Exception:
            pass
    return RunConfig()


def save_config(cfg: RunConfig):
    with open(CONFIG_PATH, "w") as f:
        json.dump(asdict(cfg), f, indent=2)


_current_config = load_config()


# --- Subprocess Management ---

class ProcessManager:
    def __init__(self):
        self._adapter: subprocess.Popen | None = None
        self._mesocosm: subprocess.Popen | None = None
        self._adapter_log: list[str] = []
        self._mesocosm_log: list[str] = []
        self._adapter_port: int = 0
        self._lock = threading.Lock()

    # --- adapter ---

    def start_adapter(self, config: RunConfig, port: int):
        env = os.environ.copy()
        env.update(config.to_env())
        env["MESOCOSM_LOCAL"] = "1"
        self._adapter_port = port
        self._adapter_log = []
        self._adapter = subprocess.Popen(
            [sys.executable, ADAPTER_PATH, "--port", str(port), "--host", "127.0.0.1"],
            env=env,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        threading.Thread(target=self._pipe_adapter, daemon=True).start()

    def _pipe_adapter(self):
        for line in self._adapter.stdout:
            with self._lock:
                self._adapter_log.append(line.rstrip())
                if len(self._adapter_log) > 500:
                    self._adapter_log.pop(0)

    def stop_adapter(self):
        if self._adapter and self._adapter.poll() is None:
            self._adapter.terminate()
            try:
                self._adapter.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._adapter.kill()
            self._adapter = None

    @property
    def adapter_alive(self) -> bool:
        return self._adapter is not None and self._adapter.poll() is None

    @property
    def adapter_port(self) -> int:
        return self._adapter_port

    def adapter_log(self, n: int = 100) -> list[str]:
        with self._lock:
            return self._adapter_log[-n:]

    # --- platform run (mesocosm run create) ---

    def start_platform_run(self, config: RunConfig):
        env = os.environ.copy()
        env.update(config.to_env())
        self._mesocosm_log = []
        self._mesocosm = subprocess.Popen(
            [
                "mesocosm", "run", "create",
                "--domain", config.domain_id,
                "--vow-version", config.vow_version,
                "--model", config.model,
                "--episodes", str(config.episodes),
            ],
            env=env,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        threading.Thread(target=self._pipe_mesocosm, daemon=True).start()

    # --- local run (mesocosm run local) ---

    def start_mesocosm(self, config: RunConfig):
        env = os.environ.copy()
        env.update(config.to_env())
        env["MESOCOSM_LOCAL"] = "1"
        self._mesocosm_log = []
        self._mesocosm = subprocess.Popen(
            [
                "mesocosm", "run", "local",
                "--model", config.model,
                "--episodes", str(config.episodes),
                "--manifest", MANIFEST_PATH,
                "--env-url", f"http://127.0.0.1:{self._adapter_port}",
            ],
            env=env,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        threading.Thread(target=self._pipe_mesocosm, daemon=True).start()

    def _pipe_mesocosm(self):
        for line in self._mesocosm.stdout:
            with self._lock:
                self._mesocosm_log.append(line.rstrip())
                if len(self._mesocosm_log) > 500:
                    self._mesocosm_log.pop(0)

    def stop_mesocosm(self):
        if self._mesocosm and self._mesocosm.poll() is None:
            self._mesocosm.terminate()
            try:
                self._mesocosm.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._mesocosm.kill()
            self._mesocosm = None

    @property
    def mesocosm_alive(self) -> bool:
        return self._mesocosm is not None and self._mesocosm.poll() is None

    def mesocosm_log(self, n: int = 100) -> list[str]:
        with self._lock:
            return self._mesocosm_log[-n:]

    def stop_all(self):
        self.stop_mesocosm()
        self.stop_adapter()


_procman = ProcessManager()


# --- Helpers ---

def _make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    elif isinstance(obj, (float, int)):
        if isinstance(obj, float) and (obj != obj or abs(obj) == float("inf")):
            return str(obj)
        return obj
    elif isinstance(obj, str):
        return obj
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)


def _find_free_port() -> int:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})


@app.get("/api/config")
async def get_config():
    return asdict(_current_config)


@app.put("/api/config")
async def update_config(data: dict):
    global _current_config
    valid = set(RunConfig.__dataclass_fields__)
    filtered = {k: v for k, v in data.items() if k in valid}
    _current_config = RunConfig(**{**asdict(_current_config), **filtered})
    save_config(_current_config)
    return asdict(_current_config)


@app.post("/api/datasets/explore")
async def explore_dataset(data: dict):
    try:
        import pandas as pd
    except ImportError:
        raise HTTPException(400, "pandas not installed. Run: pip install pandas")

    name = (data.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "Dataset name is required")

    try:
        import kagglehub
        path = kagglehub.dataset_download(name)
    except ImportError:
        raise HTTPException(400, "kagglehub not installed. Run: pip install kagglehub")
    except Exception as e:
        msg = str(e)
        if "403" in msg:
            msg = f"Dataset '{name}' requires Kaggle authentication or is no longer available. Try a different dataset."
        raise HTTPException(400, msg)

    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    if not csv_files:
        raise HTTPException(400, f"No CSV files found in dataset")
    csv_path = os.path.join(path, csv_files[0])

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise HTTPException(400, f"Failed to read CSV: {e}")

    from setup_dataset import auto_detect_target, get_columns_info, analyze, RECOMMENDED

    target_hint = data.get("target_hint") or next(
        (d["target_hint"] for d in RECOMMENDED if d["name"] == name), None
    )
    target_col = data.get("target_column") or auto_detect_target(df, target_hint)
    if target_col not in df.columns:
        target_col = auto_detect_target(df, None)

    try:
        analysis = analyze(df, target_col)
        columns = get_columns_info(df, target_col)
        preview = _make_json_safe(df.head(5).to_dict(orient="records"))

        candidates = []
        for col in df.columns:
            if col == target_col:
                continue
            nu = int(df[col].nunique())
            if nu <= 20:
                candidates.append({"name": col, "unique": nu})
        candidates.sort(key=lambda x: x["unique"])
    except Exception as e:
        raise HTTPException(400, f"Failed to analyze dataset: {e}")

    return {
        "dataset": name,
        "csv_file": os.path.basename(csv_path),
        "rows": len(df),
        "cols": len(df.columns),
        "columns": _make_json_safe(columns),
        "preview": preview,
        "target_column": target_col,
        "target_candidates": candidates,
        "is_classification": analysis["is_classification"],
    }


@app.post("/api/runs/generate-command")
async def generate_command(data: dict = None):
    cfg = _current_config
    if data:
        valid = set(RunConfig.__dataclass_fields__)
        filtered = {k: v for k, v in data.items() if k in valid}
        cfg = RunConfig(**{**asdict(cfg), **filtered})

    from setup_dataset import build_command_str
    return {"command": build_command_str(asdict(cfg))}


@app.post("/api/runs/adapter/start")
async def start_adapter():
    if _procman.adapter_alive:
        return {"status": "already_running", "port": _procman.adapter_port}
    port = _find_free_port()
    _procman.start_adapter(_current_config, port)
    return {"status": "started", "port": port}


@app.post("/api/runs/adapter/stop")
async def stop_adapter():
    _procman.stop_adapter()
    return {"status": "stopped"}


@app.post("/api/runs/mesocosm/start")
async def start_mesocosm():
    if _procman.mesocosm_alive:
        return {"status": "already_running", "mode": _current_config.mode}
    if _current_config.mode == "platform":
        _procman.start_platform_run(_current_config)
        return {"status": "started", "mode": "platform"}
    else:
        if not _procman.adapter_alive:
            raise HTTPException(400, "Adapter must be running first")
        _procman.start_mesocosm(_current_config)
        return {"status": "started", "mode": "local"}


@app.post("/api/runs/mesocosm/stop")
async def stop_mesocosm():
    _procman.stop_mesocosm()
    return {"status": "stopped"}


@app.get("/api/runs/status")
async def run_status():
    return {
        "adapter": _procman.adapter_alive,
        "adapter_port": _procman.adapter_port,
        "mesocosm": _procman.mesocosm_alive,
    }


@app.get("/api/runs/logs")
async def get_logs(n: int = 100):
    return {
        "adapter": _procman.adapter_log(n),
        "mesocosm": _procman.mesocosm_log(n),
    }


@app.get("/api/datasets/recommended")
async def get_recommended():
    from setup_dataset import RECOMMENDED
    return RECOMMENDED


@app.get("/api/trials")
async def list_trials():
    if not os.path.isdir(TRIALS_DIR):
        return []
    trials = []
    for test_dir in sorted(os.listdir(TRIALS_DIR), reverse=True):
        test_path = os.path.join(TRIALS_DIR, test_dir)
        if not os.path.isdir(test_path):
            continue
        for trial_dir in sorted(os.listdir(test_path), reverse=True):
            trial_path = os.path.join(test_path, trial_dir)
            if not os.path.isdir(trial_path):
                continue
            meta = {}
            results = {}
            try:
                mp = os.path.join(trial_path, "meta.json")
                if os.path.exists(mp):
                    with open(mp) as f:
                        meta = json.load(f)
                rp = os.path.join(trial_path, "results.json")
                if os.path.exists(rp):
                    with open(rp) as f:
                        results = json.load(f)
            except Exception:
                pass
            trials.append({
                "id": f"{test_dir}/{trial_dir}",
                "dataset": meta.get("dataset") or results.get("dataset") or "?",
                "target_column": meta.get("target_column") or results.get("target_column") or "?",
                "best_score": results.get("best_score"),
                "total_steps": results.get("total_steps"),
                "is_classification": meta.get("is_classification") or results.get("is_classification"),
                "test_size": meta.get("test_size"),
            })
    return trials


@app.get("/api/trials/{test_id}/{trial_id}")
async def get_trial(test_id: str, trial_id: str):
    trial_path = os.path.join(TRIALS_DIR, test_id, trial_id)
    if not os.path.isdir(trial_path):
        raise HTTPException(404, "Trial not found")

    meta = {}
    results = {}
    try:
        mp = os.path.join(trial_path, "meta.json")
        if os.path.exists(mp):
            with open(mp) as f:
                meta = json.load(f)
        rp = os.path.join(trial_path, "results.json")
        if os.path.exists(rp):
            with open(rp) as f:
                results = json.load(f)
    except Exception:
        pass

    steps = []
    for d in sorted(os.listdir(trial_path)):
        if not d.startswith("step_"):
            continue
        step_dir = os.path.join(trial_path, d)
        if not os.path.isdir(step_dir):
            continue
        step_num = int(d.split("_")[1])
        sd: dict = {"step": step_num}
        try:
            sp = os.path.join(step_dir, "score.json")
            if os.path.exists(sp):
                with open(sp) as f:
                    sd.update(json.load(f))
            pp = os.path.join(step_dir, "program.py")
            if os.path.exists(pp):
                with open(pp) as f:
                    sd["program"] = f.read()
            predp = os.path.join(step_dir, "predictions.json")
            if os.path.exists(predp):
                with open(predp) as f:
                    sd["predictions"] = json.load(f)
        except Exception:
            pass
        steps.append(sd)
    steps.sort(key=lambda x: x["step"])

    best_program = ""
    bp = os.path.join(trial_path, "best_program.py")
    if os.path.exists(bp):
        with open(bp) as f:
            best_program = f.read()

    return {"meta": meta, "results": results, "steps": steps, "best_program": best_program}


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

os.makedirs(TEST_RESULTS_DIR, exist_ok=True)


def _parse_test_output(text: str) -> dict:
    tests = []
    current_section = ""
    passed = 0
    failed = 0

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("=== ") and stripped.endswith(" ==="):
            current_section = stripped.strip("= ")
        m = re.match(r"\[(PASS|FAIL)\]\s+(.*)", stripped)
        if m:
            is_pass = m.group(1) == "PASS"
            desc = m.group(2).strip()
            tests.append({
                "section": current_section,
                "name": desc,
                "passed": is_pass,
            })
            if is_pass:
                passed += 1
            else:
                failed += 1

    total = passed + failed
    summary = {"passed": passed, "failed": failed, "total": total}

    # Try to extract the final summary line
    m = re.search(r"Results:\s+(\d+)/(\d+)\s+passed", text)
    if m:
        summary["passed"] = int(m.group(1))
        summary["total"] = int(m.group(2))
        summary["failed"] = summary["total"] - summary["passed"]

    return {"tests": tests, **summary}


@app.post("/api/tests/run")
async def run_tests():
    if not _procman.adapter_alive:
        raise HTTPException(400, "Adapter must be running first")

    port = _procman.adapter_port
    test_script = os.path.join(TOOLING_DIR, "test_env.py")
    if not os.path.exists(test_script):
        raise HTTPException(500, f"Test script not found: {test_script}")

    try:
        result = subprocess.run(
            [sys.executable, test_script, "--url", f"http://127.0.0.1:{port}"],
            capture_output=True, text=True, timeout=60, cwd=REPO_ROOT,
        )
        output = result.stdout
        if result.stderr:
            output += "\n--- stderr ---\n" + result.stderr
    except subprocess.TimeoutExpired:
        raise HTTPException(500, "Tests timed out (60s)")
    except Exception as e:
        raise HTTPException(500, str(e))

    parsed = _parse_test_output(output)
    parsed["output"] = output
    parsed["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Store to disk
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fname = f"test_{ts}.json"
    fpath = os.path.join(TEST_RESULTS_DIR, fname)
    with open(fpath, "w") as f:
        json.dump(parsed, f, indent=2)

    return {"id": fname, **parsed}


@app.get("/api/tests")
async def list_tests():
    if not os.path.isdir(TEST_RESULTS_DIR):
        return []
    results = []
    for fname in sorted(os.listdir(TEST_RESULTS_DIR), reverse=True):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(TEST_RESULTS_DIR, fname)
        try:
            with open(fpath) as f:
                data = json.load(f)
            results.append({
                "id": fname,
                "timestamp": data.get("timestamp", "?"),
                "passed": data.get("passed", 0),
                "failed": data.get("failed", 0),
                "total": data.get("total", 0),
                "test_count": len(data.get("tests", [])),
            })
        except Exception:
            pass
    return results


@app.get("/api/tests/{test_id}")
async def get_test_result(test_id: str):
    # Sanitize: only allow alphanumeric, dots, underscores, hyphens
    if not re.match(r"^[\w\.\-]+$", test_id):
        raise HTTPException(400, "Invalid test id")
    fpath = os.path.join(TEST_RESULTS_DIR, test_id)
    if not os.path.isfile(fpath):
        raise HTTPException(404, "Test result not found")
    with open(fpath) as f:
        return json.load(f)


@app.on_event("shutdown")
def _shutdown():
    _procman.stop_all()
