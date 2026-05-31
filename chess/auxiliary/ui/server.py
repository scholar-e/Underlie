"""FastAPI server for the Chess Local Testing UI."""

import json
import os
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
AUX_DIR = os.path.dirname(UI_DIR)
CHESS_DIR = os.path.dirname(AUX_DIR)
REPO_ROOT = os.path.dirname(CHESS_DIR)
CONFIG_PATH = os.path.join(AUX_DIR, "run_config.json")
TRIALS_DIR = os.path.join(AUX_DIR, "trials")
TEST_RESULTS_DIR = os.path.join(AUX_DIR, "test_results")
ADAPTER_PATH = os.path.join(CHESS_DIR, "adapter.py")
MANIFEST_PATH = os.path.join(CHESS_DIR, "benchanything.json")

sys.path.insert(0, AUX_DIR)

# --- FastAPI ---

app = FastAPI(title="Chess Player — Local UI")

static_dir = os.path.join(UI_DIR, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

templates = Jinja2Templates(directory=os.path.join(UI_DIR, "templates"))


# --- Config ---

@dataclass
class RunConfig:
    stockfish_path: str = r".\stockfish\stockfish-windows-x86-64-avx2.exe"
    model: str = "ollama/llama3.2"
    episodes: int = 1
    max_steps: int = 150
    mode: str = "local"
    env_name: str = "Chess Player"
    github_url: str = ""
    env_description: str = "Evaluates an agent's ability to play competitive chess using Standard Algebraic Notation."

    def to_env(self) -> dict[str, str]:
        return {
            "STOCKFISH_PATH": self.stockfish_path,
            "MAX_STEPS": str(self.max_steps),
        }


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
            cwd=CHESS_DIR,
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

    # --- platform env submit ---

    def start_platform_submit(self, config: RunConfig):
        cmd_parts = [
            "mesocosm env submit",
            f'--name "{config.env_name}"',
        ]
        if config.github_url:
            cmd_parts.append(f'--github-url "{config.github_url}"')
        self._mesocosm_log = [" ".join(cmd_parts)]
        self._mesocosm = None

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


@app.post("/api/runs/generate-command")
async def generate_command(data: dict = None):
    cfg = _current_config
    if data:
        valid = set(RunConfig.__dataclass_fields__)
        filtered = {k: v for k, v in data.items() if k in valid}
        cfg = RunConfig(**{**asdict(cfg), **filtered})

    if cfg.mode == "platform":
        parts = [
            "mesocosm env submit",
            f'--name "{cfg.env_name}"',
        ]
        if cfg.github_url:
            parts.append(f'--github-url "{cfg.github_url}"')
        return {"command": " ".join(parts)}
    else:
        return {
            "command": (
                f"STOCKFISH_PATH={cfg.stockfish_path} "
                f"MAX_STEPS={cfg.max_steps} "
                f"mesocosm run local --model {cfg.model} "
                f"--episodes {cfg.episodes} --manifest {MANIFEST_PATH}"
            )
        }


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
        _procman.start_platform_submit(_current_config)
        return {
            "status": "command_generated",
            "mode": "platform",
            "command": _procman._mesocosm_log[0] if _procman._mesocosm_log else "",
        }
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


# --- Trials ---

os.makedirs(TRIALS_DIR, exist_ok=True)


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
                "best_reward": results.get("best_reward") or meta.get("best_reward"),
                "total_steps": results.get("total_steps") or meta.get("total_steps"),
                "game_result": results.get("game_result") or meta.get("game_result"),
                "num_moves": results.get("num_moves") or meta.get("num_moves"),
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
            sp = os.path.join(step_dir, "step_data.json")
            if os.path.exists(sp):
                with open(sp) as f:
                    sd.update(json.load(f))
        except Exception:
            pass
        steps.append(sd)
    steps.sort(key=lambda x: x["step"])

    return {"meta": meta, "results": results, "steps": steps}


# --- Tests (placeholder) ---

os.makedirs(TEST_RESULTS_DIR, exist_ok=True)


@app.post("/api/tests/run")
async def run_tests():
    return {
        "id": "placeholder",
        "passed": 0,
        "failed": 0,
        "total": 0,
        "tests": [],
        "output": "No test suite available yet. Create chess/auxiliary/test_env.py to add integration tests.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/tests")
async def list_tests():
    return []


@app.get("/api/tests/{test_id}")
async def get_test_result(test_id: str):
    raise HTTPException(404, "No test results available")


@app.on_event("shutdown")
def _shutdown():
    _procman.stop_all()
