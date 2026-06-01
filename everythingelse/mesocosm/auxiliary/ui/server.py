"""FastAPI server for the Brain Activation Matching Local Testing UI."""

import glob
import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

# --- Paths ---

UI_DIR = os.path.dirname(os.path.abspath(__file__))
AUX_DIR = os.path.dirname(UI_DIR)
ENV_DIR = os.path.dirname(AUX_DIR)
REPO_ROOT = os.path.dirname(ENV_DIR)
PROJECT_ROOT = os.path.dirname(REPO_ROOT)
CONFIG_PATH = os.path.join(AUX_DIR, "run_config.json")
RESULTS_DIR = os.path.join(AUX_DIR, "results")
COMPARISONS_DIR = os.path.join(AUX_DIR, "comparisons")
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "adapter.py")
MANIFEST_PATH = os.path.join(AUX_DIR, "benchanything.json")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(COMPARISONS_DIR, exist_ok=True)

sys.path.insert(0, AUX_DIR)

# --- FastAPI ---

app = FastAPI(title="Brain Activation Matching — Local UI")

static_dir = os.path.join(UI_DIR, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

templates = Jinja2Templates(directory=os.path.join(UI_DIR, "templates"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class _CompareProxy(BaseModel):
    url: str
    sentence_a: str
    sentence_b: str


@app.post("/api/compare")
async def proxy_compare(req: _CompareProxy):
    """Proxy to the brain server's /compare_sync (avoids CORS issues)."""
    body = json.dumps({"sentence_a": req.sentence_a, "sentence_b": req.sentence_b}).encode()
    try:
        r = urllib.request.Request(
            f"{req.url.rstrip('/')}/compare_sync",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(r, timeout=300) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise HTTPException(e.code, detail=e.read().decode()[:500])
    except urllib.error.URLError as e:
        raise HTTPException(502, detail=str(e.reason))


class _CompareAsyncStart(BaseModel):
    url: str
    sentence_a: str
    sentence_b: str


@app.post("/api/compare-async")
async def proxy_compare_async(req: _CompareAsyncStart):
    """Start async compare, return job_id for polling."""
    body = json.dumps({"sentence_a": req.sentence_a, "sentence_b": req.sentence_b}).encode()
    try:
        r = urllib.request.Request(
            f"{req.url.rstrip('/')}/compare",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(r, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise HTTPException(e.code, detail=e.read().decode()[:500])
    except urllib.error.URLError as e:
        raise HTTPException(502, detail=str(e.reason))


class _StatusProxy(BaseModel):
    url: str
    job_id: str


@app.post("/api/compare-status")
async def proxy_status(req: _StatusProxy):
    """Poll job status."""
    try:
        r = urllib.request.Request(f"{req.url.rstrip('/')}/status/{req.job_id}")
        with urllib.request.urlopen(r, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise HTTPException(e.code, detail=e.read().decode()[:500])
    except urllib.error.URLError as e:
        raise HTTPException(502, detail=str(e.reason))

    record = {
        "id": datetime.now(timezone.utc).strftime("cmp_%Y%m%d_%H%M%S_%f"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sentence_a": req.sentence_a,
        "sentence_b": req.sentence_b,
        "brain_api_url": req.url,
        "result": result,
    }
    path = os.path.join(COMPARISONS_DIR, f"{record['id']}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return result


# --- Config ---

@dataclass
class RunConfig:
    model: str = "deepseek/deepseek-reasoner"
    episodes: int = 1
    max_steps: int = 20
    temperature: float = 0.6
    max_tokens: int = 4096
    system_prompt: str = ""
    brain_api_url: str = ""
    mode: str = "local"
    env_name: str = "Brain Activation Matching"
    github_url: str = ""
    env_description: str = "Agent submits sentences to match a target brain activation map computed by TRIBE v2."

    def to_env(self) -> dict[str, str]:
        env = {"MAX_STEPS": str(self.max_steps)}
        if self.brain_api_url:
            env["BRAIN_API_URL"] = self.brain_api_url
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

    def start_adapter(self, config: RunConfig, port: int):
        env = os.environ.copy()
        env.update(config.to_env())
        env["MESOCOSM_LOCAL"] = "1"
        self._adapter_port = port
        self._adapter_log = []
        self._adapter = subprocess.Popen(
            [sys.executable, ADAPTER_PATH, "--port", str(port), "--host", "127.0.0.1"],
            env=env,
            cwd=PROJECT_ROOT,
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

    def start_platform_submit(self, config: RunConfig):
        cmd_parts = [
            "mesocosm env submit",
            f'--name "{config.env_name}"',
        ]
        if config.github_url:
            cmd_parts.append(f'--github-url "{config.github_url}"')
        self._mesocosm_log = [" ".join(cmd_parts)]
        self._mesocosm = None

    def start_mesocosm(self, config: RunConfig):
        env = os.environ.copy()
        env.update(config.to_env())
        env["MESOCOSM_LOCAL"] = "1"
        self._mesocosm_log = []
        cmd = [
            "mesocosm", "run", "local",
            "--model", config.model,
            "--episodes", str(config.episodes),
            "--manifest", MANIFEST_PATH,
            "--env-url", f"http://127.0.0.1:{self._adapter_port}",
            "--temperature", str(config.temperature),
            "--max-tokens", str(config.max_tokens),
        ]
        if config.system_prompt:
            cmd += ["--system-prompt", config.system_prompt]
        self._mesocosm = subprocess.Popen(
            cmd,
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


# --- Saved results ---

def _fmt_time(ts: str) -> str:
    try:
        d = datetime.fromisoformat(ts)
        return d.strftime("%b %d %H:%M")
    except Exception:
        return ts[:16]


@app.get("/api/results")
async def list_results():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")), reverse=True)
    out = []
    for fp in files:
        try:
            with open(fp) as f:
                data = json.load(f)
            run_id = data.get("run_id", os.path.basename(fp).replace(".json", ""))
            poem = data.get("poem", {})
            out.append({
                "id": run_id,
                "timestamp": _fmt_time(data.get("timestamp", "")),
                "poem_title": poem.get("title", "?"),
                "poem_author": poem.get("author", "?"),
                "best_similarity": round(data.get("best_similarity", 0), 4),
                "total_steps": data.get("total_steps", 0),
            })
        except Exception:
            continue
    return out


@app.get("/api/results/{run_id}")
async def get_result(run_id: str):
    path = os.path.join(RESULTS_DIR, f"{run_id}.json")
    if not os.path.isfile(path):
        raise HTTPException(404, f"Run {run_id} not found")
    with open(path) as f:
        return json.load(f)


# --- Comparison History ---

def _fmt_cmp_time(ts: str) -> str:
    try:
        d = datetime.fromisoformat(ts)
        return d.strftime("%b %d %H:%M:%S")
    except Exception:
        return ts[:19]


@app.get("/api/comparisons")
async def list_comparisons():
    files = sorted(glob.glob(os.path.join(COMPARISONS_DIR, "*.json")), reverse=True)
    out = []
    for fp in files:
        try:
            with open(fp) as f:
                data = json.load(f)
            r = data.get("result", {})
            out.append({
                "id": data["id"],
                "timestamp": _fmt_cmp_time(data.get("timestamp", "")),
                "sentence_a": data["sentence_a"][:60],
                "sentence_b": data["sentence_b"][:60],
                "cosine_similarity": r.get("cosine_similarity"),
                "correlation": r.get("correlation"),
                "mse": r.get("mse"),
                "mae": r.get("mae"),
            })
        except Exception:
            continue
    return out


@app.get("/api/comparisons/{cmp_id}")
async def get_comparison(cmp_id: str):
    path = os.path.join(COMPARISONS_DIR, f"{cmp_id}.json")
    if not os.path.isfile(path):
        raise HTTPException(404, f"Comparison {cmp_id} not found")
    with open(path) as f:
        return json.load(f)


# --- Tests (placeholder) ---

@app.post("/api/tests/run")
async def run_tests():
    return {
        "id": "placeholder",
        "passed": 0, "failed": 0, "total": 0, "tests": [],
        "output": "No test suite available.",
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
