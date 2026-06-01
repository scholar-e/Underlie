"""FastAPI server that compares brain activity predictions for two sentences using TRIBE v2.

Async version — returns immediately with a job ID, poll /status/{id} for results.
GET /visualize/{job_id}?sentence=a  returns a brain activity grid image as PNG.
"""

import os
os.environ["TQDM_DISABLE"] = "1"

# Windows fix: exca uses os.kill(pid, 0) which raises OSError instead of ProcessLookupError
import ctypes
kernel32 = ctypes.CDLL("kernel32", use_last_error=True)
def _is_pid_alive_win(pid: int) -> bool:
    handle = kernel32.OpenProcess(0x400, False, pid)
    if not handle:
        return False
    kernel32.CloseHandle(handle)
    return True

import exca.cachedict.inflight as _inflight
_inflight._is_pid_alive = _is_pid_alive_win

import tempfile
import uuid
import threading
import io
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import compare_tribev2_predictions

TRIBE_PATH = Path(__file__).resolve().parent.parent / "everythingelse" / "tribev2"
sys.path.insert(0, str(TRIBE_PATH))

CACHE_DIR = str(TRIBE_PATH / "cache")

app = FastAPI(title="TRIBE v2 Sentence Comparison")

_model = None
_model_lock = threading.Lock()
_jobs: dict[str, dict] = {}


class CompareRequest(BaseModel):
    sentence_a: str
    sentence_b: str


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from tribev2 import TribeModel
                _model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)
    return _model


def _predict_sentence(model, text: str) -> tuple[np.ndarray, list]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text)
        tmp_path = f.name
    try:
        df = model.get_events_dataframe(text_path=tmp_path)
        preds, segments = model.predict(events=df, verbose=False)
        return preds, segments
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _align_shapes(a: np.ndarray, b: np.ndarray):
    n = min(a.shape[0], b.shape[0])
    return a[:n], b[:n]


def _run_job(job_id: str, sentence_a: str, sentence_b: str):
    try:
        model = _get_model()
        preds_a, segs_a = _predict_sentence(model, sentence_a)
        preds_b, segs_b = _predict_sentence(model, sentence_b)

        if preds_a.shape[0] == 0 or preds_b.shape[0] == 0:
            raise ValueError("One or both sentences produced no predictions")

        # Store for visualization
        _jobs[job_id]["preds_a"] = preds_a
        _jobs[job_id]["segs_a"] = segs_a
        _jobs[job_id]["preds_b"] = preds_b
        _jobs[job_id]["segs_b"] = segs_b
        _jobs[job_id]["text_a"] = sentence_a
        _jobs[job_id]["text_b"] = sentence_b

        a, b = _align_shapes(preds_a, preds_b)
        result = compare_tribev2_predictions(a, b)
        result["n_segments_a"] = int(preds_a.shape[0])
        result["n_segments_b"] = int(preds_b.shape[0])
        result["n_segments_compared"] = int(a.shape[0])

        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["result"] = result
    except Exception as e:
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = str(e)


@app.post("/compare")
def compare(req: CompareRequest):
    if not req.sentence_a.strip() or not req.sentence_b.strip():
        raise HTTPException(400, "Both sentences must be non-empty")

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"status": "processing"}

    t = threading.Thread(target=_run_job, args=(job_id, req.sentence_a, req.sentence_b), daemon=True)
    t.start()

    return {"job_id": job_id, "status": "processing"}


class StatusOut(BaseModel):
    job_id: str
    status: str
    result: dict | None = None
    error: str | None = None


@app.get("/status/{job_id}", response_model=StatusOut)
def get_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return {"job_id": job_id, "status": job["status"],
            "result": job.get("result"), "error": job.get("error")}


@app.get("/visualize/{job_id}")
def visualize(job_id: str, sentence: str = "a", n_timesteps: int = 10):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    if job["status"] != "done":
        raise HTTPException(400, "Job not yet completed")
    key = "a" if sentence == "a" else "b"
    preds = job.get(f"preds_{key}")
    segments = job.get(f"segs_{key}")
    text = job.get(f"text_{key}")
    if preds is None or len(preds) == 0:
        raise HTTPException(400, "No predictions available for this sentence")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(n_timesteps, len(preds))
    from tribev2.plotting import PlotBrain
    plotter = PlotBrain(mesh="fsaverage5")
    fig = plotter.plot_timesteps(
        preds[:n],
        segments=segments[:n] if segments else None,
        views="left",
        cmap="fire",
        norm_percentile=99,
        alpha_cmap=(0, 0.2),
        show_stimuli=True,
    )
    fig.suptitle(f"Sentence {key.upper()}: {text}", fontsize=14)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run("tribev2_api:app", host=args.host, port=args.port, reload=False)
