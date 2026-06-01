"""Brain server — TRIBE v2 with Phi-3-mini text encoder (MIT, hidden=3072, exact projector fit).
Runs on your GPU/CPU machine. Expose via ngrok, set BRAIN_API_URL for the cloud env.

Usage:
    python brain_server.py --port 8766
    # In another terminal: ngrok http 8766
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

os.environ["TQDM_DISABLE"] = "1"

AUX_DIR = Path(__file__).resolve().parent
TRIBE_SRC = AUX_DIR.parent.parent / "tribev2"
sys.path.insert(0, str(TRIBE_SRC))

CACHE_DIR = str(AUX_DIR / "model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TEXT_MODEL = "microsoft/Phi-3-mini-4k-instruct"

# ———— TRIBE v2 model (patched with Phi-3) ————

_model = None
_model_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

app = FastAPI(title="Brain Activation Server (TRIBE v2 + Phi-3)")


class CompareRequest(BaseModel):
    sentence_a: str
    sentence_b: str


class CompareResponse(BaseModel):
    cosine_similarity: float
    mse: float
    mae: float
    correlation: float
    n_segments_a: int
    n_segments_b: int
    n_segments_compared: int


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from tribev2 import TribeModel

                _original = TribeModel.from_pretrained

                @classmethod
                def _patched_from_pretrained(cls, checkpoint_dir, cache_folder=None,
                                             cluster="auto", device="auto", config_update=None):
                    merged = config_update or {}
                    merged["data.text_feature.model_name"] = TEXT_MODEL
                    merged["data.text_feature.device"] = device
                    return _original(checkpoint_dir, cache_folder=cache_folder,
                                     cluster=cluster, device=device, config_update=merged)

                TribeModel.from_pretrained = _patched_from_pretrained
                print(f"Loading TRIBE v2 with text encoder: {TEXT_MODEL}")
                _model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)
                print("Model loaded.")
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


def _compare(preds_a: np.ndarray, preds_b: np.ndarray) -> dict:
    a, b = _align_shapes(preds_a, preds_b)
    mse = float(np.mean((a - b) ** 2))
    mae = float(np.mean(np.abs(a - b)))
    a_flat = a.flatten()
    b_flat = b.flatten()
    if np.std(a_flat) == 0 or np.std(b_flat) == 0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(a_flat, b_flat)[0, 1])
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    cosine_similarity = float(np.dot(a_flat, b_flat) / denom) if denom != 0 else 0.0
    return {
        "mse": mse, "mae": mae,
        "correlation": correlation,
        "cosine_similarity": cosine_similarity,
    }


def _set_progress(job_id: str, msg: str):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["progress"] = msg


def _run_job(job_id: str, sentence_a: str, sentence_b: str):
    try:
        _set_progress(job_id, "Loading model…")
        model = _get_model()

        _set_progress(job_id, "Processing text A (TTS + brain prediction)…")
        preds_a, segs_a = _predict_sentence(model, sentence_a)

        _set_progress(job_id, "Processing text B (TTS + brain prediction)…")
        preds_b, segs_b = _predict_sentence(model, sentence_b)

        if preds_a.shape[0] == 0 or preds_b.shape[0] == 0:
            raise ValueError("One or both sentences produced no predictions")

        with _jobs_lock:
            _jobs[job_id]["preds_a"] = preds_a
            _jobs[job_id]["segs_a"] = segs_a
            _jobs[job_id]["preds_b"] = preds_b
            _jobs[job_id]["segs_b"] = segs_b
            _jobs[job_id]["text_a"] = sentence_a
            _jobs[job_id]["text_b"] = sentence_b

        _set_progress(job_id, "Comparing activations…")
        metrics = _compare(preds_a, preds_b)
        metrics["n_segments_a"] = int(preds_a.shape[0])
        metrics["n_segments_b"] = int(preds_b.shape[0])
        metrics["n_segments_compared"] = int(min(preds_a.shape[0], preds_b.shape[0]))

        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result"] = {**metrics}
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(e)


# ———— Endpoints ————


@app.get("/health")
def health():
    return {"status": "ok", "text_model": TEXT_MODEL, "model_loaded": _model is not None}


@app.post("/compare")
def compare(req: CompareRequest):
    if not req.sentence_a.strip() or not req.sentence_b.strip():
        raise HTTPException(400, "Both sentences must be non-empty")
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {"status": "processing"}
    t = threading.Thread(target=_run_job, args=(job_id, req.sentence_a, req.sentence_b), daemon=True)
    t.start()
    return {"job_id": job_id, "status": "processing"}


@app.post("/compare_sync", response_model=CompareResponse)
def compare_sync(req: CompareRequest):
    if not req.sentence_a.strip() or not req.sentence_b.strip():
        raise HTTPException(400, "Both sentences must be non-empty")
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {"status": "processing"}
    t = threading.Thread(target=_run_job, args=(job_id, req.sentence_a, req.sentence_b), daemon=True)
    t.start()
    for _ in range(600):
        time.sleep(0.5)
        with _jobs_lock:
            status = _jobs[job_id]["status"]
            if status == "done":
                return CompareResponse(**_jobs[job_id]["result"])
            if status == "error":
                raise HTTPException(500, detail=_jobs[job_id].get("error", "Unknown error"))
    raise HTTPException(504, detail="Job timed out (300s)")


@app.get("/status/{job_id}")
def get_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return {"job_id": job_id, "status": job["status"], "progress": job.get("progress", ""),
            "result": job.get("result"), "error": job.get("error")}


@app.get("/visualize/{job_id}")
def visualize(job_id: str, sentence: str = "a", n_timesteps: int = 10):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    if job["status"] != "done":
        raise HTTPException(400, "Job not yet completed")
    key = "a" if sentence == "a" else "b"
    preds = job.get(f"preds_{key}")
    segs = job.get(f"segs_{key}")
    text = job.get(f"text_{key}")
    if preds is None or len(preds) == 0:
        raise HTTPException(400, "No predictions available")

    n = min(n_timesteps, len(preds))
    from tribev2.plotting import PlotBrainPyvista
    plotter = PlotBrainPyvista(mesh="fsaverage5")
    views = ["left", "right"]
    fig, axes = plt.subplots(n, len(views), figsize=(2.5 * len(views), 2.5 * n),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.1})
    if n == 1:
        axes = axes.reshape(1, -1)
    for t in range(n):
        for v, view in enumerate(views):
            ax = axes[t, v]
            plotter.plot_surf(preds[t], axes=ax, views=view)
            if t == 0:
                ax.set_title(view, fontsize=10)
            if v == 0:
                ax.set_ylabel(f"t={t}", fontsize=10)
    fig.suptitle(f"{'A' if key=='a' else 'B'}: {text[:60]}", fontsize=12)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Brain Activation Server (TRIBE v2 + Phi-3)")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    print(f"Brain server starting on {args.host}:{args.port}")
    print(f"  Text encoder: {TEXT_MODEL}")
    print(f"  Compare: POST http://{args.host}:{args.port}/compare_sync")
    print(f"  Brain viz: GET  http://{args.host}:{args.port}/visualize/{{job_id}}?sentence=a|b")
    print(f"  Expose via ngrok: ngrok http {args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
