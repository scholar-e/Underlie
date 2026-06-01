"""FastAPI server that compares brain activity predictions for two sentences using TRIBE v2.

Async version — returns immediately with a job ID, poll /status/{id} for results.
"""

import os
os.environ["TQDM_DISABLE"] = "1"

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torchvision  # registers C++ ops like torchvision::nms needed by transitive deps

import tempfile
import time
import uuid
import threading

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import compare_tribev2_predictions

TRIBE_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(TRIBE_PATH))

CACHE_DIR = str(TRIBE_PATH / "cache")
OUTPUT_DIR = TRIBE_PATH / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="TRIBE v2 Sentence Comparison")

_model = None
_model_lock = threading.Lock()
_plotter = None
_plotter_lock = threading.Lock()
_jobs: dict[str, dict] = {}


class CompareRequest(BaseModel):
    sentence_a: str
    sentence_b: str


class CompareResponse(BaseModel):
    mse: float
    mae: float
    correlation: float
    cosine_similarity: float
    n_segments_a: int
    n_segments_b: int
    n_segments_compared: int
    image_a: str
    image_b: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # "processing", "done", "error"
    result: CompareResponse | None = None
    error: str | None = None


MODEL_SOURCE: str | None = None  # set via --checkpoint or --endpoint

def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from tribev2 import TribeModel
                source = MODEL_SOURCE or "facebook/tribev2"
                for attempt in range(5):
                    try:
                        _model = TribeModel.from_pretrained(source, cache_folder=CACHE_DIR)
                        break
                    except Exception:
                        if attempt < 4:
                            time.sleep(2 ** attempt)
                        else:
                            raise
    return _model


def _get_plotter():
    global _plotter
    if _plotter is None:
        with _plotter_lock:
            if _plotter is None:
                from tribev2.plotting import PlotBrain
                _plotter = PlotBrain(mesh="fsaverage5")
    return _plotter


def _save_brain_image(preds: np.ndarray, segments, save_path: str):
    plotter = _get_plotter()
    n_timesteps = min(len(preds), 12)
    fig = plotter.plot_timesteps(
        preds[:n_timesteps],
        segments=segments[:n_timesteps] if segments else None,
        views=["left", "right"],
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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

        image_a = str(OUTPUT_DIR / f"{job_id}_a.png")
        image_b = str(OUTPUT_DIR / f"{job_id}_b.png")
        _save_brain_image(preds_a, segs_a, image_a)
        _save_brain_image(preds_b, segs_b, image_b)

        a, b = _align_shapes(preds_a, preds_b)
        result = compare_tribev2_predictions(a, b)
        result["n_segments_a"] = int(preds_a.shape[0])
        result["n_segments_b"] = int(preds_b.shape[0])
        result["n_segments_compared"] = int(a.shape[0])
        result["image_a"] = image_a
        result["image_b"] = image_b

        _jobs[job_id] = {"status": "done", "result": result, "error": None}
    except Exception as e:
        _jobs[job_id] = {"status": "error", "result": None, "error": str(e)}


@app.post("/compare")
def compare(req: CompareRequest):
    if not req.sentence_a.strip() or not req.sentence_b.strip():
        raise HTTPException(400, "Both sentences must be non-empty")

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"status": "processing", "result": None, "error": None}

    t = threading.Thread(target=_run_job, args=(job_id, req.sentence_a, req.sentence_b), daemon=True)
    t.start()

    return {"job_id": job_id, "status": "processing"}


@app.get("/status/{job_id}", response_model=JobStatus)
def get_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return {"job_id": job_id, **job}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TRIBE v2 Brain Comparison API")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Local path to model dir (config.yaml + best.ckpt). "
                             "Default: download from HuggingFace")
    parser.add_argument("--endpoint", type=str, default=None,
                        help="HuggingFace Hub mirror URL (e.g. https://hf-mirror.com)")
    args = parser.parse_args()

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint

    if args.checkpoint:
        MODEL_SOURCE = str(Path(args.checkpoint).resolve())

    uvicorn.run("tribev2_api:app", host=args.host, port=args.port, reload=False)
