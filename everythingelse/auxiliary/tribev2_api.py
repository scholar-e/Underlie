"""FastAPI server that compares brain activity predictions for two sentences using TRIBE v2.

Async version — returns immediately with a job ID, poll /status/{id} for results.
"""

import os
os.environ["TQDM_DISABLE"] = "1"

import tempfile
import uuid
import threading
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
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


class CompareResponse(BaseModel):
    mse: float
    mae: float
    correlation: float
    cosine_similarity: float
    n_segments_a: int
    n_segments_b: int
    n_segments_compared: int


class JobStatus(BaseModel):
    job_id: str
    status: str  # "processing", "done", "error"
    result: CompareResponse | None = None
    error: str | None = None


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from tribev2 import TribeModel
                _model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)
    return _model


def _predict_sentence(model, text: str) -> np.ndarray:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text)
        tmp_path = f.name
    try:
        df = model.get_events_dataframe(text_path=tmp_path)
        preds, _ = model.predict(events=df, verbose=False)
        return preds
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _align_shapes(a: np.ndarray, b: np.ndarray):
    n = min(a.shape[0], b.shape[0])
    return a[:n], b[:n]


def _run_job(job_id: str, sentence_a: str, sentence_b: str):
    try:
        model = _get_model()
        preds_a = _predict_sentence(model, sentence_a)
        preds_b = _predict_sentence(model, sentence_b)

        if preds_a.shape[0] == 0 or preds_b.shape[0] == 0:
            raise ValueError("One or both sentences produced no predictions")

        a, b = _align_shapes(preds_a, preds_b)
        result = compare_tribev2_predictions(a, b)
        result["n_segments_a"] = int(preds_a.shape[0])
        result["n_segments_b"] = int(preds_b.shape[0])
        result["n_segments_compared"] = int(a.shape[0])

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run("tribev2_api:app", host=args.host, port=args.port, reload=False)
