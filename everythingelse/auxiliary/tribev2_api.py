"""TRIBE v2 API — patched to use Qwen text encoder (no HF token needed)."""

import os
os.environ["TQDM_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["DISPLAY"] = ""
os.environ["PYVISTA_OFF_SCREEN"] = "true"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tempfile
import uuid
import threading
import io
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sys
import torch

TRIBE_SRC = Path(__file__).resolve().parent.parent / "tribev2"
sys.path.insert(0, str(TRIBE_SRC))

CACHE_DIR = str(TRIBE_SRC / "cache")

TEXT_MODEL = "Qwen/Qwen2.5-1.5B"

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
                _original = TribeModel.from_pretrained

                @classmethod
                def _patched_from_pretrained(cls, checkpoint_dir, cache_folder=None, cluster="auto", device="auto", config_update=None):
                    merged = config_update or {}
                    merged["data.text_feature.model_name"] = TEXT_MODEL
                    xp = _original(checkpoint_dir, cache_folder=cache_folder, cluster=cluster, device=device, config_update=merged)
                    model = xp._model
                    if "text" in model.projectors:
                        old_w = model.projectors["text"].weight.data
                        old_b = model.projectors["text"].bias.data
                        old_in, out = old_w.shape[1], old_w.shape[0]
                        from transformers import AutoConfig
                        hf_cfg = AutoConfig.from_pretrained(TEXT_MODEL)
                        hs = getattr(hf_cfg, 'hidden_size', getattr(hf_cfg, 'd_model', 768))
                        new_in = hs * 2
                        if new_in != old_in:
                            new_proj = torch.nn.Linear(new_in, out)
                            with torch.no_grad():
                                if new_in <= old_in:
                                    new_proj.weight.data = old_w[:, :new_in]
                                else:
                                    new_proj.weight.data[:, :old_in] = old_w
                            new_proj.bias.data = old_b
                            model.projectors["text"] = new_proj
                    model.to("cpu")
                    model.eval()
                    xp._model = model
                    return xp

                TribeModel.from_pretrained = _patched_from_pretrained
                _model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)
    return _model


def _predict_sentence(model, text: str, progress_cb=None) -> tuple[np.ndarray, list]:
    if progress_cb:
        progress_cb("Converting text to speech and extracting words...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text)
        tmp_path = f.name
    try:
        df = model.get_events_dataframe(text_path=tmp_path)
        if progress_cb:
            progress_cb("Computing brain predictions...")
        preds, segments = model.predict(events=df, verbose=False)
        return preds, segments
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _align_shapes(a: np.ndarray, b: np.ndarray):
    n = min(a.shape[0], b.shape[0])
    return a[:n], b[:n]


def _set_progress(job_id: str, msg: str):
    _jobs[job_id]["progress"] = msg


def _run_job(job_id: str, sentence_a: str, sentence_b: str):
    try:
        prog = lambda msg: _set_progress(job_id, msg)

        prog("Loading model...")
        t0 = time.time()
        model = _get_model()
        prog(f"Model loaded in {time.time()-t0:.0f}s — processing sentence A...")

        t0 = time.time()
        preds_a, segs_a = _predict_sentence(model, sentence_a, prog)
        prog(f"Sentence A done ({time.time()-t0:.0f}s) — processing sentence B...")

        t0 = time.time()
        preds_b, segs_b = _predict_sentence(model, sentence_b, prog)
        prog(f"Sentence B done ({time.time()-t0:.0f}s) — comparing...")

        if preds_a.shape[0] == 0 or preds_b.shape[0] == 0:
            raise ValueError("One or both sentences produced no predictions")

        _jobs[job_id]["preds_a"] = preds_a
        _jobs[job_id]["segs_a"] = segs_a
        _jobs[job_id]["preds_b"] = preds_b
        _jobs[job_id]["segs_b"] = segs_b
        _jobs[job_id]["text_a"] = sentence_a
        _jobs[job_id]["text_b"] = sentence_b

        a, b = _align_shapes(preds_a, preds_b)

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from compare import compare_tribev2_predictions
        result = compare_tribev2_predictions(a, b)
        result["n_segments_a"] = int(preds_a.shape[0])
        result["n_segments_b"] = int(preds_b.shape[0])
        result["n_segments_compared"] = int(a.shape[0])

        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["result"] = result
        _jobs[job_id]["progress"] = "Done"
    except Exception as e:
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = str(e)
        _jobs[job_id]["progress"] = f"Error: {e}"


with open(Path(__file__).resolve().parent / "tribe.html") as _f:
    _INDEX_HTML = _f.read()


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=_INDEX_HTML)


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
    progress: str | None = None
    result: dict | None = None
    error: str | None = None


@app.get("/status/{job_id}", response_model=StatusOut)
def get_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return {"job_id": job_id, "status": job["status"],
            "progress": job.get("progress"),
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

    n = min(n_timesteps, len(preds))
    from tribev2.plotting import PlotBrainPyvista
    plotter = PlotBrainPyvista(mesh="fsaverage5")
    views = ["left", "right"]
    fig, axes = plt.subplots(
        n, len(views),
        figsize=(2.5 * len(views), 2.5 * n),
        gridspec_kw={"wspace": 0.05, "hspace": 0.1},
    )
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

    fig.suptitle(f"Sentence {key.upper()}: {text[:60]}", fontsize=12)
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
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
