"""FastAPI server for comparing two sentences using TribeV2."""

import logging
import time
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.spatial.distance import cosine

from tribev2 import TribeModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model: TribeModel | None = None


def sentence_to_word_events(sentence: str, timeline: str = "default") -> pd.DataFrame:
    """Create Word events from a sentence (bypasses TTS/whisperx)."""
    words = sentence.split()
    events = []
    char_offset = 0
    for i, word in enumerate(words):
        pos = sentence.find(word, char_offset)
        if pos == -1:
            pos = char_offset
        context = sentence[:pos + len(word)]
        events.append({
            "type": "Word",
            "text": word,
            "start": float(i) * 0.3,
            "duration": 0.3,
            "timeline": timeline,
            "subject": "default",
            "language": "english",
            "sentence": sentence,
            "sentence_char": pos,
            "context": context,
            "stop": float(i) * 0.3 + 0.3,
        })
        char_offset = pos + len(word)
    return pd.DataFrame(events)


def predict_sentences(sentences: list[str]) -> np.ndarray:
    """Run TribeV2 inference on multiple sentences, returns (len(sentences), n_vertices)."""
    global model
    events = pd.concat(
        [sentence_to_word_events(s, timeline=f"sentence_{i}")
         for i, s in enumerate(sentences)],
        ignore_index=True,
    )
    preds, _ = model.predict(events, verbose=False)
    n_timelines = len(sentences)
    preds_per_timeline = np.array_split(preds, n_timelines)
    return np.array([p.mean(axis=0) for p in preds_per_timeline])


class CompareRequest(BaseModel):
    sentence1: str
    sentence2: str


class CompareResponse(BaseModel):
    similarity: float
    sentence1_preds: list[float]
    sentence2_preds: list[float]
    difference: list[float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info("Loading TribeV2 model...")
    t0 = time.time()
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder="./cache",
    )
    logger.info("Model loaded in %.1fs", time.time() - t0)
    yield


app = FastAPI(
    title="TribeV2 Sentence Comparison",
    description="Compare two sentences using the TribeV2 brain-activity prediction model.",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/compare", response_model=CompareResponse)
async def compare(req: CompareRequest):
    if not req.sentence1.strip() or not req.sentence2.strip():
        raise HTTPException(400, "Both sentences must be non-empty")
    try:
        logger.info("Comparing: %r vs %r", req.sentence1[:60], req.sentence2[:60])
        results = predict_sentences([req.sentence1, req.sentence2])
        p1, p2 = results[0], results[1]
        diff = (p1 - p2).tolist()
        sim = float(1 - cosine(p1, p2))
        return CompareResponse(
            similarity=sim,
            sentence1_preds=p1.tolist(),
            sentence2_preds=p2.tolist(),
            difference=diff,
        )
    except Exception as e:
        logger.exception("Comparison failed")
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
