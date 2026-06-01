"""Brain Activation Matching — agent matches a poem's brain activation with a short sentence.

Calls back to your local GPU machine via BRAIN_API_URL for TRIBE v2 inference.
"""

from __future__ import annotations

import json
import os
import re
import uuid
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

from bench_common.env_sdk.base import BaseEnv, StepResult

AUX_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "everythingelse/mesocosm/auxiliary")
)
RESULTS_DIR = os.path.join(AUX_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEFAULT_BRAIN_API = os.environ.get("BRAIN_API_URL", "https://headsman-zips-antacid.ngrok-free.dev")

POEMS = [
    {"title": "Stopping by Woods", "author": "Robert Frost",
     "text": "Whose woods these are I think I know. His house is in the village though; He will not see me stopping here To watch his woods fill up with snow."},
    {"title": "Fog", "author": "Carl Sandburg",
     "text": "The fog comes on little cat feet. It sits looking over harbor and city on silent haunches and then moves on."},
    {"title": "Hope", "author": "Emily Dickinson",
     "text": "Hope is the thing with feathers that perches in the soul and sings the tune without the words and never stops at all."},
    {"title": "The Raven", "author": "Edgar Allan Poe",
     "text": "Once upon a midnight dreary while I pondered weak and weary over many a quaint and curious volume of forgotten lore."},
    {"title": "The Tyger", "author": "William Blake",
     "text": "Tyger Tyger burning bright in the forests of the night what immortal hand or eye could frame thy fearful symmetry?"},
    {"title": "Still I Rise", "author": "Maya Angelou",
     "text": "You may write me down in history with your bitter twisted lies you may trod me in the very dirt but still like dust Ill rise."},
]


def _call_compare_sync(api_url: str, sentence_a: str, sentence_b: str, timeout: int = 300) -> dict:
    body = json.dumps({"sentence_a": sentence_a, "sentence_b": sentence_b}).encode()
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/compare_sync",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode()
        raise RuntimeError(f"Brain API error ({e.code}): {detail[:300]}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach Brain API at {api_url}: {e.reason}")


class MyEnv(BaseEnv):
    MAX_STEPS = 20
    MAX_SENTENCE_WORDS = 15

    def __init__(self) -> None:
        self._current_step: int = 0
        self._poem: dict | None = None
        self._best_similarity: float = -1.0
        self._brain_api_url: str = DEFAULT_BRAIN_API
        self._run_id: str = ""
        self._steps: list[dict] = []

    def reset(self, seed: int | None = None, **params: Any) -> dict[str, Any]:
        self._current_step = 0
        self._best_similarity = -1.0
        self._run_id = params.get("run_id") or uuid.uuid4().hex[:12]
        self._steps = []

        api_url = params.get("brain_api_url") or self._brain_api_url
        if not api_url:
            raise RuntimeError(
                "BRAIN_API_URL not set. Pass it as scenario_param or set the env var.\n"
                "  On your GPU machine: python brain_server.py && ngrok http 8766"
            )
        self._brain_api_url = api_url

        import random as _random
        rng = _random.Random(seed)
        self._poem = rng.choice(POEMS)

        return {
            "task": (
                f"Write a short sentence (max {self.MAX_SENTENCE_WORDS} words) that captures "
                f"the same emotional impact as this poem:\n\n"
                f"\"{self._poem['text']}\"\n"
                f"— {self._poem['author']}\n\n"
                "Your response should contain your sentence on a line starting with SENTENCE:.\n"
                "Example: SENTENCE: Quiet mist settles over everything.\n\n"
                "The reward is the cosine similarity of brain activation patterns "
                "between the poem and your sentence. Higher is better (max 1.0). "
                "Keep it under 15 words."
            ),
            "poem": self._poem["text"],
            "poem_author": self._poem["author"],
            "poem_title": self._poem["title"],
            "step": self._current_step,
            "max_steps": self.MAX_STEPS,
            "max_sentence_words": self.MAX_SENTENCE_WORDS,
            "best_similarity": self._best_similarity,
        }

    def step(self, action: Any) -> StepResult:
        if self._poem is None:
            raise RuntimeError("Call reset() before step()")

        self._current_step += 1
        step_num = self._current_step

        raw_input = str(action).strip()

        sentence = raw_input
        sentence_match = re.search(r"(?:^|\n)\s*SENTENCE:\s*(.+)", raw_input, re.IGNORECASE)
        if sentence_match:
            sentence = sentence_match.group(1).strip()

        if not sentence:
            return StepResult(
                observation={
                    "error": "Empty sentence. Provide a non-empty sentence.",
                    "step": step_num, "max_steps": self.MAX_STEPS,
                    "best_similarity": self._best_similarity,
                },
                reward=-0.5, terminated=False, truncated=False,
                info={"error": "Empty sentence"},
            )

        word_count = len(sentence.split())
        try:
            metrics = _call_compare_sync(self._brain_api_url, self._poem["text"], sentence)
        except Exception as exc:
            return StepResult(
                observation={
                    "error": f"Brain API call failed: {exc}",
                    "step": step_num, "max_steps": self.MAX_STEPS,
                    "best_similarity": self._best_similarity,
                },
                reward=-0.5, terminated=False, truncated=False,
                info={"error": str(exc)},
            )

        reward = float(metrics.get("cosine_similarity", 0))
        self._best_similarity = max(self._best_similarity, reward)
        terminated = reward >= 0.95 or self._current_step >= self.MAX_STEPS

        em_a = metrics.get("emotion_a", {})
        em_b = metrics.get("emotion_b", {})
        poem_emotions = ", ".join(f"{k}={v:.2f}" for k, v in sorted(em_a.items(), key=lambda x: -x[1])[:3])
        sent_emotions = ", ".join(f"{k}={v:.2f}" for k, v in sorted(em_b.items(), key=lambda x: -x[1])[:3])

        feedback = (
            f"Brain similarity: {reward:.4f} (best: {self._best_similarity:.4f})\n"
            f"Your words: {word_count}/{self.MAX_SENTENCE_WORDS}\n"
            f"Poem emotions: {poem_emotions}\n"
            f"Your emotions: {sent_emotions}"
        )

        step_data = {
            "step": step_num,
            "sentence": sentence,
            "word_count": word_count,
            "reward": reward,
            "best_similarity": self._best_similarity,
            "metrics": {
                "cosine_similarity": metrics.get("cosine_similarity"),
                "mse": metrics.get("mse"),
                "mae": metrics.get("mae"),
                "correlation": metrics.get("correlation"),
            },
            "emotion_poem": em_a,
            "emotion_sentence": em_b,
            "terminated": terminated,
        }
        self._steps.append(step_data)

        return StepResult(
            observation={
                "metrics": step_data["metrics"],
                "emotion_poem": em_a,
                "emotion_sentence": em_b,
                "reward": reward,
                "best_similarity": self._best_similarity,
                "word_count": word_count,
                "max_words": self.MAX_SENTENCE_WORDS,
                "step": step_num, "max_steps": self.MAX_STEPS,
                "feedback": feedback,
            },
            reward=reward,
            terminated=terminated,
            truncated=False,
            info={
                "cosine_similarity": metrics.get("cosine_similarity"),
                "mse": metrics.get("mse"),
                "correlation": metrics.get("correlation"),
                "best_similarity": self._best_similarity,
                "word_count": word_count,
            },
        )

    def close(self) -> None:
        if not self._steps:
            return
        record = {
            "run_id": self._run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "poem": self._poem,
            "best_similarity": self._best_similarity,
            "total_steps": self._current_step,
            "steps": self._steps,
        }
        path = os.path.join(RESULTS_DIR, f"{self._run_id}.json")
        with open(path, "w") as f:
            json.dump(record, f, indent=2)

    def __del__(self) -> None:
        pass
