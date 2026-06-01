"""Brain Activation Matching — agent composes a poem whose brain activation matches a target poem.

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

TARGET_POEMS = [
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
    {"title": "A Dream Within a Dream", "author": "Edgar Allan Poe",
     "text": "Take this kiss upon the brow and in parting from you now thus much let me avow you are not wrong who deem that my days have been a dream yet if hope has flown away in a night or in a day in a vision or in none is it therefore the less gone all that we see or seem is but a dream within a dream."},
    {"title": "Do Not Go Gentle", "author": "Dylan Thomas",
     "text": "Do not go gentle into that good night old age should burn and rave at close of day rage rage against the dying of the light though wise men at their end know dark is right because their words had forked no lightning they do not go gentle into that good night."},
    {"title": "Shall I Compare Thee", "author": "William Shakespeare",
     "text": "Shall I compare thee to a summers day thou art more lovely and more temperate rough winds do shake the darling buds of may and summers lease hath all too short a date sometime too hot the eye of heaven shines and often is his gold complexion dimmed and every fair from fair sometime declines by chance or natures changing course untrimmed but thy eternal summer shall not fade."},
    {"title": "Harlem", "author": "Langston Hughes",
     "text": "What happens to a dream deferred does it dry up like a raisin in the sun or fester like a sore and then run does it stink like rotten meat or crust and sugar over like a syrupy sweet maybe it just sags like a heavy load or does it explode."},
    {"title": "The Road Not Taken", "author": "Robert Frost",
     "text": "Two roads diverged in a yellow wood and sorry I could not travel both and be one traveler long I stood and looked down one as far as I could to where it bent in the undergrowth then took the other as just as fair and having perhaps the better claim because it was grassy and wanted wear though as for that the passing there had worn them really about the same."},
    {"title": "Invictus", "author": "William Ernest Henley",
     "text": "Out of the night that covers me black as the pit from pole to pole I thank whatever gods may be for my unconquerable soul in the fell clutch of circumstance I have not winced nor cried aloud under the bludgeonings of chance my head is bloody but unbowed beyond this place of wrath and tears looms but the horror of the shade and yet the menace of the years finds and shall find me unafraid it matters not how strait the gate how charged with punishments the scroll I am the master of my fate I am the captain of my soul."},
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

    def __init__(self) -> None:
        self._current_step: int = 0
        self._target: dict | None = None
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
        self._target = rng.choice(TARGET_POEMS)

        return {
            "task": (
                f"Compose a poem that produces a TRIBE v2 brain activation pattern "
                f"as close as possible to this target poem:\n\n"
                f"\"{self._target['text']}\"\n"
                f"— {self._target['author']}\n\n"
                "Your response should contain your poem on a line starting with POEM:.\n"
                "You can write any length or style. The reward is the cosine similarity "
                "of brain activation patterns between the target and your poem.\n"
                "Higher similarity = better (max 1.0). Try to match the rhythm, "
                "structure, and imagery to get closer brain activations."
            ),
            "target_poem": self._target["text"],
            "target_author": self._target["author"],
            "target_title": self._target["title"],
            "step": self._current_step,
            "max_steps": self.MAX_STEPS,
            "best_similarity": self._best_similarity,
        }

    def step(self, action: Any) -> StepResult:
        if self._target is None:
            raise RuntimeError("Call reset() before step()")

        self._current_step += 1
        step_num = self._current_step

        raw_input = str(action).strip()

        poem = raw_input
        poem_match = re.search(r"(?:^|\n)\s*POEM:\s*(.+)", raw_input, re.IGNORECASE | re.DOTALL)
        if poem_match:
            poem = poem_match.group(1).strip()

        if not poem:
            return StepResult(
                observation={
                    "error": "Empty poem. Provide a non-empty poem.",
                    "step": step_num, "max_steps": self.MAX_STEPS,
                    "best_similarity": self._best_similarity,
                },
                reward=-0.5, terminated=False, truncated=False,
                info={"error": "Empty poem"},
            )

        try:
            metrics = _call_compare_sync(self._brain_api_url, self._target["text"], poem)
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

        feedback = (
            f"Brain similarity: {reward:.4f} (best: {self._best_similarity:.4f})\n"
            f"Segments: {metrics.get('n_segments_a', '?')} (target) vs {metrics.get('n_segments_b', '?')} (yours)"
        )

        step_data = {
            "step": step_num,
            "poem": poem,
            "reward": reward,
            "best_similarity": self._best_similarity,
            "metrics": {
                "cosine_similarity": metrics.get("cosine_similarity"),
                "mse": metrics.get("mse"),
                "mae": metrics.get("mae"),
                "correlation": metrics.get("correlation"),
            },
            "terminated": terminated,
        }
        self._steps.append(step_data)

        return StepResult(
            observation={
                "metrics": step_data["metrics"],
                "reward": reward,
                "best_similarity": self._best_similarity,
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
            },
        )

    def close(self) -> None:
        if not self._steps:
            return
        record = {
            "run_id": self._run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target": self._target,
            "best_similarity": self._best_similarity,
            "total_steps": self._current_step,
            "steps": self._steps,
        }
        path = os.path.join(RESULTS_DIR, f"{self._run_id}.json")
        with open(path, "w") as f:
            json.dump(record, f, indent=2)

    def __del__(self) -> None:
        pass
