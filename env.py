"""Brain Activation Matching — agent composes a poem matching a target's brain activation.

Calls your GPU machine (BRAIN_API_URL) for TRIBE v2 inference.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from bench_common.env_sdk.base import BaseEnv, StepResult

BRAIN_API = os.environ.get("BRAIN_API_URL", "https://headsman-zips-antacid.ngrok-free.dev")

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


def _compare(api_url: str, text_a: str, text_b: str, timeout: int = 120) -> dict:
    body = json.dumps({"sentence_a": text_a, "sentence_b": text_b}).encode()
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/compare_sync",
        data=body, headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Brain API error ({e.code}): {e.read().decode()[:200]}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach brain server at {api_url}: {e.reason}")


class MyEnv(BaseEnv):
    MAX_STEPS = 20

    def __init__(self) -> None:
        self._step = 0
        self._target = None
        self._best = -1.0
        self._api_url = BRAIN_API

    def reset(self, seed=None, **params):
        self._step = 0
        self._best = -1.0

        api = params.get("brain_api_url") or self._api_url
        if not api:
            raise RuntimeError("BRAIN_API_URL not set")
        self._api_url = api

        import random as _random
        self._target = _random.Random(seed).choice(TARGET_POEMS)

        return {
            "task": (
                f"Compose a poem whose brain activation pattern matches this target:\n\n"
                f"\"{self._target['text']}\"\n— {self._target['author']}\n\n"
                "Start your response with POEM: on its own line."
            ),
            "target_poem": self._target["text"],
            "target_author": self._target["author"],
            "target_title": self._target["title"],
            "step": self._step,
            "max_steps": self.MAX_STEPS,
            "best_similarity": self._best,
        }

    def step(self, action):
        if self._target is None:
            raise RuntimeError("Call reset() first")

        self._step += 1
        raw = str(action).strip()

        poem = raw
        m = re.search(r"(?:^|\n)\s*POEM:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
        if m:
            poem = m.group(1).strip()

        if not poem:
            return StepResult(
                observation={"error": "Empty poem", "step": self._step, "max_steps": self.MAX_STEPS, "best_similarity": self._best},
                reward=-0.5, terminated=False, truncated=False,
                info={"error": "Empty poem"},
            )

        try:
            metrics = _compare(self._api_url, self._target["text"], poem)
        except Exception as e:
            return StepResult(
                observation={"error": str(e), "step": self._step, "max_steps": self.MAX_STEPS, "best_similarity": self._best},
                reward=-0.5, terminated=False, truncated=False,
                info={"error": str(e)},
            )

        sim = float(metrics.get("cosine_similarity", 0))
        self._best = max(self._best, sim)
        done = sim >= 0.95 or self._step >= self.MAX_STEPS

        return StepResult(
            observation={
                "cosine_similarity": sim, "mse": metrics.get("mse"),
                "correlation": metrics.get("correlation"),
                "best_similarity": self._best,
                "step": self._step, "max_steps": self.MAX_STEPS,
            },
            reward=sim,
            terminated=done,
            truncated=False,
            info={"cosine_similarity": sim, "best_similarity": self._best, "mse": metrics.get("mse")},
        )

    def close(self):
        pass
