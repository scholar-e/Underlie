"""
HTTP adapter — exposes your env via the BenchAnything four-endpoint protocol.

Local dev:
    python adapter.py
    python adapter.py --port 9000
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Type

import uvicorn
from bench_common.env_sdk.base import BaseEnv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env import MyEnv

log = logging.getLogger(__name__)


class _ResetRequest(BaseModel):
    episode_id: str
    seed: int | None = None
    scenario_params: dict[str, Any] | None = None


class _StepRequest(BaseModel):
    episode_id: str
    action: Any


class _CloseRequest(BaseModel):
    episode_id: str


def serve(env_class: Type[BaseEnv], *, host: str = "0.0.0.0", port: int = 8765, log_level: str = "info") -> None:
    _episodes: dict[str, BaseEnv] = {}

    app = FastAPI(
        title=f"BenchAnything Env Adapter — {env_class.__name__}",
        version="1.0.0",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "env": env_class.__name__, "episodes": len(_episodes)}

    @app.post("/reset")
    def reset(req: _ResetRequest) -> dict:
        if req.episode_id in _episodes:
            try:
                _episodes[req.episode_id].close()
            except Exception:
                pass

        env = env_class()
        _episodes[req.episode_id] = env

        params = req.scenario_params or {}

        try:
            obs = env.reset(seed=req.seed, **params)
        except Exception as exc:
            _episodes.pop(req.episode_id, None)
            raise HTTPException(status_code=500, detail=str(exc))

        data = obs
        content_type = "application/json"
        system_prompt = None

        if isinstance(obs, dict) and "data" in obs and ("content_type" in obs or "system_prompt" in obs):
            data = obs.get("data")
            content_type = obs.get("content_type", content_type)
            system_prompt = obs.get("system_prompt")

        return {"data": data, "content_type": content_type, "system_prompt": system_prompt}

    @app.post("/step")
    def step(req: _StepRequest) -> dict:
        env = _episodes.get(req.episode_id)
        if env is None:
            raise HTTPException(status_code=404, detail=f"No active episode '{req.episode_id}'. Call /reset first.")
        try:
            result = env.step(req.action)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        return {
            "observation": {"data": result.observation, "content_type": "application/json"},
            "reward": float(result.reward),
            "terminated": bool(result.terminated),
            "truncated": bool(result.truncated),
            "info": {str(k): str(v) for k, v in result.info.items()},
            "system_prompt": result.system_prompt,
        }

    @app.post("/close")
    def close(req: _CloseRequest) -> dict:
        env = _episodes.pop(req.episode_id, None)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        return {}

    log.info(f"Starting {env_class.__name__} adapter on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print(f"MyEnv adapter → http://{args.host}:{args.port}")
    serve(MyEnv, host=args.host, port=args.port)
