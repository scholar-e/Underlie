# Local Testing UI

A lightweight FastAPI-based web UI for configuring datasets and running the Kaggle Prediction Benchmark locally via `mesocosm run local`.

## Overview

```
tooling/ui/
  README.md         ← this file
  __init__.py
  __main__.py       ← python -m tooling.ui
  server.py         ← FastAPI app (auto-selects free port)
  static/style.css
  templates/index.html
```

The UI runs on its own port (independent of the adapter), provides a 4-tab control panel, and persists the active run configuration to `tooling/run_config.json` so it can be reproduced from the CLI or showcase.

## Architecture

```
┌──────────────────────────────────────────────┐
│  tooling/ui/server.py  (auto port, e.g. 8766) │
│                                                │
│  Tab 1: Dataset Explorer                      │
│  Tab 2: Run Configuration                     │
│  Tab 3: Monitor (subprocess logs)             │
│  Tab 4: Results Browser                       │
└──────────────┬───────────────────────────────┘
               │
     ┌─────────┼─────────┐
     │         │         │
     ▼         ▼         ▼
  setup_    run_config  adapter.py
  dataset   .json       + mesocosm
  .py (import)          (subprocess)
```

## How It Works

### 1. Dataset Tab
- Pick from recommended datasets or enter a custom Kaggle slug.
- Click **Explore** — downloads the dataset, shows column analysis (dtype, nulls, unique values), auto-detects the target column, and renders a data preview.
- Choose the target column from a dropdown (overrides auto-detect).

### 2. Config Tab
- Editable parameters: `TEST_SIZE`, `MAX_STEPS`, `MAX_FAILS`, model name, episode count.
- Toggle toy data (synthetic) vs real Kaggle dataset.
- **Generate Command** — shows a copyable bash command block:
  ```bash
  KAGGLE_DATASET=uciml/iris TARGET_COLUMN=species TEST_SIZE=0.3 \
    MAX_STEPS=5 MAX_FAILS=3 python adapter.py --port 8765
  ```
- **Run Now** — saves config to `tooling/run_config.json`, starts the adapter, then runs mesocosm.

### 3. Monitor Tab
- Start/Stop adapter button with green/red status dot.
- **Run mesocosm** button (enabled when adapter is healthy).
- Live scrolling log output from both subprocesses.
- Port information displayed (which port the adapter was assigned).

### 4. Results Tab
- Lists completed trials from `tooling/trials/` sorted by recency.
- Shows dataset, target column, best score, step count, and timestamp per trial.
- Expandable per-step details: program code, score breakdown, predictions vs expected.

## Shared Config File

`tooling/run_config.json` is the bridge between the UI, CLI tooling, and showcase:

```json
{
  "dataset": "uciml/iris",
  "target_column": "species",
  "test_size": 0.3,
  "max_steps": 5,
  "max_fails": 3,
  "toy_data": false,
  "model": "ollama/llama3.2",
  "episodes": 1
}
```

**Config priority in env.py** (highest to lowest):
1. `reset(**params)` — direct API call
2. Environment variables (`KAGGLE_DATASET`, etc.)
3. `tooling/run_config.json` — persistent shared config
4. Hardcoded defaults

This means running `python adapter.py` without any env vars will pick up the last settings configured in the UI.

## Running the UI

```bash
# From the repo root:
python -m tooling.ui

# The UI prints the URL it's on (auto-selected free port).
# The adapter is spawned on a separate free port.
```

## Modifications to Existing Tooling

| File | Change |
|------|--------|
| `tooling/setup_dataset.py` | Add `silent=True` param to `download_dataset()`; raise instead of `sys.exit()` in `find_csv()`; add `get_columns_info()` and `build_command_str()` for programmatic callers |
| `tooling/env.py` | Add `_read_run_config()` helper; check `run_config.json` as fallback in `_load_dataset()` |
| `tooling/adapter.py` | No changes needed (config flows through env vars) |
