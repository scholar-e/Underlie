# Tooling — Kaggle Prediction Benchmark

This directory implements a **BenchAnything** environment that evaluates an AI agent's ability to write predictive programs from Kaggle datasets.

The agent receives training data from a downloaded Kaggle dataset, writes a Python program with a `predict()` function, and iteratively improves it based on score feedback. The environment runs the agent's program against held-out test data and scores it on prediction accuracy.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Terminal 1 — start the env server
python adapter.py

# Terminal 2 — run benchmark locally (uses benchanything.json)
mesocosm run local --episodes 5

# With a custom dataset
KAGGLE_DATASET="username/dataset-name" TARGET_COLUMN="price" mesocosm run local --episodes 3
```

## Files

| File | Purpose |
|------|---------|
| `env.py` | `BaseEnv` implementation — `reset()` and `step()` for the benchmark |
| `benchanything.json` | Manifest for the BenchAnything platform (vow, spaces, scoring) |
| `AI_README.md` | Full specification for the AI agent being benchmarked |
| `cleanup_trial.py` | Deletes trial directories and cleans imported Python modules |
| `eval_trial.py` | Analyzes trial results outside of env.py — scores, programs, predictions |
| `trials/` | Per-episode trial directories (auto-created, gitignored) |
| `data/traces/` | JSONL trace files from local benchmark runs |

## How It Works

1. **`reset()`** downloads the Kaggle dataset (cached), splits into train/test, saves CSVs into a fresh trial directory under `trials/`, and returns training data to the agent
2. **`step(code)`** writes the agent's code to the trial dir, imports it as a Python module, calls `predict()`, compares output to ground truth, and returns a score
3. Multiple steps per episode let the agent iteratively improve its program using the persisted `work_dir`
4. All step artifacts (programs, predictions, scores) are saved to disk for later analysis

Each trial is fully self-contained in its own directory — clean it up with `cleanup_trial.py` and inspect results with `eval_trial.py`.

## Configuration

Settings are passed as params to `reset()` or set as environment variables:

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| Dataset name | `KAGGLE_DATASET` | `"uciml/iris"` | Kaggle dataset identifier (`"owner/name"`) |
| Target column | `TARGET_COLUMN` | auto-detect | Column the program should predict |
| Test size | `TEST_SIZE` | `0.3` | Fraction of data held out for evaluation |
| Max steps | `MAX_STEPS` | `5` | Successful program submissions per episode |
| Max failures | `MAX_FAILS` | `3` | Crashes before episode is terminated |
| Toy mode | `KAGGLE_TOY_DATA` | `"0"` | Set to `"1"` to use synthetic data (skips Kaggle download) |

## Dependencies

- **pandas**, **scikit-learn**, **kagglehub** — listed in `requirements.txt`, installed by the platform on `env submit`
- **bench_common** — provided by `pip install swecc-mesocosm` (see `LOCAL_DEV.md` in the repo root)

## Scoring

- **Classification** (target has ≤10 unique values): accuracy score (0.0–1.0)
- **Regression** (target has >10 unique values): R² score, clamped to [0.0, 1.0]
- Primary metric: `accuracy` aggregated as mean episode reward across episodes

## Scripts

### `eval_trial.py` — examine AI performance

```bash
# Evaluate a specific trial
python auxiliary/eval_trial.py auxiliary/trials/trial_20260528_120000_a1b2c3d4/

# Evaluate all trials
python auxiliary/eval_trial.py

# Show the best program and detailed prediction comparison
python auxiliary/eval_trial.py --programs --detailed
```

### `cleanup_trial.py` — clean up trial artifacts

```bash
# Delete a specific trial and its imported modules
python auxiliary/cleanup_trial.py auxiliary/trials/trial_20260528_120000_a1b2c3d4/

# Delete all trials
python auxiliary/cleanup_trial.py auxiliary/trials/trial_*

# Dry run — see what would be deleted
python auxiliary/cleanup_trial.py --dry-run auxiliary/trials/trial_*
```

## See Also

- `AI_README.md` — detailed spec that the AI agent receives, describing the task format and trial directory structure
- `LOCAL_DEV.md` (repo root) — instructions for local development with Ollama
- `adapter.py` (repo root) — HTTP server wrapping the environment
