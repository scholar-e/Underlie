# Kaggle Prediction Benchmark

You (the AI) will write a Python program that predicts values from a Kaggle dataset. You receive training data, build a predictor, then iteratively improve it based on your score.

You can break your solution into multiple chained steps, where each step builds on files saved in previous steps. This lets you construct complex pipelines with memory across steps.

## How It Works

```
reset() → downloads Kaggle dataset (cached), samples train/test split,
          returns training data + file paths + optional work_dir in observation

step(program_code) → writes your code to program.py, imports it as a module,
                     runs it (executes all top-level code), then calls predict(),
                     compares predictions to ground truth,
                     returns score + feedback

                 ↓ (repeat up to max_steps per episode)

terminated=True → episode ends, stats recorded
```

## Your Task

Write a Python program that:

1. Reads `train.csv` (features + target) and `test.csv` (features only) from the `data_dir` provided in the observation
2. Learns the relationship between features and target from `train.csv`
3. Defines a function `predict()` that returns a list/array of predictions for `test.csv` (same row order)

Your program can use any approach: machine learning, a simple formula, a lookup table, rules, statistics — anything that produces accurate predictions.

## Working Directory & Multi-Step Chaining

Each episode gives you a **working directory** (`work_dir`) that persists across all steps. Files you write here in one step are available in later steps. Use it to build multi-step logic chains:

- **Step 1**: Write a program that preprocesses data, trains a model, saves artifacts to `work_dir`
- **Step 2**: Write a program that loads the artifacts from `work_dir`, fine-tunes or improves the model, saves back
- **Step 3**: Write a program that loads the improved model and predicts

The working directory is optional — if you don't need it, just write a single self-contained program.

### Multi-step chain example

**Step 1 — Preprocess and train a baseline:**

```python
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

features = [c for c in train.columns if c != "target"]
target = "target"

model = RandomForestClassifier(n_estimators=50)
model.fit(train[features], train[target])

# Save for next step
joblib.dump(model, "work_dir/model.pkl")
joblib.dump(features, "work_dir/features.pkl")

def predict():
    return model.predict(test[features])
```

**Step 2 — Load and improve:**

```python
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Load artifacts from previous step
model = joblib.load("work_dir/model.pkl")
features = joblib.load("work_dir/features.pkl")
target = "target"

# Improve: more trees
model.n_estimators = 200
model.fit(train[features], train[target])

joblib.dump(model, "work_dir/model.pkl")

def predict():
    return model.predict(test[features])
```

## What You Get (Observation)

```json
{
  "task": "Write a Python program that defines predict()",
  "data_dir": "/tmp/kaggle_bench_abc123/",
  "work_dir": "/tmp/kaggle_bench_abc123/work_dir/",
  "train_file": "train.csv",
  "test_file": "test.csv",
  "target_column": "species",
  "feature_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
  "data_preview": [{"sepal_length": 5.1, "sepal_width": 3.5, ...}],
  "is_classification": true,
  "step": 0,
  "max_steps": 5,
  "best_score": 0.0
}
```

| Field | Description |
|-------|-------------|
| `task` | Description of what you need to do |
| `data_dir` | Directory containing the CSV files |
| `work_dir` | Persists across steps — save intermediate files here |
| `train_file` | Training data (features + target) |
| `test_file` | Test data (features only) |
| `target_column` | Name of the column to predict |
| `feature_columns` | Names of input feature columns |
| `data_preview` | First 5 rows of training data (to understand the format) |
| `is_classification` | Whether the task is classification (≤10 unique target values) |
| `step` | Current step number (0-indexed) |
| `max_steps` | Maximum steps in this episode |
| `best_score` | Best score achieved so far in this episode |

## What You Send (Action)

Your action is raw Python source code. The environment writes it to `program.py`, imports it as a module (executing all top-level code), and calls `program.predict()`.

Because import runs everything at the top level, your training and setup logic executes automatically — just put `predict()` at the bottom with the rest of your code above it.

## How You're Scored

- **Classification** (≤10 unique target values): accuracy = correct predictions / total predictions
- **Regression** (>10 unique target values): R² score clamped to [0, 1]
- Score is 0.0 to 1.0. Higher is better.

### On failure

If your program crashes, the step **doesn't count** toward `max_steps`. You get a free retry — but too many consecutive failures (default 3) terminates the episode.

## Multiple Trials

Each episode uses a train/test split of the dataset. The seed controls the split:
- Same seed across episodes = same split (test on the same data repeatedly)
- Different seeds = different splits (test generalization)

You can iterate within an episode (up to `max_steps` successful attempts) and across episodes.

## Trial Directory

Each episode creates a self-contained trial directory under `auxiliary/trials/trial_<timestamp>_<id>/`:

```
trial_20260528_120000_a1b2c3d4/
├── meta.json            (dataset, target, seed, split config)
├── train.csv            (training data — features + target)
├── test.csv             (test data — features only)
├── expected.json        (ground truth targets for test.csv)
├── work_dir/            (your scratch space, persists across steps)
├── step_000/
│   ├── program.py       (your submitted code)
│   ├── predictions.json (what predict() returned)
│   ├── score.json       (score, best_score, errors)
│   └── stderr.txt       (error output, if any)
├── step_001/
│   └── ...
├── best_program.py      (copy of your best-scoring program)
└── results.json         (summary of the entire episode)
```

You can examine trial directories after a run:
```bash
# Evaluate a specific trial
python auxiliary/eval_trial.py auxiliary/trials/trial_20260528_120000_a1b2c3d4/

# Evaluate all trials
python auxiliary/eval_trial.py

# Show best programs and detailed comparison
python auxiliary/eval_trial.py --programs --detailed

# Clean up a trial (removes dir and imported modules)
python auxiliary/cleanup_trial.py auxiliary/trials/trial_20260528_120000_a1b2c3d4/

# Clean up all trials
python auxiliary/cleanup_trial.py auxiliary/trials/trial_*
```

## Configuration

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| Dataset | `KAGGLE_DATASET` | `"uciml/iris"` | Kaggle dataset name |
| Target column | `TARGET_COLUMN` | auto-detect | Which column to predict |
| Test size | `TEST_SIZE` | `0.3` | Fraction held out for evaluation |
| Max steps | `MAX_STEPS` | `5` | Successful steps per episode |
| Max failures | `MAX_FAILS` | `3` | Crashes before episode terminates |
| Toy mode | `KAGGLE_TOY_DATA` | `"0"` | Use synthetic data (no Kaggle) |
