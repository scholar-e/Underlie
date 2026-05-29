"""Kaggle Prediction Benchmark — AI writes a program with predict()."""

from __future__ import annotations

import importlib.util
import json
import os
import re as _re
import shutil
import subprocess
import sys
import uuid
from typing import Any

from bench_common.env_sdk.base import BaseEnv, StepResult

# --- Constants ---

TRIALS_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trials")
os.makedirs(TRIALS_BASE, exist_ok=True)

DATA_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Test counter: persists for the lifetime of the adapter process.
# Each process restart creates a new test_N directory.
_TEST_NUMBER: int | None = None


def _get_test_number() -> int:
    global _TEST_NUMBER
    if _TEST_NUMBER is None:
        existing = []
        if os.path.isdir(TRIALS_BASE):
            for d in os.listdir(TRIALS_BASE):
                parts = d.split("_")
                if len(parts) == 2 and parts[0] == "test" and parts[1].isdigit():
                    existing.append(int(parts[1]))
        _TEST_NUMBER = max(existing) + 1 if existing else 1
    return _TEST_NUMBER

FALLBACK_ITEMS = [
    {"question": "What is 2 + 2?", "answer": "4"},
    {"question": "What color is the sky?", "answer": "blue"},
]


class MyEnv(BaseEnv):
    def __init__(self) -> None:
        self._trial_dir: str | None = None
        self._train_df: pd.DataFrame | None = None
        self._test_df: pd.DataFrame | None = None
        self._train_targets: pd.Series | None = None
        self._test_targets: pd.Series | None = None
        self._target_column: str | None = None
        self._feature_columns: list[str] = []
        self._is_classification: bool = False
        self._current_step: int = 0
        self._fail_count: int = 0
        self._best_score: float = 0.0
        self._max_steps: int = 5
        self._max_fails: int = 3
        self._test_size: float = 0.3
        self._dataset_name: str = "shyamnadhs/heart-disease-prediction-dataset"
        self._label_encoder: Any = None
        self._rng: Any = None

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _generate_toy_data(self, seed: int | None) -> pd.DataFrame:
        import pandas as pd
        rng = __import__("numpy").random.RandomState(seed or 42)
        n = 50
        x1 = rng.rand(n) * 10
        x2 = rng.rand(n) * 10
        y = 2.0 * x1 + 3.0 * x2 + rng.randn(n) * 0.5
        return pd.DataFrame({"x1": x1, "x2": x2, "target": y})

    def _download_and_load(self, dataset_name: str) -> pd.DataFrame:
        import kagglehub
        import pandas as pd
        path = kagglehub.dataset_download(dataset_name)
        csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in Kaggle dataset '{dataset_name}'")
        csv_path = os.path.join(path, csv_files[0])
        df = pd.read_csv(csv_path)

        cache_path = self._cache_path_for(dataset_name)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path, index=False)

        return df

    def _auto_detect_target(self, df: pd.DataFrame) -> str:
        candidates = ["target", "label", "class", "y", "outcome", "answer"]
        for c in candidates:
            if c in df.columns:
                return c
        return df.columns[-1]

    @staticmethod
    def _read_run_config() -> dict:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_config.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    @staticmethod
    def _cache_path_for(dataset_name: str) -> str:
        safe = dataset_name.replace("/", "_").replace("-", "_")
        return os.path.join(DATA_CACHE_DIR, safe, "data.csv")

    def _load_dataset(self, seed: int | None = None, **params: Any) -> None:
        try:
            import pandas as pd
            from sklearn.model_selection import train_test_split
        except ImportError:
            raise RuntimeError(
                "Missing ML dependencies. Install with: pip install pandas scikit-learn"
            )

        cfg = self._read_run_config()

        dataset_name = (
            params.get("dataset")
            or os.environ.get("KAGGLE_DATASET")
            or cfg.get("dataset", "shyamnadhs/heart-disease-prediction-dataset")
        )
        target_column = (
            params.get("target_column")
            or os.environ.get("TARGET_COLUMN")
            or cfg.get("target_column")
        )
        self._test_size = float(
            params.get("test_size")
            or os.environ.get("TEST_SIZE")
            or cfg.get("test_size", 0.3)
        )
        self._max_steps = int(
            params.get("max_steps")
            or os.environ.get("MAX_STEPS")
            or cfg.get("max_steps", 5)
        )
        self._max_fails = int(
            params.get("max_fails")
            or os.environ.get("MAX_FAILS")
            or cfg.get("max_fails", 3)
        )
        use_toy = (
            str(
                params.get("toy_data")
                or os.environ.get("KAGGLE_TOY_DATA")
                or str(cfg.get("toy_data", "0"))
            )
            .lower()
            in ("1", "true", "yes")
        )

        if use_toy:
            df = self._generate_toy_data(seed)
            self._dataset_name = "toy-data"
        else:
            cache_path = self._cache_path_for(dataset_name)
            if os.path.isfile(cache_path):
                df = pd.read_csv(cache_path)
                self._dataset_name = dataset_name
            else:
                try:
                    import kagglehub  # noqa: F811
                    df = self._download_and_load(dataset_name)
                    self._dataset_name = dataset_name
                except ImportError:
                    import logging
                    logging.getLogger(__name__).warning(
                        "kagglehub not installed and no cached dataset at %s — falling back to toy data",
                        cache_path,
                    )
                    df = self._generate_toy_data(seed)
                    self._dataset_name = "toy-data"
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).warning(
                        "Failed to load dataset '%s': %s — falling back to toy data",
                        dataset_name, exc,
                    )
                    df = self._generate_toy_data(seed)
                    self._dataset_name = "toy-data"

        if target_column:
            if target_column not in df.columns:
                raise ValueError(
                    f"Target column '{target_column}' not found. "
                    f"Available columns: {list(df.columns)}"
                )
            self._target_column = target_column
        else:
            self._target_column = self._auto_detect_target(df)

        self._feature_columns = [c for c in df.columns if c != self._target_column]
        self._is_classification = df[self._target_column].nunique() <= 10

        X = df[self._feature_columns]
        y = df[self._target_column]

        # Encode string targets to integers so the env can compare predictions
        # (predict() must return numbers, not strings like 'Iris-setosa').
        _label_encoder = None
        if pd.api.types.is_string_dtype(y.dtype) or y.dtype == object:
            from sklearn.preprocessing import LabelEncoder
            _label_encoder = LabelEncoder()
            y = pd.Series(
                _label_encoder.fit_transform(y),
                index=y.index,
                name=y.name,
            )

        split_seed = seed if seed is not None else 0
        self._train_df, self._test_df, self._train_targets, self._test_targets = (
            train_test_split(X, y, test_size=self._test_size, random_state=split_seed)
        )

        self._train_df[self._target_column] = self._train_targets

        self._label_encoder = _label_encoder

    # ------------------------------------------------------------------
    # Trial directory
    # ------------------------------------------------------------------

    @staticmethod
    def _next_trial_in_test(test_dir: str) -> int:
        existing = []
        if os.path.isdir(test_dir):
            for d in os.listdir(test_dir):
                parts = d.split("_")
                if len(parts) == 2 and parts[0] == "trial" and parts[1].isdigit():
                    existing.append(int(parts[1]))
        return max(existing) + 1 if existing else 1

    def _create_trial_dir(self) -> str:
        test_num = _get_test_number()
        test_dir = os.path.join(TRIALS_BASE, f"test_{test_num}")
        os.makedirs(test_dir, exist_ok=True)
        trial_num = self._next_trial_in_test(test_dir)
        name = f"trial_{trial_num}"
        trial_dir = os.path.join(test_dir, name)
        os.makedirs(os.path.join(trial_dir, "work_dir"), exist_ok=True)
        return trial_dir

    def _save_trial_data(self) -> None:
        import pandas as pd
        train_path = os.path.join(self._trial_dir, "train.csv")
        test_path = os.path.join(self._trial_dir, "test.csv")
        expected_path = os.path.join(self._trial_dir, "expected.json")
        meta_path = os.path.join(self._trial_dir, "meta.json")

        # Rename the target column to "target" so the AI can hardcode
        # 'target' in its program regardless of the original column name.
        train_to_save = self._train_df.rename(
            columns={self._target_column: "target"}
        )

        # Encode string feature columns (e.g. gender="Male"/"Female") so
        # that any sklearn model can consume the CSV directly.
        str_cols = [
            c for c in train_to_save.columns
            if pd.api.types.is_string_dtype(train_to_save[c].dtype)
               and c != "target"
        ]
        str_encoders = {}
        for col in str_cols:
            le = __import__("sklearn").preprocessing.LabelEncoder()
            all_vals = pd.concat([
                train_to_save[col],
                self._test_df[col] if col in self._test_df.columns else pd.Series(dtype="object"),
            ], ignore_index=True).dropna().unique()
            le.fit(all_vals)
            train_to_save[col] = le.transform(train_to_save[col])
            if col in self._test_df.columns:
                self._test_df[col] = le.transform(self._test_df[col])
            str_encoders[col] = le.classes_.tolist()

        train_to_save.to_csv(train_path, index=False)
        self._test_df.to_csv(test_path, index=False)

        with open(expected_path, "w") as f:
            json.dump(self._test_targets.tolist(), f)

        meta = {
            "dataset": self._dataset_name,
            "target_column": self._target_column,
            "feature_columns": self._feature_columns,
            "test_size": self._test_size,
            "max_steps": self._max_steps,
            "max_fails": self._max_fails,
            "is_classification": self._is_classification,
            "dataset_rows": len(self._train_df) + len(self._test_df),
            "train_rows": len(self._train_df),
            "test_rows": len(self._test_df),
        }
        if self._label_encoder is not None:
            meta["target_classes"] = self._label_encoder.classes_.tolist()
        if str_encoders:
            meta["feature_encodings"] = str_encoders
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def _save_step(
        self, step: int, program: str, predictions: list | None,
        score: float, best_score: float, crashed: bool, stderr: str,
        reasoning: str = "",
    ) -> None:
        step_dir = os.path.join(self._trial_dir, f"step_{step}")
        os.makedirs(step_dir, exist_ok=True)

        with open(os.path.join(step_dir, "program.py"), "w") as f:
            f.write(program)

        with open(os.path.join(step_dir, "score.json"), "w") as f:
            json.dump(
                {
                    "score": score,
                    "best_score": best_score,
                    "crashed": crashed,
                    "stderr": stderr[:1000] if stderr else "",
                },
                f,
                indent=2,
            )

        if predictions is not None:
            with open(os.path.join(step_dir, "predictions.json"), "w") as f:
                json.dump([float(p) for p in predictions], f)

        if stderr:
            with open(os.path.join(step_dir, "stderr.txt"), "w") as f:
                f.write(stderr[:2000])

        if reasoning:
            with open(os.path.join(step_dir, "reasoning.txt"), "w") as f:
                f.write(reasoning)

    def _write_results(self) -> None:
        steps = []
        step_dirs = sorted(
            (d for d in os.listdir(self._trial_dir) if d.startswith("step_")),
            key=lambda x: int(x.split("_")[1]),
        )
        for sd in step_dirs:
            score_path = os.path.join(self._trial_dir, sd, "score.json")
            if os.path.isfile(score_path):
                with open(score_path) as f:
                    steps.append(json.load(f))

        results_path = os.path.join(self._trial_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(
                {
                    "dataset": self._dataset_name,
                    "target_column": self._target_column,
                    "is_classification": self._is_classification,
                    "test_size": self._test_size,
                    "best_score": self._best_score,
                    "total_steps": self._current_step,
                    "total_attempts": self._current_step + self._fail_count,
                    "steps": steps,
                },
                f,
                indent=2,
            )

    # ------------------------------------------------------------------
    # Module cleanup
    # ------------------------------------------------------------------

    def _clean_modules_from_trial(self) -> None:
        if not self._trial_dir:
            return
        trial_abspath = os.path.abspath(self._trial_dir)
        to_delete = []
        for name, mod in list(sys.modules.items()):
            if hasattr(mod, "__file__") and mod.__file__:
                try:
                    mod_path = os.path.abspath(mod.__file__)
                    if mod_path.startswith(trial_abspath):
                        to_delete.append(name)
                except Exception:
                    pass
        for name in to_delete:
            del sys.modules[name]

    # ------------------------------------------------------------------
    # Program execution
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_code(raw: str) -> str:
        lines = raw.split("\n")

        # If wrapped in a fenced code block, extract the interior
        in_block = False
        extracted = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                if in_block:
                    in_block = False
                    continue
                in_block = True
                continue
            if in_block:
                extracted.append(line)

        if extracted:
            code = "\n".join(extracted)
        else:
            code = raw

        import re as _re

        code_lines = code.split("\n")

        # Remove module-level print(predict()) calls
        cleaned = []
        for line in code_lines:
            if _re.match(r"^\s*print\s*\(.*predict\s*\(.*\)\s*\)\s*$", line):
                continue
            cleaned.append(line)

        # Remove module-level bare expression calls to predict()
        cleaned2 = []
        for line in cleaned:
            if _re.match(r"^\s*predict\s*\(\s*\)\s*$", line):
                continue
            cleaned2.append(line)

        # Strip trailing non-code content.
        # Anything that is a top-level (unindented) prose comment, plain text,
        # print() call, or blank line after the last real code block is removed.
        code_start = True  # haven't seen real code yet
        code_lines_out = []
        trailing_junk = []
        for line in cleaned2:
            stripped = line.strip()
            if code_start:
                # Haven't hit code yet — pass through but track
                if stripped and not stripped.startswith("#"):
                    code_start = False
                code_lines_out.append(line)
            else:
                # We've seen code. Check if this is trailing prose.
                is_indented = line.startswith((" ", "\t"))
                is_comment = stripped.startswith("#")
                is_print = bool(_re.match(r"^print\s*\(", stripped))
                is_blank = not stripped
                is_python_keyword = bool(
                    _re.match(
                        r"^(def |class |import |from |return |pass |if |elif |else:|for |while |try:|except|finally:|with |raise |yield |del |assert )",
                        stripped,
                    )
                )
                is_expression = bool(
                    _re.match(r"^[a-zA-Z_][a-zA-Z0-9_.]*\s*(=|\()", stripped)
                )

                if is_blank:
                    trailing_junk.append(line)
                elif (is_comment or is_print) and not is_indented:
                    trailing_junk.append(line)
                elif not is_indented and not is_python_keyword and not is_expression and not is_comment:
                    trailing_junk.append(line)
                else:
                    code_lines_out.extend(trailing_junk)
                    code_lines_out.append(line)
                    trailing_junk = []

        code = "\n".join(code_lines_out)
        return code.strip()

    @staticmethod
    def _extract_reasoning(raw: str) -> str:
        """Extract the model's thinking/reasoning text (everything outside ``` code blocks)."""
        lines = raw.split("\n")
        in_block = False
        reasoning_parts = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_block = not in_block
                continue
            if not in_block:
                reasoning_parts.append(line)
        # Strip empty lines from start/end
        while reasoning_parts and not reasoning_parts[0].strip():
            reasoning_parts.pop(0)
        while reasoning_parts and not reasoning_parts[-1].strip():
            reasoning_parts.pop()
        return "\n".join(reasoning_parts).strip()

    def _import_and_run(self, clean: str) -> tuple[list[float], str, str]:
        """Lower level: write + import + predict(). No length check."""
        module_name = f"_kaggle_prog_{uuid.uuid4().hex}"
        program_path = os.path.join(self._trial_dir, "program.py")

        clean = clean.replace("path_to_your_data_directory", self._trial_dir)

        preamble = (
            f'data_dir = {self._trial_dir!r}\n'
            f'work_dir = {os.path.join(self._trial_dir, "work_dir")!r}\n'
            f'train_file = "train.csv"\n'
            f'test_file = "test.csv"\n'
            f'target_column = "target"\n'
            f'feature_columns = {self._feature_columns!r}\n'
            f'is_classification = {self._is_classification!r}\n'
            f'\n'
        )
        full_pgm = preamble + clean

        with open(program_path, "w") as f:
            f.write(full_pgm)

        self._clean_modules_from_trial()

        stderr = ""
        orig_cwd = os.getcwd()
        os.chdir(self._trial_dir)
        try:
            spec = importlib.util.spec_from_file_location(module_name, program_path)
            if spec is None or spec.loader is None:
                raise RuntimeError("Failed to create module spec from program")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            if not hasattr(module, "predict") or not callable(module.predict):
                raise RuntimeError("Program must define a callable predict() function")

            result = module.predict()
            arr = __import__("numpy").asarray(result)
            predictions = [float(v) for v in arr.ravel()]

            return predictions, stderr, clean
        except subprocess.TimeoutExpired:
            raise
        except Exception as exc:
            stderr = str(exc)
            raise
        finally:
            os.chdir(orig_cwd)

    def _execute_program(self, clean: str) -> tuple[list[float], str, str]:
        predictions, stderr, clean = self._import_and_run(clean)
        if len(predictions) != len(self._test_targets):
            raise RuntimeError(
                f"predict() returned {len(predictions)} values, "
                f"expected {len(self._test_targets)}"
            )
        return predictions, stderr, clean

    def _compute_score(self, predictions: list[float]) -> float:
        from sklearn.metrics import accuracy_score, r2_score
        if self._is_classification:
            return float(accuracy_score(self._test_targets, predictions))
        else:
            return max(0.0, float(r2_score(self._test_targets, predictions)))

    # ------------------------------------------------------------------
    # Public API (BaseEnv)
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, **params: Any) -> dict[str, Any]:
        self._load_dataset(seed=seed, **params)

        old_trial = self._trial_dir
        self._trial_dir = self._create_trial_dir()
        self._save_trial_data()

        self._current_step = 0
        self._fail_count = 0
        self._best_score = 0.0

        if old_trial and os.path.isdir(old_trial):
            self._clean_modules_from_trial()

        # Build preview with column names matching the CSV files
        # (_save_trial_data renames the target column to "target")
        preview = _make_json_safe(
            self._train_df.head(5).rename(columns={self._target_column: "target"}).to_dict(orient="records")
        )

        skeleton = self._make_skeleton(self._is_classification)
        task_type = "classification" if self._is_classification else "regression"
        return {
            "task": (
                f"Write a Python program that defines predict(). "
                f"This is a {task_type} task ({len(self._test_targets)} test rows, "
                f"{len(self._feature_columns)} features). "
                f"Use this exact skeleton:\n{skeleton}\n\n"
                f"CRITICAL: Do NOT re-split train.csv with train_test_split. "
                f"Do NOT inverse-transform predictions. "
                f"Do NOT return accuracy or score. "
                f"Return one prediction per test row."
            ),
            "data_dir": self._trial_dir,
            "work_dir": os.path.join(self._trial_dir, "work_dir"),
            "train_file": "train.csv",
            "test_file": "test.csv",
            "target_column": "target",
            "feature_columns": self._feature_columns,
            "data_preview": preview,
            "is_classification": self._is_classification,
            "step": self._current_step,
            "max_steps": self._max_steps,
            "best_score": self._best_score,
        }

    @staticmethod
    def _make_skeleton(is_classification: bool) -> str:
        algo = "RandomForestClassifier" if is_classification else "RandomForestRegressor"
        return f"""\
import pandas as pd
from sklearn.ensemble import {algo}

# Variables available in your environment:
#   data_dir, work_dir, train_file, test_file
#   target_column (= "target"), feature_columns, is_classification

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# train.csv has BOTH features and the target (column "target").
# test.csv has ONLY features (no target column).
# All string features are pre-encoded to integers.

feat = [c for c in train.columns if c != "target"]
model = {algo}()
model.fit(train[feat], train["target"])

def predict():
    # ═══════════════════════════════════════════════════════════
    # CRITICAL RULES — violations cause crashes:
    # 1. Return ONE prediction per row in test.csv (same order)
    # 2. Return a list/array of predictions, NOT a single value
    # 3. Do NOT return accuracy, score, or probabilities
    # 4. Do NOT re-split train.csv with train_test_split
    # 5. Do NOT use scaler.inverse_transform() on predictions
    # 6. For classification: use model.predict(), NOT predict_proba()
    # 7. Check is_classification to choose classifier vs regressor
    # ═══════════════════════════════════════════════════════════
    return model.predict(test[feat])"""

    @staticmethod
    def _syntax_check(program: str) -> str | None:
        try:
            compile(program, "<program>", "exec")
            return None
        except SyntaxError as e:
            return f"SyntaxError: {e.msg} (line {e.lineno})"

    @staticmethod
    def _classify_error(exc: Exception, is_classification: bool) -> str:
        msg = str(exc)
        # ── Wrong algorithm type ──
        if "Unknown label type: continuous" in msg:
            return (
                "You used a classifier on a regression task "
                "(the target has continuous values). "
                "Use a regressor like RandomForestRegressor or LinearRegression instead. "
                f"Error: {msg[:200]}"
            )
        if "Unknown label type" in msg:
            return (
                f"You used a classifier on the wrong task type. "
                f"Use a regressor for continuous targets or a classifier for discrete targets. "
                f"Check is_classification to decide which to use. "
                f"Error: {msg[:200]}"
            )
        # ── Classification metric on regression predictions ──
        if "Classification metrics can't handle a mix" in msg:
            return (
                "You used accuracy_score (a classification metric) on regression predictions. "
                "Remove accuracy_score entirely — just return the raw predictions from model.predict()."
            )
        # ── inverse_transform on 1D predictions ──
        if "Expected 2D array, got 1D array instead" in msg and "inverse_transform" in msg:
            return (
                "scaler.inverse_transform() requires 2D input, but predictions are 1D. "
                "Do NOT inverse-transform model predictions — they are already in the correct space. "
                "Remove scaler.inverse_transform() and return model.predict() directly."
            )
        # ── Wrong prediction count ──
        if "predict() returned" in msg and "values, expected" in msg:
            return (
                f"{msg} — you're predicting on the wrong data. "
                f"Make sure predict() returns one prediction per row in test.csv. "
                f"Check that you're using test.csv, not train.csv."
            )
        # ── NameError (undefined variable) ──
        if "name '" in msg and "is not defined" in msg:
            return (
                f"{msg} — your program references a variable that doesn't exist. "
                f"Check that all variable names are spelled correctly and defined before use."
            )
        # ── Fallback: include the raw error plus context ──
        task_type = "regression" if not is_classification else "classification"
        return (
            f"Error: {msg[:400]}\n"
            f"(Task type: {task_type})"
        )

    def step(self, action: Any) -> StepResult:
        if self._trial_dir is None:
            raise RuntimeError("Call reset() before step()")

        program = str(action)
        reasoning = self._extract_reasoning(program)
        clean_program = self._extract_code(program)

        # Syntax check — free retries, no fail_count consumed.
        err = self._syntax_check(clean_program)
        if err:
            return StepResult(
                observation={
                    "error": err, "step": self._current_step,
                    "max_steps": self._max_steps,
                    "fail_count": self._fail_count, "max_fails": self._max_fails,
                    "message": "Syntax error — fix and resubmit (no attempt consumed).",
                },
                reward=0.0, terminated=False, truncated=False,
                info={"error": err, "syntax_error": True},
            )

        # — Single execution per attempt —
        # Run the program.  If anything fails (bad types, wrong count,
        # crash) it consumes a trial — no separate validation pass.
        predictions = None
        score = 0.0
        stderr = ""
        crashed = False
        exec_error = ""  # user-facing error message

        try:
            predictions, stderr, clean_program = self._execute_program(clean_program)
        except subprocess.TimeoutExpired:
            exec_error = "Program timed out (30s limit)"
            crashed = True
        except Exception as exc:
            exec_error = self._classify_error(exc, self._is_classification)
            crashed = True

        if not crashed:
            # Verify predictions are valid numbers.
            for i in range(min(3, len(predictions))):
                if not isinstance(predictions[i], (int, float)):
                    exec_error = (
                        f"predict()[{i}] is {type(predictions[i]).__name__} "
                        f"({predictions[i]!r}), expected a number. "
                        f"Use model.predict() not predict_proba()."
                    )
                    crashed = True
                    break

        if not crashed:
            score = self._compute_score(predictions)

        if crashed:
            self._fail_count += 1
            _cleanup_temp_pgm(self._trial_dir)
            step_num = self._current_step + self._fail_count
            if self._fail_count >= self._max_fails:
                self._save_step(
                    step_num, clean_program, None,
                    0.0, self._best_score, True, exec_error, reasoning,
                )
                self._write_results()
                return StepResult(
                    observation={
                        "error": exec_error,
                        "reasoning": reasoning,
                        "step": self._current_step,
                        "max_steps": self._max_steps,
                        "fail_count": self._fail_count,
                        "max_fails": self._max_fails,
                        "message": f"Too many failures ({self._fail_count}/{self._max_fails}). Episode terminated.",
                    },
                    reward=0.0,
                    terminated=True,
                    truncated=True,
                    info={"error": exec_error, "crashed": True},
                )

            self._save_step(
                step_num, clean_program, None,
                0.0, self._best_score, True, exec_error, reasoning,
            )
            _cleanup_temp_pgm(self._trial_dir)
            return StepResult(
                observation={
                    "error": exec_error,
                    "reasoning": reasoning,
                    "step": self._current_step,
                    "max_steps": self._max_steps,
                    "fail_count": self._fail_count,
                    "max_fails": self._max_fails,
                    "message": (
                        f"Program crashed (attempt {self._fail_count}/{self._max_fails}). "
                        "Fix the error and try again."
                    ),
                },
                reward=0.0,
                terminated=False,
                truncated=False,
                info={"error": exec_error, "crashed": True},
            )

        self._current_step += 1
        self._best_score = max(self._best_score, score)
        terminated = (score >= 1.0) or (self._current_step >= self._max_steps)
        step_num = self._current_step + self._fail_count

        self._save_step(
            step_num, clean_program, predictions,
            score, self._best_score, False, stderr, reasoning,
        )
        _cleanup_temp_pgm(self._trial_dir)

        if score >= self._best_score:
            best_src = os.path.join(self._trial_dir, f"step_{step_num}", "program.py")
            best_dst = os.path.join(self._trial_dir, "best_program.py")
            if os.path.isfile(best_src):
                shutil.copy2(best_src, best_dst)

        if terminated:
            self._write_results()

        return StepResult(
            observation={
                "score": score,
                "best_score": self._best_score,
                "step": self._current_step,
                "max_steps": self._max_steps,
                "is_classification": self._is_classification,
                "reasoning": reasoning,
                "feedback": (
                    f"Score: {score:.4f}. Best so far: {self._best_score:.4f}."
                    + (f" Stderr: {stderr[:200]}" if stderr else "")
                ),
            },
            reward=score,
            terminated=terminated,
            truncated=False,
            info={
                "score": score,
                "best_score": self._best_score,
                "num_predictions": len(predictions) if predictions else 0,
                "expected_predictions": len(self._test_targets),
                "stderr": stderr[:500] if stderr else "",
            },
        )


    def close(self) -> None:
        if self._trial_dir and self._current_step > 0:
            self._write_results()

    def __del__(self) -> None:
        pass


def _cleanup_temp_pgm(trial_dir: str | None) -> None:
    if trial_dir:
        p = os.path.join(trial_dir, "program.py")
        if os.path.isfile(p):
            os.remove(p)


def _make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    elif isinstance(obj, (float, int)):
        if isinstance(obj, float) and (obj != obj or abs(obj) == float("inf")):
            return str(obj)
        return obj
    elif isinstance(obj, str):
        return obj
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)
