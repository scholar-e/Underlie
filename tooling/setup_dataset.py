#!/usr/bin/env python3
"""Download, inspect, and configure a Kaggle dataset for the benchmark.

Usage:
    python3 tooling/setup_dataset.py
    python3 tooling/setup_dataset.py uciml/iris
"""

import argparse
import os
import sys
import textwrap

try:
    import kagglehub
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pandas kagglehub")
    sys.exit(1)


RECOMMENDED = [
    {
        "name": "uciml/iris",
        "desc": "Iris flower species classification (3 classes, 150 rows)",
        "target_hint": "Species",
        "task": "classification",
    },
    {
        "name": "shyamnadhs/heart-disease-prediction-dataset",
        "desc": "Heart disease prediction (binary, 1000 rows, mixed features)",
        "target_hint": "disease",
        "task": "classification",
    },
    {
        "name": "altavish/boston-housing-dataset",
        "desc": "Boston housing price regression (506 rows, 13 features)",
        "target_hint": "MEDV",
        "task": "regression",
    },
    {
        "name": "mathan/fifa-2018-match-statistics",
        "desc": "FIFA 2018 match statistics (128 rows, 26 features)",
        "target_hint": "Man of the Match",
        "task": "classification",
    },
    {
        "name": "yasserh/student-marks-dataset",
        "desc": "Student marks prediction (100 rows, 2 features)",
        "target_hint": "Marks",
        "task": "regression",
    },
]


def pick_dataset() -> str:
    print("Recommended Kaggle datasets for benchmarking:\n")
    for i, d in enumerate(RECOMMENDED, 1):
        print(f"  {i:2d}) {d['name']:25s}  {d['desc']}")
    print(f"  q)  Enter a custom dataset name")

    while True:
        choice = input("\nSelect a dataset [1-{0} or name]: ".format(len(RECOMMENDED))).strip()
        if not choice:
            continue
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(RECOMMENDED):
                return RECOMMENDED[idx]["name"]
        except ValueError:
            pass
        if choice.lower() == "q":
            custom = input("Enter Kaggle dataset name (owner/name): ").strip()
            if custom:
                return custom
            continue
        # Maybe they typed a full name directly
        if "/" in choice:
            return choice
        print(f"  Invalid. Enter 1-{len(RECOMMENDED)} or a dataset name like 'owner/name'.")


def download_dataset(name: str, silent: bool = False) -> str:
    if not silent:
        print(f"\nDownloading {name}...", end=" ", flush=True)
    try:
        path = kagglehub.dataset_download(name)
        if not silent:
            print("done")
        return path
    except Exception as e:
        if silent:
            raise RuntimeError(f"Failed to download '{name}': {e}") from e
        print(f"\nError downloading '{name}': {e}")
        sys.exit(1)


def find_csv(path: str) -> str:
    for f in os.listdir(path):
        if f.endswith(".csv"):
            return os.path.join(path, f)
    raise FileNotFoundError(f"No CSV files found in dataset. Contents: {os.listdir(path)}")


def auto_detect_target(df: pd.DataFrame, hint: str | None = None) -> str:
    if hint and hint in df.columns:
        return hint
    candidates = ["target", "label", "class", "y", "outcome", "answer", "species", "Survived"]
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[-1]


def describe_column(col: str, dtype, nulls: int, n_unique: int, n_rows: int, is_target: bool) -> str:
    marker = "★ TARGET" if is_target else " feature"
    extra = ""
    if not is_target and n_unique <= 10:
        extra = " [low-card, maybe useful as target?]"
    if n_unique == n_rows:
        extra += " [unique ID column]"
    return f"  {col:20s}  {str(dtype):10s}  {nulls:4} nulls  {n_unique:4} unique  → {marker}{extra}"


def analyze(df: pd.DataFrame, target_col: str) -> dict:
    is_class = df[target_col].nunique() <= 10
    task = "classification" if is_class else "regression"
    return {"is_classification": is_class, "task": task}


def show_dataset_info(name: str, csv_path: str, df: pd.DataFrame, target_col: str):
    file_name = os.path.basename(csv_path)
    rows, cols = df.shape
    print(f"\nDataset: {name}")
    print(f"File:    {file_name}  ({rows:,} rows, {cols} columns)")
    print(f"\nColumns:\n")

    n_rows = len(df)
    for col in df.columns:
        dtype = df[col].dtype
        nulls = int(df[col].isna().sum())
        n_unique = int(df[col].nunique())
        is_target = col == target_col
        print(describe_column(col, dtype, nulls, n_unique, n_rows, is_target))

    print(f"\nPreview (first 5 rows):\n")
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)
    pd.set_option("display.max_colwidth", 30)
    print(df.head().to_string(index=False))


def suggest_command(name: str, target_col: str, analysis: dict):
    task = analysis["task"]
    print(f"\n{'=' * 60}")
    print(f"Task type: {task}")
    print(f"Target:    {target_col}")
    print(f"\nRun the benchmark with:\n")
    print(f"  KAGGLE_DATASET={name} TARGET_COLUMN={target_col} python3 adapter.py")
    print()
    print(f"Or with mesocosm:\n")
    print(f"  KAGGLE_DATASET={name} TARGET_COLUMN={target_col} \\")
    print(f"    mesocosm run local --manifest tooling/benchanything.json")

    show_all_flag = ""
    if not analysis["is_classification"]:
        show_all_flag = "  # regression: score is R² (clamped to 0-1)"
    print(f"\n  Expected good score: ~0.95+ accuracy{show_all_flag}")


def get_columns_info(df: pd.DataFrame, target_col: str) -> list[dict]:
    n_rows = len(df)
    columns = []
    for col in df.columns:
        columns.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "nulls": int(df[col].isna().sum()),
            "unique": int(df[col].nunique()),
            "is_target": col == target_col,
            "is_id": int(df[col].nunique()) == n_rows,
        })
    return columns


def build_command_str(config: dict) -> str:
    env_vars = []
    if config.get("dataset") and not config.get("toy_data"):
        env_vars.append(f"KAGGLE_DATASET={config['dataset']}")
    if config.get("target_column"):
        env_vars.append(f"TARGET_COLUMN={config['target_column']}")
    if config.get("test_size"):
        env_vars.append(f"TEST_SIZE={config['test_size']}")
    if config.get("max_steps"):
        env_vars.append(f"MAX_STEPS={config['max_steps']}")
    if config.get("max_fails"):
        env_vars.append(f"MAX_FAILS={config['max_fails']}")
    if config.get("toy_data"):
        env_vars.append("KAGGLE_TOY_DATA=1")

    mode = config.get("mode", "local")
    prefix = (" \\\n  ".join(env_vars) + " \\\n  ") if env_vars else ""
    model = config.get("model", "ollama/llama3.2")
    episodes = config.get("episodes", 1)

    if mode == "platform":
        domain = config.get("domain_id", "YOUR_DOMAIN_ID")
        vow = config.get("vow_version", "1.0.0")
        cmd = f"mesocosm run create --domain {domain} --vow-version {vow} --model {model} --episodes {episodes}"
    else:
        cmd = f"mesocosm run local --model {model} --manifest tooling/benchanything.json --episodes {episodes}"

    if prefix:
        return f"{prefix}{cmd}"
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Setup a Kaggle dataset for the prediction benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python3 tooling/setup_dataset.py
              python3 tooling/setup_dataset.py uciml/iris
              python3 tooling/setup_dataset.py --target Species uciml/iris
        """),
    )
    parser.add_argument("dataset", nargs="?", help="Kaggle dataset name (owner/name)")
    parser.add_argument("--target", "-t", help="Target column name (auto-detected if omitted)")
    args = parser.parse_args()

    name = args.dataset or pick_dataset()
    hint = args.target

    try:
        path = download_dataset(name)
        csv_path = find_csv(path)
    except (RuntimeError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    df = pd.read_csv(csv_path)

    # Auto-detect target
    target_col = hint or auto_detect_target(df, next(
        (d["target_hint"] for d in RECOMMENDED if d["name"] == name), None
    ))
    if target_col not in df.columns:
        print(f"\nTarget column '{target_col}' not found.")
        print(f"Available columns: {list(df.columns)}")
        target_col = auto_detect_target(df, None)
        print(f"Using '{target_col}' instead.")

    analysis = analyze(df, target_col)
    show_dataset_info(name, csv_path, df, target_col)

    # Show other target candidates
    candidates = []
    for col in df.columns:
        if col == target_col:
            continue
        nu = df[col].nunique()
        if nu <= 20:
            candidates.append((col, nu))
    if candidates:
        candidates.sort(key=lambda x: x[1])
        print(f"\nOther low-cardinality columns (potential targets):")
        for col, nu in candidates[:5]:
            print(f"  {col:20s}  ({nu} unique)")

    suggest_command(name, target_col, analysis)


if __name__ == "__main__":
    main()
