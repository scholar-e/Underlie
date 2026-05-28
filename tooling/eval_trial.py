#!/usr/bin/env python3
"""Examine how well an AI performed in one or more trials."""

import argparse
import json
import os
import sys


def eval_trial(trial_dir: str, show_programs: bool = False, detailed: bool = False) -> dict | None:
    trial_abspath = os.path.abspath(trial_dir)

    results_path = os.path.join(trial_abspath, "results.json")
    meta_path = os.path.join(trial_abspath, "meta.json")

    if not os.path.isfile(results_path):
        print(f"No results.json found in {trial_abspath}")
        return None

    with open(results_path) as f:
        results = json.load(f)

    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    # Summary line
    dataset = results.get("dataset", meta.get("dataset", "?"))
    target = results.get("target_column", meta.get("target_column", "?"))
    best = results.get("best_score", 0)
    steps = results.get("total_steps", 0)
    attempts = results.get("total_attempts", steps)
    task_type = "cls" if results.get("is_classification", meta.get("is_classification", False)) else "reg"

    print(f"{os.path.basename(trial_abspath)}")
    print(f"  Dataset: {dataset}")
    print(f"  Target:  {target}  ({task_type})")
    print(f"  Best:    {best:.4f}")
    print(f"  Steps:   {steps} ok / {attempts} total")
    print()

    if detailed:
        if meta:
            print("  Metadata:")
            for k, v in meta.items():
                print(f"    {k}: {v}")
            print()

    # Per-step breakdown
    step_list = results.get("steps", [])
    if step_list:
        print("  Steps:")
        for i, s in enumerate(step_list):
            score = s.get("score", 0)
            best_at_step = s.get("best_score", score)
            crashed = s.get("crashed", False)
            status = " CRASH" if crashed else " OK   "
            stderr = (s.get("stderr") or "").strip()
            stderr_short = stderr[:120] if stderr else ""
            print(f"    {i:3d}  [{status}]  score={score:.4f}  best={best_at_step:.4f}"
                  + (f"  {stderr_short}" if stderr_short else ""))

        print()

    if best > 0 and show_programs:
        best_prog = os.path.join(trial_abspath, "best_program.py")
        if os.path.isfile(best_prog):
            print(f"  Best program ({best:.4f}):")
            with open(best_prog) as f:
                for line in f:
                    print(f"    {line}", end="")
            print()

    # Compare predictions vs expected
    if detailed:
        expected_path = os.path.join(trial_abspath, "expected.json")
        if os.path.isfile(expected_path):
            with open(expected_path) as f:
                expected = json.load(f)
            print(f"  Expected values ({len(expected)} total): first 10 shown")
            print(f"    {expected[:10]}")
            print()

            # Find best step's predictions
            for s in reversed(step_list):
                score = s.get("score", 0)
                if score >= best and not s.get("crashed", False):
                    step_idx = step_list.index(s)
                    pred_path = os.path.join(trial_abspath, f"step_{step_idx:03d}", "predictions.json")
                    if os.path.isfile(pred_path):
                        with open(pred_path) as f:
                            predictions = json.load(f)
                        correct = sum(1 for p, e in zip(predictions, expected) if abs(p - e) < 0.01)
                        print(f"  Best-step predictions (step {step_idx}, score={score:.4f}): first 10 shown")
                        print(f"    {predictions[:10]}")
                        print(f"  Exact matches: {correct}/{len(expected)}")
                    break

    return results


def eval_all(trials_base: str, **kwargs) -> None:
    if not os.path.isdir(trials_base):
        print(f"Trials directory not found: {trials_base}")
        return

    all_results = []
    for entry in sorted(os.listdir(trials_base)):
        trial_dir = os.path.join(trials_base, entry)
        if os.path.isdir(trial_dir) and entry.startswith("trial_"):
            result = eval_trial(trial_dir, **kwargs)
            if result:
                all_results.append(result)
            print("-" * 60)

    if all_results:
        print()
        print("Summary across all trials:")
        best_scores = [r.get("best_score", 0) for r in all_results]
        print(f"  Trials:        {len(all_results)}")
        print(f"  Best score:    {max(best_scores):.4f}")
        print(f"  Worst score:   {min(best_scores):.4f}")
        print(f"  Average score: {sum(best_scores) / len(best_scores):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Examine trial results from the Kaggle prediction benchmark."
    )
    parser.add_argument(
        "trials",
        nargs="*",
        default=None,
        help="Trial directories to evaluate (default: all trials in tooling/trials/)",
    )
    parser.add_argument(
        "--programs",
        action="store_true",
        help="Show the best program for each trial",
    )
    parser.add_argument(
        "--detailed",
        "-d",
        action="store_true",
        help="Show metadata and prediction comparison",
    )
    args = parser.parse_args()

    if args.trials:
        import glob
        expanded = []
        for pattern in args.trials:
            matches = glob.glob(pattern)
            if matches:
                expanded.extend(matches)
            else:
                expanded.append(pattern)
        for trial in expanded:
            eval_trial(trial, show_programs=args.programs, detailed=args.detailed)
            print("-" * 60)
    else:
        trials_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trials")
        eval_all(trials_base, show_programs=args.programs, detailed=args.detailed)


if __name__ == "__main__":
    main()
