#!/usr/bin/env python3
"""Clean up one or more trial directories and remove imported modules."""

import argparse
import os
import shutil
import sys


def cleanup_trial(trial_dir: str, remove_modules: bool = True, dry_run: bool = False) -> None:
    trial_abspath = os.path.abspath(trial_dir)

    if not os.path.isdir(trial_abspath):
        print(f"Not found: {trial_abspath}")
        return

    # Remove any Python modules that were imported from this trial directory
    if remove_modules:
        to_remove = []
        for name, mod in list(sys.modules.items()):
            if hasattr(mod, "__file__") and mod.__file__:
                try:
                    mod_path = os.path.abspath(mod.__file__)
                    if mod_path.startswith(trial_abspath):
                        to_remove.append(name)
                except Exception:
                    pass
        if to_remove:
            if dry_run:
                print(f"Would remove modules: {', '.join(to_remove)}")
            else:
                for name in to_remove:
                    del sys.modules[name]
                print(f"Removed {len(to_remove)} module(s) from sys.modules")

    # Remove the trial directory
    if dry_run:
        print(f"Would remove directory: {trial_abspath}")
    else:
        shutil.rmtree(trial_abspath)
        print(f"Removed directory: {trial_abspath}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean up trial directories and imported modules."
    )
    parser.add_argument(
        "trials",
        nargs="+",
        help="Trial directories to clean (accepts glob patterns like tooling/trials/trial_*)",
    )
    parser.add_argument(
        "--no-modules",
        action="store_true",
        help="Skip cleaning imported modules",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Print what would be done without actually deleting",
    )
    args = parser.parse_args()

    import glob
    expanded = []
    for pattern in args.trials:
        matches = glob.glob(pattern)
        if matches:
            expanded.extend(matches)
        else:
            expanded.append(pattern)

    for trial in expanded:
        cleanup_trial(trial, remove_modules=not args.no_modules, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
