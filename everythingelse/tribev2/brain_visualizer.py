"""Brain visualizer for TRIBE v2 predictions.

Usage:
    python brain_visualizer.py --text "To be or not to be"
    python brain_visualizer.py --video path/to/video.mp4
    python brain_visualizer.py --audio path/to/audio.wav
    python brain_visualizer.py --text "Hello world" --save_gif animation.gif
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

# Ensure ffmpeg is on PATH (winget installs to user PATH)
_user_path = os.environ.get("PATH", "")
_machine_path = ""
try:
    import ctypes
    _machine_path = ctypes.create_string_buffer(32767)
    ctypes.windll.kernel32.GetEnvironmentVariableA("PATH", _machine_path, 32767)
    _machine_path = _machine_path.value.decode() if _machine_path.value else ""
except Exception:
    pass
for _p in [_user_path, _machine_path]:
    os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")

if not shutil.which("ffmpeg"):
    _candidates = list(Path(os.environ.get("LOCALAPPDATA", "C:")).glob("**/ffmpeg.exe"))
    if _candidates:
        os.environ["PATH"] = str(_candidates[0].parent) + os.pathsep + os.environ["PATH"]

os.environ["TQDM_DISABLE"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TRIBE_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(TRIBE_PATH))

from tribev2 import TribeModel
from tribev2.plotting import PlotBrain


def parse_args():
    parser = argparse.ArgumentParser(description="TRIBE v2 Brain Visualizer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Input text string")
    group.add_argument("--text_file", type=str, help="Path to .txt file")
    group.add_argument("--video", type=str, help="Path to video file (.mp4, .avi, etc.)")
    group.add_argument("--audio", type=str, help="Path to audio file (.wav, .mp3, etc.)")
    parser.add_argument("--cache", type=str, default=str(TRIBE_PATH / "cache"),
                        help="Cache folder for model features")
    parser.add_argument("--mesh", type=str, default="fsaverage5",
                        choices=["fsaverage3", "fsaverage4", "fsaverage5", "fsaverage6", "fsaverage7"],
                        help="Cortical mesh resolution")
    parser.add_argument("--views", type=str, nargs="+",
                        default=["left", "right", "dorsal", "ventral"],
                        help="Brain views to render")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Number of timesteps to visualize (default: all)")
    parser.add_argument("--cmap", type=str, default="fire", help="Colormap")
    parser.add_argument("--norm_percentile", type=float, default=99, help="Robust normalization percentile")
    parser.add_argument("--vmin", type=float, default=None, help="Min colorbar value")
    parser.add_argument("--alpha_cmap", type=float, nargs=2, default=None,
                        help="Alpha cmap threshold and scale (e.g. 0 .2)")
    parser.add_argument("--save", type=str, default=None, help="Save single-timestep plot to path")
    parser.add_argument("--save_grid", type=str, default=None, help="Save multi-timestep grid to path")
    parser.add_argument("--save_gif", type=str, default=None, help="Save rotating brain GIF to path")
    parser.add_argument("--show_stimuli", action="store_true", help="Show stimulus frames in grid")
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"], help="Torch device")
    return parser.parse_args()


def get_input_path(args) -> tuple[str, str]:
    if args.text:
        p = Path(TRIBE_PATH / "cache" / "_input.txt")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(args.text)
        return "text_path", str(p)
    if args.text_file:
        return "text_path", args.text_file
    if args.video:
        return "video_path", args.video
    if args.audio:
        return "audio_path", args.audio


def main():
    args = parse_args()
    CACHE_FOLDER = Path(args.cache)
    CACHE_FOLDER.mkdir(parents=True, exist_ok=True)

    print("Loading TRIBE v2 model...")
    t0 = time.time()
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=str(CACHE_FOLDER),
        device=args.device,
    )
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    print("Initializing brain plotter...")
    plotter = PlotBrain(mesh=args.mesh)

    kw, path = get_input_path(args)
    print(f"Building events dataframe from {kw}={path}...")
    print("  (First run may download WhisperX large-v3 model (~3GB) for transcription)")
    sys.stdout.flush()
    t0 = time.time()
    df = model.get_events_dataframe(**{kw: path})
    print(f"  Events built in {time.time()-t0:.1f}s")
    print(df.head(8)[["type", "start", "duration", "filepath", "text", "context"]])

    print("Running prediction...")
    t0 = time.time()
    preds, segments = model.predict(events=df)
    print(f"  Prediction done in {time.time()-t0:.1f}s")
    print(f"Predictions shape: {preds.shape}")

    n_timesteps = args.timesteps or min(len(preds), 15)
    preds_subset = preds[:n_timesteps]
    seg_subset = segments[:n_timesteps] if segments else None

    plot_kwargs = dict(
        cmap=args.cmap,
        norm_percentile=args.norm_percentile,
        vmin=args.vmin,
    )
    if args.alpha_cmap is not None:
        plot_kwargs["alpha_cmap"] = tuple(args.alpha_cmap)

    if args.save_gif:
        print(f"Saving rotating GIF to {args.save_gif}...")
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw={"projection": "3d"})
        plotter.plot_surf(preds_subset[0], axes=[ax], views=["left"], **plot_kwargs)
        plotter.save_gif(ax, save_path=args.save_gif)
        plt.close(fig)
        print(f"Saved {args.save_gif}")

    if args.save is not None:
        print(f"Saving single-timestep plot to {args.save}...")
        fig = plotter.plot_timesteps(
            preds_subset[:1],
            segments=seg_subset[:1] if seg_subset else None,
            views="left",
            show_stimuli=args.show_stimuli,
            **plot_kwargs,
        )
        fig.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {args.save}")

    if args.save_grid is not None:
        print(f"Saving multi-timestep grid to {args.save_grid}...")
        fig = plotter.plot_timesteps(
            preds_subset,
            segments=seg_subset if seg_subset else None,
            views="left",
            show_stimuli=args.show_stimuli,
            **plot_kwargs,
        )
        fig.savefig(args.save_grid, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {args.save_grid}")

    if not any([args.save, args.save_grid, args.save_gif]):
        print("\nNo output path specified. Displaying interactive plot...")
        matplotlib.use("TkAgg")
        fig = plotter.plot_timesteps(
            preds_subset,
            segments=seg_subset if seg_subset else None,
            views="left",
            show_stimuli=args.show_stimuli,
            **plot_kwargs,
        )
        plt.show()

    print("Done.")


if __name__ == "__main__":
    main()
