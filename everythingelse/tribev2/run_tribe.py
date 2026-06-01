import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tribev2 import TribeModel


def main():
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
    df = model.get_events_dataframe(text_path="./test_text.txt")
    preds, segments = model.predict(events=df)

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "predictions.npy", preds)

    # Save segment info (onset, duration, text for each segment)
    import pandas as pd
    rows = []
    for s in segments:
        rows.append({"start": s.start, "duration": s.duration, "text": getattr(s, "text", "")})
    pd.DataFrame(rows).to_csv(out_dir / "segments.csv", index=False)

    print(f"Saved predictions.npy ({preds.shape}) and segments.csv to {out_dir}/")


if __name__ == "__main__":
    import numpy as np
    main()