import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tribev2 import TribeModel


def main():
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
    df = model.get_events_dataframe(text_path="./test_text.txt")
    preds, segments = model.predict(events=df)
    print(preds.shape)  # (n_timesteps, n_vertices)


if __name__ == "__main__":
    main()