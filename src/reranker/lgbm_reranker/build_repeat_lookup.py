# scripts/build_repeat_lookup.py  (run once)
import pandas as pd, pickle, pathlib, json
TRAIN_INTER = pathlib.Path("data/splits/donate/w_ts_heavy/train.parquet")

train_df = pd.read_parquet(TRAIN_INTER)          # cols: user_id, streamer_id, …
repeat = (
    train_df
    .groupby("user_id")["streamer_id"]
    .apply(set)                                  # per-user set for O(1) look-ups
    .to_dict()
)

pickle.dump(repeat, open("reranker/repeat_lookup.pkl", "wb"))
