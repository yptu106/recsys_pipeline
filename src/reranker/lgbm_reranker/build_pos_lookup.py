# scripts/build_gt_lookup.py
import pandas as pd, pickle, pathlib

VAL_FILE = pathlib.Path("data/splits/donate/w_ts_heavy/val.parquet") 
val_df = pd.read_parquet(VAL_FILE)                  # cols: user_id, streamer_id, label
val_df = val_df[val_df.label == 1]               # only keep positive samples

gt_val = (
    val_df.groupby("user_id")["streamer_id"]
          .apply(set)            # per-user set of *true* positives
          .to_dict()
)

pickle.dump(gt_val, open("reranker/gt_val_lookup.pkl", "wb"))
