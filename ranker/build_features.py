# features/ranker/build_features.py

import pandas as pd
import numpy as np
import argparse
import pathlib

from src.config import USER_ID_COL, STREAMER_ID_COL

def build_user_item_features(interaction_logs: pd.DataFrame):
    user_df = (
        interaction_logs.groupby(USER_ID_COL).agg(
            u_watch_tot=("watch_ts", "sum"),
            u_watch_cnt=("watch_ts", "size"),
            u_gift_cnt =("consume_cnt", "sum"),
            u_gift_amt =("prod_total", "sum"),
            u_follow_cnt=("is_follow", "sum"),
        )
    )

    item_df = (
        interaction_logs.groupby(STREAMER_ID_COL).agg(
            i_watch_tot=("watch_ts", "sum"),
            i_watch_cnt=("watch_ts", "size"),
            i_unique_user=("user_id", "nunique"),
            i_live_cnt  =("live_cnt", "max"),
            i_followers =("is_follow", "sum"),
            i_gift_amt  =("prod_total", "sum"),
        )
        .assign(
            i_watch_avg = lambda d: d.i_watch_tot / d.i_watch_cnt,
            i_pop_z     = lambda d: (d.i_watch_cnt - d.i_watch_cnt.mean()) / (d.i_watch_cnt.std(ddof=0) + 1e-6),
        )
    )

    return user_df, item_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interaction-logs", type=str, required=True, help="Path to interaction logs (parquet)")
    parser.add_argument("--train-split", type=str, default=None, help="Path to train split (parquet)")
    parser.add_argument("--output-dir", type=str, default="features/ranker/")
    args = parser.parse_args()

    logs_df = pd.read_parquet(args.interaction_logs)
    print(f"Loaded {len(logs_df)} interaction logs")

    if args.train_split:
        print(f"Loading train split from {args.train_split}")
        train_df = pd.read_parquet(args.train_split)
        logs_df = logs_df.merge(
            train_df[[USER_ID_COL, STREAMER_ID_COL]], 
            on=[USER_ID_COL, STREAMER_ID_COL], 
            how="inner"
        )
        print(f"Filtered logs to {len(logs_df)} interactions in train split")

    # Build user and item features
    print("Building user and item features...")
    user_features, item_features = build_user_item_features(logs_df)

    # Save features
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    user_out = output_dir / f"user.parquet"
    item_out = output_dir / f"item.parquet"
    user_features.to_parquet(user_out)
    item_features.to_parquet(item_out)

    print(f"> Saved user features to {user_out}")
    print(f"> Saved item features to {item_out}")

if __name__ == "__main__":
    main()
