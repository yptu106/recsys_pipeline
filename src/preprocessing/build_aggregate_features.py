"""
build_aggregate_features.py

Build streamer and user aggregated features from interaction logs.

The output is written to:
    features/streamer/<YYYY-MM-DD>.parquet
    features/user/<YYYY-MM-DD>.parquet
and a symlink `latest.parquet` is updated for downstream jobs.

Usage:

python -m src.preprocessing.build_aggregate_features \
    --user-interactions <path_to_user_interactions (should come from training set to prevent data leakage)> \
    --out-dir features

Notes:
- The script ensures all streamer IDs are unique and all tag dictionaries have the same keys.
- To prevent data leakage, the aggregated features should be built only on the training set interactions.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import pathlib
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import USER_ID_COL, STREAMER_ID_COL

def _streamer_aggregated_features(interaction_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and normalize aggregated features for each streamer from the interaction logs.
    Returns a DataFrame indexed by streamer_id with normalized columns:
    i_watch_tot, i_watch_cnt, i_unique_user, i_live_cnt, i_followers, i_gift_amt, i_watch_avg, i_pop_z
    """
    item_df = (
        interaction_df.groupby(STREAMER_ID_COL).agg(
            i_watch_tot=("watch_ts", "sum"),
            i_watch_cnt=("watch_ts", "size"),
            i_unique_user=(USER_ID_COL, "nunique"),
            # i_live_cnt  =("live_cnt", "max"),      # already monthly total
            # i_followers =("is_follow", "sum"),
            i_gift_amt  =("prod_total", "sum"),
        )
        .assign(
            i_watch_avg = lambda d: d.i_watch_tot / d.i_watch_cnt,
            i_pop_z     = lambda d: ((d.i_watch_cnt - d.i_watch_cnt.mean()) 
                                    / d.i_watch_cnt.std()),
        )
    )

    # define columns to normalize
    log_norm_cols = [
        "i_watch_tot", "i_watch_cnt", "i_unique_user", 
        "i_gift_amt", "i_watch_avg"
    ]
    # direct_norm_cols = ["i_live_cnt"]

    # Log1p transform skewed columns
    for col in log_norm_cols:
        item_df[col] = np.log1p(item_df[col])

    scaler = StandardScaler()
    # scaled_cols = log_norm_cols + direct_norm_cols
    scaled_cols = log_norm_cols
    item_df[scaled_cols] = scaler.fit_transform(item_df[scaled_cols])

    return item_df.reset_index()

def _user_aggregated_features(interaction_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and normalize aggregated features for each user from the interaction logs.
    """
    user_df = (
        interaction_df.groupby(USER_ID_COL).agg(
            u_watch_tot=("watch_ts", "sum"),
            u_watch_cnt=("watch_ts", "size"),
            # u_gift_cnt =("consume_cnt", "sum"),
            u_gift_amt =("prod_total", "sum"),
            # u_follow_cnt=("is_follow", "sum"),
        )
    )

    # define columns to normalize
    log_norm_cols = [
        "u_watch_tot", "u_watch_cnt", "u_gift_amt"
    ]
    # direct_norm_cols = ["u_follow_cnt"]
    
    # Log1p transform skewed columns
    for col in log_norm_cols:
        user_df[col] = np.log1p(user_df[col])

    scaler = StandardScaler()
    # scaled_cols = log_norm_cols + direct_norm_cols
    scaled_cols = log_norm_cols
    user_df[scaled_cols] = scaler.fit_transform(user_df[scaled_cols])

    return user_df.reset_index()

def update_latest_symlink(latest_path: pathlib.Path, target_path: pathlib.Path) -> None:
    """
    Point `latest_path` at `target_path`. Prefer a relative symlink;
    fall back to copy if symlinks aren’t supported.
    """
    try:
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        # use a relative symlink so moving the folder still works
        rel_target = target_path.relative_to(latest_path.parent)
        latest_path.symlink_to(rel_target)
    except OSError:
        # Windows or restricted environments → copy
        shutil.copyfile(target_path, latest_path)

def write_parquet(df: pd.DataFrame, out_dir: pathlib.Path, date: dt) -> None:
    out_path = out_dir / f"{date}.parquet"

    print(f"› Writing {out_path} …")
    df.to_parquet(out_path, index=False)

    # Update symlink
    latest = out_dir / "latest.parquet"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out_path.name)
    except OSError:
        # On Windows symlink may require admin; fallback to copy
        import shutil
        shutil.copy(out_path, latest)

    print("✓ Feature parquet built:", out_path)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-interactions", required=True, help="User interactions parquet path")
    parser.add_argument("--out-dir", default="features", help="Output directory root")
    parser.add_argument("--date", default=dt.date.today().isoformat(), help="Date suffix for parquet filename")
    args = parser.parse_args()  

    # Load user interactions to compute aggregated features
    print("› Loading user interactions …")
    interactions_df = pd.read_parquet(args.user_interactions)
    interactions_df.rename(columns={"pfid": USER_ID_COL, "anchor_id": STREAMER_ID_COL}, inplace=True)

    item_df = _streamer_aggregated_features(interactions_df)
    user_df = _user_aggregated_features(interactions_df)

    # Create output directory
    outdir = pathlib.Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    streamer_dir = outdir / "streamer"
    streamer_dir.mkdir(parents=True, exist_ok=True)

    write_parquet(
        item_df, 
        out_dir=streamer_dir, 
        date=args.date,
    )

    user_dir = outdir / "user"
    user_dir.mkdir(parents=True, exist_ok=True)

    write_parquet(
        user_df, 
        out_dir=user_dir, 
        date=args.date,
    )

if __name__ == "__main__":
    main()