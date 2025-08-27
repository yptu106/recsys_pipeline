"""
build_features.py

Construct streamer item sentence from raw streamer CSV.

The output is written to:
    features/item_sentence/<YYYY-MM-DD>.parquet
and a symlink `latest.parquet` is updated for downstream jobs.

Features included:
- pfid: Streamer ID
- item_sentence: Flattened string of all tag key-value pairs (e.g., "gender 女 personality 活潑、開朗 ...")

Usage:

python -m preprocessing.build_item_sentence \
    --streamers-csv data/raw/streamers.csv \
    --out-dir features/item_sentence

Notes:
- The script ensures all streamer IDs are unique and all tag dictionaries have the same keys.
- The original tags column is dropped in the output.
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

gender_map = {
    '女': '女',
    '女性': '女',
    '女生': '女',
    'woman': '女',
    'female': '女',
    '男': '男',
    '男性': '男',
    '男生': '男',
    'man': '男',
    'male': '男'
}

def _normalize_tags(tags: dict) -> dict:
    tags = tags.copy()
    if "gender" in tags:
        raw_gender = tags["gender"].strip().lower()
        if raw_gender not in gender_map:
            print(f"Unknow raw gender: {raw_gender}")
        tags["gender"] = gender_map.get(raw_gender, raw_gender) # fallback to original if not mapped
    return tags

def _flatten_tags(tags: dict) -> str:
    """
    Convert key-value pairs into a flat sequence like: "key value key value ..."
    """
    return " ".join(f"{key} {value}" for key, value in tags.items())

def _format_streamer_sentence(tags: dict) -> str:
    """
    Format the tags dictionary into a single sentence string.
    """
    gender = tags.get("gender", "女")  # default to "女" if not specified
    pronoun = "她" if gender == "女" else "他"
    possessive = "她的" if gender == "女" else "他的"
    gender_word = "女" if gender == "女" else "男"

    personality = tags.get("personality", "未知個性")
    appearance = tags.get("appearance", "外貌特徵不詳")
    talents = tags.get("talents", "才藝不詳")
    topics = tags.get("featured_topics", "多種主題")
    style = tags.get("live_streaming_style", "風格多樣")

    sentence = (
        f"{pronoun}是一位{gender_word}實況主，個性屬於 {personality}，外貌特徵為 {appearance}。"
        f"{pronoun}擅長 {talents}，直播內容常涵蓋 {topics} 等主題。"
        f"{possessive}直播風格為 {style}。"
    )
    return sentence

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
    parser.add_argument("--streamers-csv", required=True, help="Raw streamer CSV path")
    parser.add_argument("--out-dir", default="features/item_sentence", help="Output directory root")
    parser.add_argument("--date", default=dt.date.today().isoformat(), help="Date suffix for parquet filename")
    args = parser.parse_args()  

    # Load raw CSV
    print("› Loading CSV …")
    df = pd.read_csv(args.streamers_csv)

    # only keep `pfid` and `tags`
    df = df[["pfid", "tags"]]
    # convert the `tags` column from string to dictionary
    df["tags"] = df["tags"].apply(ast.literal_eval)

    # verify streamers are unique
    if df["pfid"].duplicated().any():
        raise ValueError("Streamer IDs (pfid) must be unique in the CSV")

    # assert that all tags have the same keys
    keys = df["tags"].apply(lambda x: set(x.keys()))
    if not all(keys == keys.iloc[0]):
        raise ValueError("All tags must have the same keys in the CSV")

    print(f"› Number of streamers: {len(df)}")

    # Rename `pfid` to `streamer_id` for consistency
    df.rename(columns={"pfid": STREAMER_ID_COL}, inplace=True)

    # Normalize tags
    print("› Normalizing tags …")
    df["tags"] = df["tags"].apply(_normalize_tags)

    # build item sentence
    print("› Flattening tags into item sentences …")
    df["item_sentence"] = df["tags"].apply(_flatten_tags)

    # build formatted sentences
    print("› Formatting item sentences …")
    df["format_sentence"] = df["tags"].apply(_format_streamer_sentence)

    df.drop(columns=["tags"], inplace=True) # remove the `tags` column as it's no longer needed

    # Create output directory
    outdir = pathlib.Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    write_parquet(
        df, 
        out_dir=outdir, 
        date=args.date,
    )

if __name__ == "__main__":
    main()