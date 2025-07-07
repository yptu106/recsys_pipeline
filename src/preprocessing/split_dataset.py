"""
split_dataset.py
Utility to split an implicit-feedback interaction table into
train / val / test sets per user.

Currently supports:
    • random_loo    – random leave-one-out (1 val, 1 test, rest train)
    • time_cutoff   – before / after a timestamp threshold  (if ts exists) (not implemented yet)

Negative sampling (uniform) for val / test is built-in.

Typical call from code
----------------------
from preprocessing.split_dataset import split_dataset
train,val,test = split_dataset(df, strategy=\"random_loo\", neg_per_user=100)

CLI
---
python -m preprocessing.split_dataset \\
       --interactions data/processed/interactions/latest.parquet \\
       --out_dir data/splits
"""

from __future__ import annotations
import argparse
import pathlib
import numpy as np
import pandas as pd
from numpy.random import default_rng

RNG = default_rng(seed=42)

def _sample_negatives(
    pos_df: pd.DataFrame,  # one positive row per user
    all_items: np.ndarray,
    neg_per_user: int,
) -> pd.DataFrame:
    """Return a DF of (user_id, streamer_id, label=0).
    
    For each user in the positive set, samples a specified number of
    negative items (streamer_ids) that are not in the user's positive set.
    """
    def sample_row(r):
        seen = {r.streamer_id}
        cand = np.setdiff1d(all_items, list(seen))
        negs = RNG.choice(cand, neg_per_user, replace=False)
        return pd.DataFrame({"user_id": r.user_id,
                             "streamer_id": negs,
                             "label": 0})

    negs = pd.concat([sample_row(r) for r in pos_df.itertuples(index=False)],
                     ignore_index=True)
    return negs


def split_dataset(
    df: pd.DataFrame,
    strategy: str = "random_loo",
    neg_per_user: int = 100,
    cutoff_ts: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split user-item interactions (+ optional timestamp) into train/val/test.

    df columns required:
        user_id : int64
        streamer_id    : int64
        # timestamp (optional, pandas datetime64[ns])
    All rows are assumed to be positive interactions.

    for each user, the validation (`val`) and test (`test`) DataFrames each contain at most one 
    positive interaction

    `train`: all positive interactions
    `test`: one positive interaction per user + N negatives
    `val`: at most one positive interaction per user (if available) + N negatives
    """
    assert {"user_id", "streamer_id"}.issubset(df.columns)

    if strategy == "random_loo":
        df = df.sample(frac=1, random_state=42)    # shuffle rows
        # choose 2 rows per user, if possible
        grp = df.groupby("user_id")
        test = grp.head(1)
        val  = grp.nth(1, dropna="any")  # second row, may be NaN
        train_idx = df.index.difference(test.index).difference(val.index)
        train = df.loc[train_idx]

    elif strategy == "time_cutoff":
        raise ValueError(f"time_cutoff strategy not implemented yet")
        # if cutoff_ts is None:
        #     raise ValueError("cutoff_ts required for time_cutoff strategy")
        # train = df[df.timestamp < cutoff_ts]
        # rest  = df[df.timestamp >= cutoff_ts]        # val+test candidates
        # # random split half/half
        # mask  = RNG.random(len(rest)) < 0.5
        # val   = rest[mask]
        # test  = rest[~mask]
    else:
        raise ValueError(f"Unknown strategy {strategy}")

    # add positive label
    for split in (train, val, test):
        split["label"] = 1

    # negative sampling for val/test
    all_items = df.streamer_id.unique()
    val = pd.concat([val, _sample_negatives(val, all_items, neg_per_user)])
    test = pd.concat([test, _sample_negatives(test, all_items, neg_per_user)])

    return train, val, test

def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactions", required=True,
                    help="interaction parquet with user_id, streamer_id[, timestamp]")
    parser.add_argument("--out_dir", required=True, help="output directory")
    parser.add_argument("--strategy", default="random_loo",
                    choices=["random_loo", "time_cutoff"])
    parser.add_argument("--neg-per-user", type=int, default=100)
    parser.add_argument("--cutoff-ts",
                    help="YYYY-MM-DD for time_cutoff strategy")
    args = parser.parse_args()

    df = pd.read_parquet(args.interactions, columns=["user_id", "streamer_id"])
    if args.strategy == "time_cutoff" and "timestamp" not in df.columns:
        raise SystemExit("timestamp column missing for time_cutoff split")

    train, val, test = split_dataset(
        df,
        strategy=args.strategy,
        neg_per_user=args.neg_per_user,
        cutoff_ts=args.cutoff_ts,
    )

    output_path = pathlib.Path(args.out_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    train.to_parquet(output_path / "train.parquet", compression="snappy")
    val  .to_parquet(output_path / "val.parquet",   compression="snappy")
    test .to_parquet(output_path / "test.parquet",  compression="snappy")
    print("✓ wrote", output_path)

if __name__ == "__main__":
    _cli()
