
import argparse
import datetime as dt
import pathlib
import pandas as pd
from typing import Callable, List

from src.config import USER_ID_COL, STREAMER_ID_COL

def filter_interactions(
    df: pd.DataFrame,
    filters: List[Callable[[pd.DataFrame], pd.Series]] = None
) -> pd.DataFrame:
    """
    Apply a list of filter functions to the DataFrame.
    Each filter should return a boolean Series (True = keep).
    Logs the number of rows filtered by each condition.
    """
    if filters is None:
        filters = []
    before = len(df)
    mask = pd.Series([True] * before, index=df.index)
    for i, f in enumerate(filters):
        filter_mask = f(df)
        filtered_out = (~filter_mask & mask).sum()
        print(f"Filter {i+1}: filtered out {filtered_out} rows")
        mask &= filter_mask
    after = mask.sum()
    print(f"Total filtered: {before - after} (from {before} to {after})")
    return df[mask]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Raw interactions CSV path")
    parser.add_argument("--filter-conditions", default=None, choices=["enter", "donate"])
    parser.add_argument("--outdir", default="data/processed/interactions", help="Output directory root")
    parser.add_argument("--date", default=dt.date.today().isoformat(), help="Date suffix for parquet filename")
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv, dtype={"pfid":"int64", "anchor_id":"int64"}) # pfid: user ID, anchor_id: streamer ID
    df = df.dropna(subset=["pfid", "anchor_id"])
    
    # rename columns for consistency
    df = df.rename(columns={"pfid": USER_ID_COL, "anchor_id": STREAMER_ID_COL})

    # filter out testing streamer ids (assuming 0 is a test ID)
    df = df[df[STREAMER_ID_COL] != 0]

    # apply filters based on the filter conditions
    if args.filter_conditions == "enter":
        filters = [
            lambda df: df["ent_live_cnt"] != 0.0,  # only keep entries with non-zero ent_live_cnt
            lambda df: df["watch_ts"] != 0.0,      # only keep entries with non-zero watch_ts
        ]
    elif args.filter_conditions == "donate":
        filters = [
            lambda df: (df["consume_cnt"] > 0) | (df["prod_total"] > 0),  # keep entries with donations or products
        ]
    else:
        filters = []

    # apply the filters
    df = filter_interactions(df, filters=filters)

    df = df.groupby([USER_ID_COL, STREAMER_ID_COL], as_index=False).first()

    # write parquet
    out_path = outdir / f"{args.date}.parquet"
    print(f"› Writing {out_path} …")
    df.to_parquet(out_path, index=False)
    
    # update symlink
    latest = outdir / "latest.parquet"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out_path.name)
    except OSError:
        # On Windows symlink may require admin; fallback to copy
        import shutil
        shutil.copy(out_path, latest)

    print("✓ Interactions parquet built:", out_path)

if __name__ == "__main__":
    main()
