
import argparse
import datetime as dt
import pathlib
import pandas as pd
from typing import Callable, List

from src.config import USER_ID_COL, STREAMER_ID_COL, TIMESTAMP_COL

# def event_time_to_timestamp(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Convert 'event_time' column to UNIX timestamp in seconds.
#     If 'event_time' is not present, return the DataFrame unchanged.
#     """
#     if "event_time" in df.columns:
#         print("Converting 'event_time' to UNIX timestamp...")
#         df["timestamp"] = pd.to_datetime(df["event_time"], errors="coerce").astype("int64") // 10**9
#         df = df.drop(columns=["event_time"])
#     return df

def deduplicate_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate interactions by keeping the first entry for each user-streamer pair.
    This ensures a unique interaction per user-streamer pair.
    """
    if "event_time" in df.columns:
        # map 'event_time' to a datetime column
        df["event_time"] = pd.to_datetime(
            df["event_time"],          # works for both ISO strings and numeric epoch
            errors="coerce",           # bad rows → NaT (you can drop them later)
            utc=True                   # keep a single time-zone; optional
        )
        # round timestamps down to the day
        df["event_time"] = df["event_time"].dt.floor("D")

        # convert to UNIX timestamp in seconds
        df[TIMESTAMP_COL] = df["event_time"].astype("int64") // 10**9
        df = df.drop(columns=["event_time"])

        agg = {
            "prod_total": "sum",
            "watch_ts": "sum",
            "follow": "sum",
        }
        
        grouped = (
            df.groupby([USER_ID_COL, STREAMER_ID_COL, TIMESTAMP_COL], as_index=False)
            .agg(**{k: (k, v) for k, v in agg.items()})             # named-agg syntax
            .sort_values(TIMESTAMP_COL)
        )

        return grouped

    return df.groupby([USER_ID_COL, STREAMER_ID_COL], as_index=False).first().sort_values(TIMESTAMP_COL)

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
    parser.add_argument("--out-dir", default="data/processed/interactions", help="Output directory root")
    parser.add_argument("--date", default=dt.date.today().isoformat(), help="Date suffix for parquet filename")
    args = parser.parse_args()

    outdir = pathlib.Path(args.out_dir)
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
        if "consume_cnt" in df.columns and "prod_total" in df.columns:
            filters = [
                lambda df: (df["consume_cnt"] > 0) | (df["prod_total"] > 0),  # keep entries with donations or products
            ]
        elif "prod_total" in df.columns:
            filters = [
                lambda df: (df["prod_total"] > 0),  # keep entries with products
            ]
        elif "consume_cnt" in df.columns:
            filters = [
                lambda df: (df["consume_cnt"] > 0),  # keep entries with donations
            ]
        else:
            print("Warning: No donation/product columns found. Skipping donation filters.")
            filters = []
    else:
        filters = []

    # apply the filters
    df = filter_interactions(df, filters=filters)

    # # convert event_time to timestamp
    # df = event_time_to_timestamp(df)

    # deduplicate interactions
    df = deduplicate_interactions(df)
    print(f"Interactions after deduplication: {len(df)} rows")

    # group by user_id and streamer_id, keeping the first entry for each pair
    # to have a unique interaction per user-streamer pair
    # df = df.groupby([USER_ID_COL, STREAMER_ID_COL], as_index=False).first()

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
