"""
build_global_list.py

Preprocesses a global list of streamers based on business logic (e.g., monthly top-100 streamers). 

The output is written to:
    data/global_list/<YYYY-MM-DD>.parquet
and a symlink `latest.parquet` is updated for downstream jobs.

This script assumes the input data is a CSV with a column "pfid" representing streamer IDs.
It renames this column to `STREAMER_ID_COL` for consistency with the rest of the pipeline.

"""

import argparse
import pandas as pd
import datetime as dt
import pathlib

from src.config import STREAMER_ID_COL

def main():
    parser = argparse.ArgumentParser(description="Build global streamer list from interactions.")
    parser.add_argument("--input-path", type=str, help="Path to the input interactions file (CSV).")
    parser.add_argument("--output-dir", type=str, default="data/global_list/", help="Path to save the global streamer list (Parquet).")
    parser.add_argument("--date", default=dt.date.today().isoformat(), help="Date suffix for parquet filename")
    args = parser.parse_args()

    outdir = pathlib.Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_path, dtype={"pfid":"int64"})
    df = df.dropna(subset=["pfid"])

    # rename columns for consistency
    df = df.rename(columns={"pfid": STREAMER_ID_COL})


    # write parquet
    output_path = outdir / f"{args.date}.parquet"
    print(f"Writing global streamer list to {output_path}")
    df.to_parquet(output_path, index=False)

    # update symlink
    latest = outdir / "latest.parquet"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(output_path.name)
    except OSError:
        # On Windows symlink may require admin; fallback to copy
        import shutil
        shutil.copy(output_path, latest)

    print("✓ Global parquet built:", output_path)

if __name__ == "__main__":
    main()