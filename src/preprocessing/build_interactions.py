
import argparse
import datetime as dt
import pathlib
import pandas as pd

from src.config import USER_ID_COL, STREAMER_ID_COL

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Raw interactions CSV path")
    parser.add_argument("--outdir", default="data/processed/interactions", help="Output directory root")
    parser.add_argument("--date", default=dt.date.today().isoformat(), help="Date suffix for parquet filename")
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv, dtype={"pfid":"int64", "anchor_id":"int64"}) # pfid: user ID, anchor_id: streamer ID
    df = df.dropna(subset=["pfid", "anchor_id"])
    
    # rename columns for consistency
    df = df.rename(columns={"pfid": USER_ID_COL, "anchor_id": STREAMER_ID_COL})

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
