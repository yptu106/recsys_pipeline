"""
run_train.py

This script runs the RecBole training process and manages the latest checkpoint symlink.
It expects a configuration file and a directory where checkpoints are stored.
After training, it updates a symlink to point to the most recent checkpoint.

Usage:
    python src.ranker.recbole.run_train.py \
        --config <path_to_config_file> \
        --ckpt-dir <path_to_checkpoint_directory> \
        [--link-name latest.pth]

This script is designed to be run in the context of a RecBole project.
It requires the RecBole library to be installed and properly configured.
"""

import os
import re
import shutil
import pathlib
from datetime import datetime
import argparse
from recbole.quick_start import run_recbole

TIME_RE = re.compile(r".*-(\w{3})-(\d{2})-(\d{4})_(\d{2})-(\d{2})-(\d{2})\.pth$")

def _parse_time_from_name(path: pathlib.Path) -> float:
    """
    Try parsing names like: BPR-Aug-14-2025_11-48-50.pth
    Falls back to file mtime if pattern doesn't match.
    """
    match = TIME_RE.match(path.name)
    if match:
        month, day, year, hour, minute, second = match.groups()
        dt = datetime.strptime(f"{year}-{month}-{day} {hour}:{minute}:{second}", "%Y-%b-%d %H:%M:%S")
        return dt.timestamp()
    else:
        return path.stat().st_mtime

def point_latest_symlink(ckpt_dir: pathlib.Path, link_name: str = "latest.pth"):
    """
    Point the symlink `link_name` in `ckpt_dir` to the most recent checkpoint.
    If the symlink already exists, it will be updated.
    """
    ckpt_dir = ckpt_dir.resolve()
    pths = [p for p in ckpt_dir.glob("*.pth") if p.is_file()]
    if not pths:
        raise FileNotFoundError(f"No .pth checkpoints under {ckpt_dir}")

    # pick newest by parsed timestamp (fallback to mtime)
    newest = max(pths, key=_parse_time_from_name)

    # update symlink
    link_path = ckpt_dir / link_name
    try:
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(newest.name)
    except OSError:
        # On Windows symlink may require admin; fallback to copy
        shutil.copy(newest, link_path)
    
    print(f"✓ Updated {link_name} to point to {newest.name}")

    return newest

def parse_args():
    parser = argparse.ArgumentParser(description="Run RecBole training")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument(
        "--ckpt-dir", 
        required=True,
        type=str,
        help="Directory where RecBole writes .pth files."
    )
    parser.add_argument("--link-name", default="latest.pth")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Run RecBole with the provided configuration
    run_recbole(config_file_list=[args.config])

    # Point latest.pth to the most recent checkpoint
    ckpt_dir = pathlib.Path(args.ckpt_dir).resolve()
    if not ckpt_dir.is_dir():
        raise NotADirectoryError(f"Checkpoint directory {ckpt_dir} does not exist or is not a directory.")
    newest = point_latest_symlink(ckpt_dir, args.link_name)
    
    print(f"✓ Latest checkpoint is now: {newest.name}")
    print(f"✓ Symlink {args.link_name} updated in {ckpt_dir}")

if __name__ == "__main__":
    main()