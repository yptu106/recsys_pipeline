"""build_features.py
Build the canonical streamer feature parquet from the raw CSV and a pre‑trained
Tag2Vec model.

The output goes to:
    features/streamer/<YYYY‑MM‑DD>.parquet
and a symlink `latest.parquet` is updated for downstream jobs.

Example
-------
python -m preprocessing.build_features \
        --csv      data/raw/streamers.csv \
        --tag2vec  embeddings/tag2vec.bin \
        --outdir   features/streamer
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import pathlib
from typing import List

import numpy as np
import pandas as pd
import pyroaring as pr  # roaring bitmap for compact multi‑hot
from gensim.models import KeyedVectors

def _normalize(tags: List[str]) -> List[str]:
    """Lower‑case & strip each tag; drop empties."""
    return [t.strip().lower() for t in tags if t and isinstance(t, str)]


def _tags_to_bitmap(tags: List[str], vocab: dict[str, int]) -> bytes:
    """Convert tag list to a serialized Roaring bitmap via the shared vocab."""
    idx = [vocab[t] for t in tags if t in vocab]
    bm = pr.BitMap(idx)
    return bm.serialize()


def _pool_tag_vecs(tags: List[str], kv: KeyedVectors) -> np.ndarray:
    """Mean‑pool tag vectors (ℓ₂‑normalised). Returns zeros if none found."""
    vecs = [kv[t] for t in tags if t in kv]
    if not vecs:
        return np.zeros(kv.vector_size, dtype=np.float32)
    v = np.mean(vecs, axis=0, dtype=np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Raw streamer CSV path")
    parser.add_argument("--tag2vec", required=True, help="Pre‑trained tag2vec .bin")
    parser.add_argument("--outdir", default="features/streamer", help="Output directory root")
    parser.add_argument("--date", default=dt.date.today().isoformat(), help="Date suffix for parquet filename")
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load raw CSV
    print("› Loading CSV …")
    df = pd.read_csv(args.csv, converters={"tags": ast.literal_eval})

    if "tags" not in df.columns or "pfid" not in df.columns:
        raise ValueError("CSV must contain at least 'pfid' and 'tags' columns")

    # Normalize tag lists (cleans and lowercases each tag in the list)
    df["tags"] = df.tags.apply(_normalize)

    # Load Tag2Vec & vocab
    print("› Loading Tag2Vec …")
    kv = KeyedVectors.load(args.tag2vec, mmap="r")
    vocab = kv.key_to_index  # dict[tag -> int]

    # Feature engineering
    # - Convert each streamer's tag list into a multi‑hot bitmap representation using Roaring bitmaps.
    # - Compute a pooled vector for each streamer's tags using the pre‑trained Tag2Vec embeddings.
    print("› Computing bitmaps and pooled vectors …")
    df["tags_multihot"] = df.tags.apply(lambda ts: _tags_to_bitmap(ts, vocab))
    df["tag2vec_pool"] = df.tags.apply(lambda ts: _pool_tag_vecs(ts, kv))

    # Optional helper columns
    df["tag_count"] = df.tags.str.len()
    df["first_tag"] = df.tags.str[0]

    # Write parquet
    out_path = outdir / f"{args.date}.parquet"
    print(f"› Writing {out_path} …")
    df.to_parquet(out_path, index=False)

    # Update symlink
    latest = outdir / "latest.parquet"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out_path.name)
    except OSError:
        # On Windows symlink may require admin; fallback to copy
        import shutil
        shutil.copy(out_path, latest)

    print("✓ Feature parquet built:", out_path)


if __name__ == "__main__":
    main()
