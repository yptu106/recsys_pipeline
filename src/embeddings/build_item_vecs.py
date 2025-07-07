"""
build_item_vecs.py
Stack one or more dense feature columns into a NumPy matrix for FAISS.

Usage
-----
python -m embeddings.build_item_vecs \
       --features  features/streamer/latest.parquet \
       --out_vec   embeddings/item_vectors.npy \
       --out_map   embeddings/lookup.parquet \
       --add-cols  tag2vec_pool tag_count first_tag
"""

from __future__ import annotations
import argparse
import pathlib
import numpy as np
import pandas as pd

DEFAULT_COLS = ["tag2vec_pool"]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Path to the streamer features parquet")
    parser.add_argument("--out_vec",  default="embeddings/item_vectors.npy")
    parser.add_argument("--out_map",  default="embeddings/lookup.parquet")
    parser.add_argument("--add-cols", nargs="*", default=[],
                    help="Extra dense columns to append (order kept)")
    args = parser.parse_args()

    cols = DEFAULT_COLS + args.add_cols
    print(f"› loading {args.features} with columns {cols}")
    df = pd.read_parquet(args.features, columns=["pfid"] + cols)

    # reset `pfid` to `streamer_id` for consistency
    df.rename(columns={"pfid": "streamer_id"}, inplace=True)

    # sanity-check column shapes
    shapes = {c: len(df[c].iloc[0]) for c in cols}
    print("   column dims:", shapes)

    # stack selected columns
    parts = [np.vstack(df[c].values).astype("float32") for c in cols]
    matrix = np.hstack(parts)
    print("   final matrix shape:", matrix.shape)

    # create output dirs
    pathlib.Path(args.out_vec).parent.mkdir(parents=True, exist_ok=True)

    np.save(args.out_vec, matrix)
    df[["streamer_id"]].reset_index(drop=True)\
      .to_parquet(args.out_map)

    print(f"✓ wrote vectors → {args.out_vec}")
    print(f"✓ wrote lookup  → {args.out_map}")

if __name__ == "__main__":
    main()