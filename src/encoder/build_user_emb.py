"""
build_user_emb.py

Construct user embeddings by aggregating streamer embeddings based on user interactions.

Usage:
python -m src.encoder.build_user_emb \
    --streamer-emb-dir <path_to_streamer_embeddings> \
    --user-log <path_to_user_interaction_log> \
    --out-dir <output_directory> \
    [--normalize True|False]
"""

from __future__ import annotations
import argparse
import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import normalize

from src.config import USER_ID_COL, STREAMER_ID_COL
from src.services.retrieval import user_embedding, get_emb_paths

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--streamer-emb-dir", required=True, help="Directory containing streamer embeddings and lookup")
    parser.add_argument("--user-log", required=True, help="Path to user interaction log (parquet or csv)")
    parser.add_argument("--out-dir", default="embeddings/users", help="Output directory for user embeddings")
    parser.add_argument("--normalize", default=True, help="Whether to L2-normalize the embeddings (default: True)", type=bool, nargs='?', const=True)
    args = parser.parse_args()

    print(f"> Loading user interaction log from {args.user_log}")   
    user_log = pd.read_parquet(args.user_log) if args.user_log.endswith('.parquet') else pd.read_csv(args.user_log)
    user_ids = user_log[USER_ID_COL].unique()

    emb_path, lookup_path = get_emb_paths(args.streamer_emb_dir)

    print(f"> Building user embeddings for {len(user_ids)} users...")
    user_embeddings = []
    for user_id in tqdm(user_ids, desc="Building user embeddings"):
        user_vec = user_embedding(
            user_id=user_id,
            emb_path=str(emb_path),
            lookup_path=str(lookup_path),
            user_log_path=args.user_log
        )
        user_embeddings.append(user_vec)

    user_embeddings = np.array(user_embeddings)
    if args.normalize:
        print("   Normalizing user embeddings (L2 norm)...")
        user_embeddings = normalize(user_embeddings, axis=1, norm='l2')

    print("   Embedding shape:", user_embeddings.shape)

    # create output directory if it doesn't exist
    output_dir = pathlib.Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # save user embeddings and lookup table
    out_emb_path = output_dir / "embeddings.npy"
    out_lookup_path = output_dir / "lookup.parquet"

    print(f"> Saving user embeddings to {out_emb_path}")
    np.save(out_emb_path, user_embeddings)

    print(f"> Saving user lookup table to {out_lookup_path}")
    user_lookup = pd.DataFrame({
        USER_ID_COL: user_ids, 
    })

    user_lookup.to_parquet(out_lookup_path)

if __name__ == "__main__":
    main()