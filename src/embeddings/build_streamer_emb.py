"""
build_streamer_embs.py

Generate dense streamer embeddings from the item_sentence column using a pre-trained sentence embedding model.

Usage:
python -m embeddings.build_streamer_embs \
    --features features/streamer/latest.parquet \
    --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
    --out_vec embeddings/streamer_vectors.npy \
    --out_map embeddings/lookup.parquet
"""

from __future__ import annotations
import os
import argparse
import pathlib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

from src.config import USER_ID_COL, STREAMER_ID_COL

COL_TO_ENCODE = "item_sentence"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Path to the streamer features parquet")
    parser.add_argument("--encode-col", default="item_sentence", choices=["item_sentence", "format_sentence"], help="Which column to encode for streamer embeddings (default: item_sentence)")
    parser.add_argument("--out_emb",  default="embeddings/streamer_embeddings.npy", help="Output .npy file for streamer embeddings")
    parser.add_argument("--out_map",  default="embeddings/lookup.parquet", help="Output .parquet file for streamer id lookup")
    parser.add_argument(
        "--model", 
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Pre-trained sentence embedding model name or path (e.g., 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
    )
    parser.add_argument("--normalize", default=True, help="Whether to L2-normalize the embeddings (default: True)", type=bool, nargs='?', const=True)
    args = parser.parse_args()

    print(f"› Loading features from {args.features}")
    df = pd.read_parquet(args.features, columns=[STREAMER_ID_COL, args.encode_col])

    print(f"› Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    print(f"› Encoding {args.encode_col} ...")
    sentences = df[args.encode_col].tolist()
    embeddings = model.encode(sentences, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if args.normalize:
        print("   Normalizing embeddings (L2 norm)...")
        embeddings = normalize(embeddings, axis=1, norm='l2')
    print("   Embedding shape:", embeddings.shape)

    # Create output directory if needed
    pathlib.Path(args.out_emb).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out_map).parent.mkdir(parents=True, exist_ok=True)

    # save embeddgings and lookup table
    np.save(args.out_emb, embeddings)
    print(f"✓ Wrote streamer embeddings → {args.out_emb}")

    df[[STREAMER_ID_COL]].reset_index(drop=True).to_parquet(args.out_map)
    print(f"✓ Wrote streamer ID lookup → {args.out_map}")

if __name__ == "__main__":
    main()