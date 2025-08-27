"""
build_streamer_embs.py

Generate embeddings for streamers based on selected column.

Usage:
python -m src.encoder.build_streamer_emb \
    --features <path_to_streamer_features> \
    --encode-col <column_to_encode> \
    --out-dir <output_directory> \
    --model <model name> \
    --model-path <path_to_fine-tuned_checkpoint> \
    [--normalize True|False]
"""

from __future__ import annotations
import os
import argparse
import pathlib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from typing import List
from tqdm import tqdm

from src.config import USER_ID_COL, STREAMER_ID_COL

def encode_with_ollama(sentences: List[str], model_name: str, host: str = "http://192.168.0.33:11434") -> np.ndarray:
    from ollama import Client
    client = Client(host=host)

    print(f"› Encoding with Ollama model: {model_name} @ {host} (1-by-1 encoding)")

    embeddings = []
    for sentence in tqdm(sentences, desc="Encoding streamers"):
        resp = client.embed(model=model_name, input=sentence)
        emb = resp["embeddings"][0] 
        embeddings.append(emb)
    embeddings = np.array(embeddings)

    print(f"› Encoded {len(sentences)} sentences with shape: {embeddings.shape}")
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Path to the streamer features parquet")
    parser.add_argument("--include-numerical-cols", action="store_true", help="Whether to include numerical columns in the embedding")
    parser.add_argument("--encode-col", default="item_sentence", choices=["item_sentence", "format_sentence"], help="Which column to encode for streamer embeddings (default: item_sentence)")
    parser.add_argument("--out-dir", default="embeddings", help="Output directory for embeddings and lookup table")
    parser.add_argument(
        "--model", 
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Pre-trained sentence embedding model name or path (e.g., 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
    )
    parser.add_argument("--model-path", default=None, help="Path to the custom model directory (if using a fine-tuned model)")
    parser.add_argument("--normalize", action="store_true", help="Whether to L2-normalize the embeddings (default: True)")
    args = parser.parse_args()

    print(f"› Loading features from {args.features}")
    df = pd.read_parquet(args.features)


    print(f"› Encoding {args.encode_col} ...")
    sentences = df[args.encode_col].tolist()

    # determine which encoder to use
    if args.model.startswith("bge-m3") or "ollama" in args.model:
        print(f"› Using Ollama model: {args.model}")
        embeddings = encode_with_ollama(df[args.encode_col].tolist(), model_name=args.model)
    else:
        from sentence_transformers import SentenceTransformer
        print(f"› Encoding with SentenceTransformer model: {args.model}")

        if args.model_path:
            print(f"   Loading custom model from {args.model_path}")
            model = SentenceTransformer(args.model_path)
        else:
            model = SentenceTransformer(args.model)
        embeddings = model.encode(sentences, show_progress_bar=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)
    
    if args.normalize:
        print("   Normalizing embeddings (L2 norm)...")
        embeddings = normalize(embeddings, axis=1, norm='l2')
        
    print("   Embedding shape:", embeddings.shape)

    # concatentate with numerical columns if requested
    if args.include_numerical_cols:
        print("Deprecated: --include-numerical-cols is deprecated and will be removed in future versions.")
        # print("› Including numerical columns in embeddings ...")
        # numerical_cols = [
        #     "i_watch_tot", "i_watch_cnt", "i_unique_user", 
        #     "i_live_cnt", "i_followers", "i_gift_amt", 
        #     "i_watch_avg", "i_pop_z"
        # ]
        # numerical_features = df[numerical_cols].to_numpy().astype(np.float32)
        # print("   Numerical features shape:", numerical_features.shape)

        # # concatenate embeddings with numerical features if requested
        # embeddings = np.concatenate([embeddings, numerical_features], axis=1)
        # print("   Combined embeddings shape:", embeddings.shape)

    # ensure output directory exists
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # define output paths
    out_emb = out_dir / "embeddings.npy"
    out_map = out_dir / "lookup.parquet"

    # Create output directory if needed
    pathlib.Path(out_emb).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out_map).parent.mkdir(parents=True, exist_ok=True)

    # save embeddgings and lookup table
    np.save(out_emb, embeddings)
    print(f"✓ Wrote streamer embeddings → {out_emb}")

    df[[STREAMER_ID_COL]].reset_index(drop=True).to_parquet(out_map)
    print(f"✓ Wrote streamer ID lookup → {out_map}")

if __name__ == "__main__":
    main()