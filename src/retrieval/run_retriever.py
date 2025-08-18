
import argparse
import pathlib
import json
import pandas as pd

from src.config import USER_ID_COL, STREAMER_ID_COL
from src.representations.store import EmbeddingStoreConfig, EmbeddingStore
from src.representations.user_embedder import UserEmbedderConfig, UserEmbedder
from src.retrieval.retriever import Retriever

DEFAULT_K = 500  # Default number of candidates to retrieve

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", required=True, help="Path to test parquet/csv with user_id, streamer_id, label columns")
    parser.add_argument("--emb-dir", required=True, help="Directory containing streamer_embeddings.npy and lookup.parquet")
    parser.add_argument("--index", required=True, help="Path to the FAISS index file")
    parser.add_argument("--user-log", default="data/processed/interactions/latest.parquet", help="Path to the user interaction log (parquet or csv) within training set")
    parser.add_argument("--out-dir", type=str, default="results/retrieval", help="Output directory for retrieval results")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of top candidates to retrieve (default: 100)")
    args = parser.parse_args()

    # Setup
    output_dir = pathlib.Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load item store and user embedder to generate user vectors
    print("Loading item store...")
    item_store_cfg = EmbeddingStoreConfig(
        data_dir=args.emb_dir,
        id_column=STREAMER_ID_COL,
        normalize=False, # assumption: embeddings are already normalized
    )
    item_store = EmbeddingStore(item_store_cfg)

    # load user embedder to generate user vectors
    print("Initializing user embedder...")
    user_emb_cfg = UserEmbedderConfig(
        user_log_path=args.user_log,
        user_col=USER_ID_COL,
        item_col=STREAMER_ID_COL,
        max_history_len=50,
        pooling="mean",
        fallback_strategy="random",
        normalize=True, 
        rng_seed=42
    )
    user_embedder = UserEmbedder(user_emb_cfg, item_store)

    # load test data
    print(f"Loading user IDs from {args.test_path}...")
    test_df = pd.read_parquet(args.test_path) if args.test_path.endswith(".parquet") else pd.read_csv(args.test_path)
    user_ids = test_df[USER_ID_COL].unique()

    # construct retriever
    print("Initializing retriever...")
    retriever = Retriever(
        user_embedder=user_embedder,
        lookup_path=args.emb_dir + "/lookup.parquet",
        index_path=args.index, 
        k=args.k
    )

    print(f"› Running retrieval for {len(user_ids)} users...")

    retriever.retrieve_many(user_ids, output_dir)

if __name__ == "__main__":
    main()
