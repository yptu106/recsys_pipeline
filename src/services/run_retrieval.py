"""
run_retrieval.py

Run retrieval for all users in a test set.

Usage:
python -m src.eval.run_retrieval \
    --test-path data/splits/test.parquet \
    --emb-dir embeddings \
    --index index/faiss/index_flat.idx \
    --user-log data/splits/interactions_train.parquet \
    --out-dir data/retrieval_results \
    --k 100

"""

import argparse, pathlib, json
import pandas as pd
from tqdm import tqdm
from src.services.retrieval import retrieve, get_emb_paths

DEFAULT_K = 100  # Default number of candidates to retrieve

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

    emb_path, lookup_path = get_emb_paths(args.emb_dir)
    index_path = args.index
    k = args.k

    test_df = pd.read_parquet(args.test_path) if args.test_path.endswith(".parquet") else pd.read_csv(args.test_path)
    user_ids = test_df["user_id"].unique()

    print(f"› Running retrieval for {len(user_ids)} users...")

    for user_id in tqdm(user_ids, desc="Retrieving"):
        json_path = output_dir / f"user_{user_id}.json"

        recs = retrieve(
            user_id=user_id,
            emb_path=str(emb_path),
            lookup_path=str(lookup_path),
            index_path=index_path,
            user_log_path=args.user_log,
            k=args.k
        )

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
