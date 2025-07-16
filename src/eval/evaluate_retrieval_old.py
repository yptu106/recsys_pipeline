from __future__ import annotations
import argparse, numpy as np, pandas as pd
from tqdm import tqdm

from src.services.retrieval import retrieve, get_emb_paths

def recall_at_k(r, k):
    return (r <= k).mean()

def ndcg_at_k(r, k):
    return np.where(r <= k, 1 / np.log2(r + 1), 0).mean()

def mrr_at_k(r, k):
    return np.where(r <= k, 1 / r, 0).mean()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", required=True, help="Path to test parquet/csv with user_id, streamer_id, label columns")
    parser.add_argument("--emb-dir", required=True, help="Path to the directory containing embeddings and lookup table")
    parser.add_argument("--index", required=True, help="Path to the FAISS index file")
    parser.add_argument("--user-log", default="data/splits/interactions_train.parquet",
                        help="Path to the user interaction log (parquet or csv) within training set")
    parser.add_argument("--ks", nargs="+", type=int, default=[10, 20, 50, 100, 500])
    args = parser.parse_args()

    # Load test set
    if args.test_path.endswith(".parquet"):
        test_df = pd.read_parquet(args.test_path)
    else:
        test_df = pd.read_csv(args.test_path)

    emb_path, lookup_path = get_emb_paths(args.emb_dir)
    index_path = args.index
    user_log_path = args.user_log
    ks = args.ks

    positive_interactions = test_df[test_df["label"] == 1][["user_id", "streamer_id"]]

    # find the ranks of the positive streamer_id for each user's recommendations
    ranks = []
    for user_id, positive_sid in tqdm(positive_interactions.itertuples(index=False)):
        recs = retrieve(
            user_id=user_id,
            emb_path=str(emb_path),
            lookup_path=str(lookup_path),
            index_path=index_path,
            user_log_path=user_log_path,
            k=max(ks)
        )
        
        if not recs:
            print(f"No recommendations found for user {user_id}. Skipping.")
            ranks.append(1_000_000)  # Assign a large rank for no recommendations
            continue

        # Find the rank of the positive streamer_id
        rec_sids = [rec["streamer_id"] for rec in recs]
        try:
            rank = rec_sids.index(positive_sid) + 1  # 1-based index
        except ValueError:
            rank = 1_000_000  # not found in top-k

        ranks.append(rank)
    
    ranks = np.array(ranks, dtype=np.int32)

    for k in ks:
        recall = recall_at_k(ranks, k)
        mrr = mrr_at_k(ranks, k)
        ndcg = ndcg_at_k(ranks, k)

        print(
            f"Recall@{k:>3}: {recall:.4f}, "
            f"MRR@{k:>3}: {mrr:.4f}, "
            f"nDCG@{k:>3}: {ndcg:.4f}"
        )

if __name__ == "__main__":
    main()