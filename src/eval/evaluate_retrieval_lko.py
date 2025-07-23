"""
evaluate_retrieval.py

Evaluate retrieval results against a leave-k-out test set.

Usage:
python -m src.eval.evaluate_retrieval \
    --test-path data/splits/test.parquet \
    --retrieval-dir data/retrieval_results \
    --ks 10 20 50 100 500

"""

from __future__ import annotations
import argparse, numpy as np, pandas as pd
import json
import pathlib
from tqdm import tqdm

from src.config import USER_ID_COL, STREAMER_ID_COL

def recall_at_k(r, k):
    return (r <= k).mean()

def ndcg_at_k(r, k):
    """
    Calculate NDCG@K for binary relevance.
    
    Args:
        r: array of ranks (1-based) for each relevant item
        k: cutoff position
    
    Returns:
        NDCG@K score
    """
    # Calculate DCG@K
    dcg = np.where(r <= k, 1 / np.log2(r + 1), 0).sum()
    
    # Calculate IDCG@K (ideal DCG)
    # For binary relevance, IDCG is the sum of 1/log2(i+1) for i=1 to min(k, num_relevant_items)
    num_relevant = len(r)
    ideal_positions = np.arange(1, min(k, num_relevant) + 1)
    idcg = (1 / np.log2(ideal_positions + 1)).sum()
    
    # Return NDCG@K
    return dcg / idcg if idcg > 0 else 0.0

def mrr_at_k(r, k):
    return np.where(r <= k, 1 / r, 0).mean()

def recall_at_k_multi(ranks_list, k):
    return np.mean([np.mean(np.array(ranks) <= k) for ranks in ranks_list])

def mrr_at_k_multi(ranks_list, k):
    return np.mean([
        np.sum(1 / np.array(r)[np.array(r) <= k]) / len(r)
        if any(np.array(r) <= k) else 0
        for r in ranks_list
    ])

def ndcg_at_k_multi(ranks_list, k):
    return np.mean([ndcg_at_k(np.array(r), k) for r in ranks_list])

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", required=True, help="Path to test parquet/csv with user_id, streamer_id, label columns")
    parser.add_argument("--retrieval-dir", required=True, help="Directory containing per-user top-k JSON results")
    parser.add_argument("--ks", nargs="+", type=int, default=[10, 20, 50, 100, 500])
    args = parser.parse_args()

    # Load test set
    if args.test_path.endswith(".parquet"):
        test_df = pd.read_parquet(args.test_path)
    else:
        test_df = pd.read_csv(args.test_path)

    retrieval_dir = pathlib.Path(args.retrieval_dir)
    ks = args.ks

    # positive_interactions = test_df[test_df["label"] == 1][[USER_ID_COL, STREAMER_ID_COL]]

    # group test positives per user
    test_gt_dict = (
        test_df[test_df["label"] == 1]
        .groupby(USER_ID_COL)[STREAMER_ID_COL]
        .apply(set)
        .to_dict()
    )

    all_ranks = []
    for user_id, pos_items in tqdm(test_gt_dict.items(), desc="Evaluating"):
        json_path = retrieval_dir / f"user_{user_id}.json"
        if not json_path.exists():
            print(f"Missing retrieval result for user {user_id}. Skipping.")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            recs = json.load(f)

        rec_sids = [rec["streamer_id"] for rec in recs]

        user_ranks = []
        for pos_sid in pos_items:
            try:
                rank = rec_sids.index(pos_sid) + 1  # Convert to 1-based rank
            except ValueError:
                rank = 1_000_000 # not found
            user_ranks.append(rank)

        all_ranks.append(user_ranks)

    for k in ks:
        recall = recall_at_k_multi(all_ranks, k)
        mrr = mrr_at_k_multi(all_ranks, k)
        ndcg = ndcg_at_k_multi(all_ranks, k)

        print(
            f"Recall@{k:>3}: {recall:.4f}, "
            f"MRR@{k:>3}: {mrr:.4f}, "
            f"nDCG@{k:>3}: {ndcg:.4f}"
        )

if __name__ == "__main__":
    main()