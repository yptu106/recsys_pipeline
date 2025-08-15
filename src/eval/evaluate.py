"""
evaluate.py

Evaluate retrieval/ranked/re-ranked results against a leave-one-out test set.

Usage:
python -m eval.evaluate \
    --test-path data/splits/test.parquet \
    --dir data/results/ \
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

def ndcg_at_k(ranks: np.ndarray, k: int) -> float:
    """
    - rank <= k: checks, for each user, whether their relevant item's rank is within the top-k.
    - 1 / np.log2(ranks + 1): computes DCG for each rank.
    - np.where(condition, value_if_true, value_if_false):
        - if ranks[i] <= k, use 1 / np.log2(ranks[i] + 1) (DCG for relevant item)
        - else, use 0 (not in top-k, so DCG is 0)
    - idcg is 1 in leave-one-out evaluation since we assume each user has exactly one relevant item.
    - Finally, we take the mean of these values to get the average nDCG across
    """

    ndcg_scores = np.where(ranks <= k, 1 / np.log2(ranks + 1), 0)
    return ndcg_scores.mean() # average nDCG across all users

def mrr_at_k(r, k):
    return np.where(r <= k, 1 / r, 0).mean()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", required=True, help="Path to test parquet/csv with user_id, streamer_id, label columns")
    parser.add_argument("--dir", required=True, help="Directory containing per-user top-k JSON results")
    parser.add_argument("--ks", nargs="+", type=int, default=[10, 20, 50, 100, 500])
    args = parser.parse_args()

    # Load test set
    if args.test_path.endswith(".parquet"):
        test_df = pd.read_parquet(args.test_path)
    else:
        test_df = pd.read_csv(args.test_path)

    json_dir = pathlib.Path(args.dir)
    ks = args.ks

    positive_interactions = test_df[test_df["label"] == 1][[USER_ID_COL, STREAMER_ID_COL]]

    ranks = []
    for user_id, positive_sid in tqdm(positive_interactions.itertuples(index=False)):
        json_path = json_dir / f"user_{user_id}.json"
        if not json_path.exists():
            print(f"Missing result for user {user_id}. Skipping.")
            ranks.append(1_000_000)
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            recs = json.load(f)

        rec_sids = [rec["streamer_id"] for rec in recs]
        try:
            rank = rec_sids.index(positive_sid) + 1  # 1-based index
        except ValueError:
            rank = 1_000_000  # not found

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