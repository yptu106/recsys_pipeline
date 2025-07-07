"""
evaluate_retrieval.py
Offline metrics for the recall stage only (no ranker).

Metrics:
    • Recall@k         (hit rate)             – fraction of users whose lone
                                                positive appears in top-k
    • nDCG@k           (discount)             – takes rank position into account
    • MRR@k            – mean reciprocal rank – sensitive to top spots
"""

from __future__ import annotations
import argparse, numpy as np, pandas as pd
from tqdm import tqdm
from ..services.retrieval import _rowid_by_streamer_id, _load_item_vectors, _load_faiss, _load_lookup

def make_user_logs(train_df: pd.DataFrame) -> dict[int, list[int]]:
    """build a map of user_id -> list of streamer_ids based on train_df"""
    # collect every streamer_id the user has interacted with
    grp = train_df.groupby("user_id").streamer_id.apply(list)

    return {u: list(streamer_ids) for u, streamer_ids in grp.items()}

def user_embedding(logs: dict[int, list[int]], user_id: int) -> np.ndarray | None:
    """build a user embedding from their interaction history"""
    if user_id not in logs:
        return None  # cold user

    streamer_ids: list[int] = logs[user_id]
    if not streamer_ids:
        return None
    
    row_map = _rowid_by_streamer_id()  # {streamer_id: row_id in item_matrix}
    vecs = []
    item_mat = _load_item_vectors()  # shape [N, dim], float32
    for streamer_id in streamer_ids:
        if streamer_id in row_map:
            vecs.append(item_mat[row_map[streamer_id]])

    if not vecs:
        return None

    v = np.mean(vecs, axis=0, dtype=np.float32)
    # Normalise (safety; should already be close to unit) -------------------
    v /= np.linalg.norm(v) + 1e-9
    return v.astype("float32")

def recall_at_k(ranks, k):
    return np.mean(ranks <= k)

def mrr_at_k(ranks, k):
    rr = 1.0 / ranks
    rr[ranks > k] = 0
    return rr.mean()

def ndcg_at_k(ranks, k):
    dcg = 1 / np.log2(ranks + 1)
    dcg[ranks > k] = 0
    idcg = 1  # single positive per user
    return dcg.mean() / idcg


def evaluate(train_path: str, split_path: str, ks=(10, 20, 50)) -> None:
    train_df = pd.read_parquet(train_path)
    logs = make_user_logs(train_df)

    split_df = pd.read_parquet(split_path) # positive + negative rows, label columns
    positive = split_df[split_df.label == 1][["user_id", "streamer_id"]]

    ranks = np.empty(len(positive), dtype=np.int32)

    index = _load_faiss()
    lookup = _load_lookup()
    for i, (uid, positvie_sid) in enumerate(tqdm(positive.itertuples(index=False))):
        user_vec = user_embedding(logs, uid)
        if user_vec is None:
            ranks[i] = 1_000_000  # cold user, rank is arbitrarily high
            continue
        
        D, I = index.search(user_vec.reshape(1, -1), max(ks))  # shape [1, k]

        # I is the row ids in the index, convert to streamer_ids
        rec_sids = lookup.iloc[I[0]].copy()
        where = np.where(rec_sids == positvie_sid)[0]
        ranks[i] = where[0] + 1 if where.size else 1_000_000

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/splits/train.parquet", help="train split parquet")
    parser.add_argument("--split", default="data/splits/val.parquet",
                    help="val or test split parquet")
    parser.add_argument("--k", nargs="+", type=int, default=[10, 20, 50])
    args = parser.parse_args()
    evaluate(args.train, args.split, tuple(args.k))
