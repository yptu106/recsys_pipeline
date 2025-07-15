"""
retrieval.py

Lightweight recall module for user-to-streamer retrieval using mean-pooled streamer embeddings
and a FAISS cosine index.

Assumptions:
- Streamer embeddings (e.g., streamer_embs.npy) are ℓ₂-normalized.
- User-streamer interaction history is available in a Parquet or CSV with columns: user_id, streamer_id.
- The FAISS index and the lookup table are row-aligned with the NumPy matrix.

Usage:
python -m src.services.retrieval \
    --user-id 12345 \
    --emb-dir embeddings/paraphrase-multilingual-MiniLM-L12-v2 \
    --index index/faiss/item_hnsw.idx \
    --k 100
"""

from __future__ import annotations
import argparse
import pathlib
import json
from functools import lru_cache
from typing import List

import faiss
import numpy as np
import pandas as pd

from src.config import USER_ID_COL, STREAMER_ID_COL

np.random.seed(42)  # For reproducibility

DEFAULT_K = 100  # Default number of candidates to retrieve

def get_emb_paths(emb_dir: str):
    emb_dir = pathlib.Path(emb_dir)
    emb_path = emb_dir / "streamer_embeddings.npy"
    lookup_path = emb_dir / "lookup.parquet"
    return emb_path, lookup_path

@lru_cache(maxsize=1)
def _load_streamer_embs(emb_path: str) -> np.memmap:
    """Load and memory-map the item (streamer) embedding matrix from disk.
    Cached after first load for efficiency.
    """
    return np.load(emb_path, mmap_mode="r")

@lru_cache(maxsize=1)
def _load_lookup(lookup_path: str) -> pd.DataFrame:
    """Load the streamer lookup table (row-aligned with embeddings).
    Cached after first load for efficiency.
    """
    df = pd.read_parquet(lookup_path)
    df.reset_index(drop=True, inplace=True)
    return df

@lru_cache(maxsize=1)
def _rowid_by_streamer_id(lookup_path: str) -> dict[int, int]:
    """Build a mapping from streamer_id to row index in the embedding matrix.
    Cached after first build for efficiency.
    """
    df = _load_lookup(lookup_path)
    return {int(sid): i for i, sid in enumerate(df[STREAMER_ID_COL].values)}

@lru_cache(maxsize=1)
def _load_faiss(index_path: str) -> faiss.Index:
    """Load the FAISS index from disk.
    Cached after first load for efficiency.
    """
    return faiss.read_index(str(index_path))

@lru_cache(maxsize=1)
def _load_user_logs(user_log_path: str) -> pd.DataFrame:
    """Load user interaction logs (user_id, streamer_id) from disk.
    Cached after first load for efficiency.
    """
    if user_log_path.endswith(".parquet"):
        return pd.read_parquet(user_log_path, columns=[USER_ID_COL, STREAMER_ID_COL])
    return pd.read_csv(user_log_path, usecols=[USER_ID_COL, STREAMER_ID_COL])

def user_embedding(
    user_id: int, 
    emb_path: str, 
    lookup_path: str,
    user_log_path: str, 
    n_fallback: int = 20, 
    max_history: int = 50
) -> np.ndarray:
    """
    Compute the mean-pooled embedding for a user based on their interaction history.
    if the user has multiple interactions with the same streamer, the streamer's embedding will be included multiple times.
        => simple frequency weighting.
    Fallback:
        - If user has no history or no valid streamer embeddings, sample n_fallback random streamers.
    """
    def sample_random_streamer_ids(n: int) -> list:
        all_ids = _load_lookup(lookup_path)[STREAMER_ID_COL].values
        return np.random.choice(all_ids, size=n, replace=False).tolist()

    # Load user interaction logs
    logs = _load_user_logs(user_log_path)
    streamer_ids: list[int] = logs.loc[logs[USER_ID_COL] == user_id, STREAMER_ID_COL].tolist()

    # if the user has no history, sample a random set of streamer ids
    if not streamer_ids:
        print(f"User {user_id} has no interaction history. Sampling random streamers.")
        streamer_ids = sample_random_streamer_ids(n_fallback)

    # limit to max_history interactions
    if max_history is not None and len(streamer_ids) > max_history:
        # assuming that the history is ordered by recency, we take the most recent interactions
        streamer_ids = streamer_ids[-max_history:]
    
    row_map = _rowid_by_streamer_id(lookup_path)  # {streamer_id: row_id in item_matrix}
    streamer_mat = _load_streamer_embs(emb_path) # shape [N, dim], float32    

    # collect embeddings for the streamers the user has interacted with
    # only include those that have valid embeddings
    vecs = [streamer_mat[row_map[sid]] for sid in streamer_ids if sid in row_map]

    if not vecs:
        print(f"User {user_id} has no valid streamer interactions. Sampling random streamers.")
        streamer_ids = sample_random_streamer_ids(n_fallback)
        vecs = [streamer_mat[row_map[sid]] for sid in streamer_ids if sid in row_map]
    
    # mean pool the embeddings (TODO: weighted by interaction recency)
    v = np.mean(vecs, axis=0, dtype=np.float32)
    v /= np.linalg.norm(v) + 1e-9

    return v.astype("float32")

def retrieve(
    user_id: int, 
    emb_path: str, 
    lookup_path: str, 
    index_path: str, 
    user_log_path: str, 
    k: int = DEFAULT_K
) -> List[dict[str, object]]:
    user_vec = user_embedding(user_id, emb_path, lookup_path, user_log_path)
    lookup = _load_lookup(lookup_path)

    index = _load_faiss(index_path)
    # D: distances (cosine similarity), I: indices of the nearest neighbors
    D, I = index.search(user_vec.reshape(1, -1), k)
    out = lookup.iloc[I[0]].copy()
    out["score"] = D[0]
    out = out.sort_values("score", ascending=False)

    return out.to_dict("records")

def main() -> None:
    parser = argparse.ArgumentParser(description="FAISS retrieval for streamer embeddings")
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--emb-dir", required=True, help="Directory containing streamer_embeddings.npy and lookup.parquet")
    parser.add_argument("--index", required=True, help="Path to FAISS index")
    parser.add_argument("--user-log", default="data/processed/interactions/latest.parquet", help="User interaction log (parquet or csv)")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--out-dir", default="data/retrieval_results", help="Output directory for retrieval results")

    args = parser.parse_args()

    emb_path, lookup_path = get_emb_paths(args.emb_dir)
    recs = retrieve(
        user_id=args.user_id,
        emb_path=str(emb_path),
        lookup_path=str(lookup_path),
        index_path=args.index,
        user_log_path=args.user_log,
        k=args.k
    )

    print(json.dumps(recs, ensure_ascii=False, indent=2))

    # Save results to output directory
    output_dir = pathlib.Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"user_{args.user_id}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()