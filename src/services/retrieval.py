"""retrieval.py
Lightweight recall module used by `candidate_service.py` (or any offline script)
that represents a **user** as the mean‑pooled embedding of every streamer they
have interacted with, then queries a FAISS cosine index for the top‑*k*
candidates.

Assumptions
-----------
* Item vectors (`item_vectors.npy`) are ℓ₂‑normalised. If they are not,
  set `--renorm-items` when building the index.
* User‑streamer interaction history is available in a Parquet or CSV with
  columns: `user_id, streamer_id`  (additional columns ignored).
* The FAISS index and the lookup table are row‑aligned with the NumPy matrix.

The module is pure‑Python and holds no global async state – safe for multi‑threaded FastAPI usage.
"""
from __future__ import annotations

import pathlib
from functools import lru_cache
from typing import List, Tuple

import argparse, json, random

import faiss
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config – edit these paths if your repo layout differs.
# ---------------------------------------------------------------------------
ITEM_VEC_PATH  = pathlib.Path("embeddings/item_vectors.npy")
LOOKUP_PATH    = pathlib.Path("embeddings/lookup.parquet")
INDEX_PATH     = pathlib.Path("index/faiss/item_hnsw.idx") # TODO: make configurable
USER_LOG_PATH  = pathlib.Path("data/processed/interactions/latest.parquet")  # fallback CSV OK

_DEFAULT_K = 100 # Default number of candidates to retrieve

# ---------------------------------------------------------------------------
# Lazy loaders – vectors and index are memory‑mapped; lookup & logs load once.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_item_vectors() -> np.memmap:  # shape [N, dim], float32
    return np.load(ITEM_VEC_PATH, mmap_mode="r")

@lru_cache(maxsize=1)
def _load_lookup() -> pd.DataFrame:
    df = pd.read_parquet(LOOKUP_PATH)
    df.reset_index(drop=True, inplace=True)         # row‑index == FAISS id
    return df

@lru_cache(maxsize=1)
def _rowid_by_streamer_id() -> dict[int, int]:
    df = _load_lookup()
    return {int(streamer_id): i for i, streamer_id in enumerate(df.streamer_id.values)}

@lru_cache(maxsize=1)
def _load_faiss() -> faiss.Index:
    return faiss.read_index(str(INDEX_PATH))

@lru_cache(maxsize=1)
def _load_user_logs() -> pd.DataFrame:
    if USER_LOG_PATH.suffix == ".parquet":
        return pd.read_parquet(USER_LOG_PATH, columns=["user_id", "streamer_id"])
    return pd.read_csv(USER_LOG_PATH, usecols=["user_id", "streamer_id"])


def user_embedding(user_id: int) -> np.ndarray | None:
    """Return ℓ₂‑normalised mean vector of streamers the user has touched.

    If the user has **no history** in logs, returns *None* (caller must decide
    fallback behaviour).
    """
    logs = _load_user_logs()
    streamer_ids: list[int] = logs.loc[logs.user_id == user_id, "streamer_id"].tolist()
    if not streamer_ids:
        return None

    row_map = _rowid_by_streamer_id() # {streamer_id: row_id in item_matrix}
    vecs = []
    item_mat = _load_item_vectors()
    for streamer_id in streamer_ids:
        if streamer_id in row_map:
            vecs.append(item_mat[row_map[streamer_id]])

    if not vecs:
        return None

    v = np.mean(vecs, axis=0, dtype=np.float32)
    # Normalise (safety; should already be close to unit) -------------------
    v /= np.linalg.norm(v) + 1e-9
    return v.astype("float32")


def retrieve(user_id: int, k: int = _DEFAULT_K) -> List[dict[str, object]]:
    """Return k candidate streamers as a list of dicts.

    Each dict contains `streamer_ids`, and the FAISS `score` (cosine sim).
    If the user has no embedding, we default to the top‑k popular rows (popularity sort not yet implemented),
    (id 0‥k‑1). Replace this logic with a smarter cold‑start strategy later.
    """
    v_user = user_embedding(user_id)
    lookup = _load_lookup()

    if v_user is None:
        print(f"User {user_id} has no interaction history; returning cold‑start candidates.")
        # Cold‑start fallback (popularity sort not yet implemented).
        cold = lookup.head(k).copy()
        cold["score"] = np.nan
        return cold.to_dict("records")

    index = _load_faiss()
    D, I = index.search(v_user.reshape(1, -1), k)     # (1, k)

    out = lookup.iloc[I[0]].copy()
    out["score"] = D[0]
    return out.to_dict("records")


def main() -> None:
    parser = argparse.ArgumentParser(description="FAISS retrieval quick‑test")
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--k", type=int, default=_DEFAULT_K)
    args = parser.parse_args()

    recs = retrieve(args.user_id, k=args.k)
    print(json.dumps(recs, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
