import argparse
import pathlib
import json
import pandas as pd
import numpy as np
import torch
from typing import Union
from collections import namedtuple

UserEmbFallbackConfig = namedtuple(
    "UserEmbFallbackConfig",
    ["emb_path", "lookup_path", "user_log_path", "n_fallback", "max_history"]
)

from ranker.models.transformer_ranker import TransformerRanker
from src.services.retrieval import user_embedding, get_emb_paths
from src.config import USER_ID_COL, STREAMER_ID_COL

QUIET_MODE = False
DEFAULT_K = 100

def log(message: str, quiet: bool):
    if not quiet:
        print(message)

def load_embeddings_and_lookup(emb_dir, id_col):
    emb_path, lookup_path = get_emb_paths(emb_dir)
    embeddings = np.load(emb_path)
    lookup_df = pd.read_parquet(lookup_path).reset_index(drop=True)
    lookup = dict(zip(lookup_df[id_col], lookup_df.index))
    return embeddings, lookup

def get_user_embedding(
        user_id: int,
        user_embeddings: np.ndarray,
        user_lookup: dict[int, int],
        fallback_config: UserEmbFallbackConfig = None
    ) -> np.ndarray:
    if user_id in user_lookup:
        return user_embeddings[user_lookup[user_id]]
    elif fallback_config:
        return user_embedding(
            user_id,
            emb_path=fallback_config.emb_path,
            lookup_path=fallback_config.lookup_path,
            user_log_path=fallback_config.user_log_path,
            n_fallback=fallback_config.n_fallback,
            max_history=fallback_config.max_history
        )
    else:
        raise ValueError(f"User {user_id} not found in user embeddings and no fallback config provided.")

def rank_user(
    user_id: int,
    retrieval_path: str,
    user_embeddings: np.ndarray, user_lookup: dict[int, int],
    item_embeddings: np.ndarray, item_lookup: dict[int, int],
    model_path: str,
    fallback_config: UserEmbFallbackConfig = None,
    topk: int = DEFAULT_K,
):
    log(f"Ranking for user {user_id}", True)

    log(f"› Loading retrieved candidates from {retrieval_path}", True)
    with open(retrieval_path, "r", encoding="utf-8") as f:
        recs = json.load(f)

    candidates_df = pd.DataFrame(recs)

    user_vec = get_user_embedding(
        user_id=user_id,
        user_embeddings=user_embeddings,
        user_lookup=user_lookup,
        fallback_config=fallback_config
    )
    user_vec = torch.tensor(user_vec, dtype=torch.float32).unsqueeze(0)  # shape: [1, emb_dim]

    # load the transformer ranker
    emb_dim = user_vec.shape[1]
    model = TransformerRanker(input_dim=emb_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # score all candidates
    scores = []
    for streamer_id in candidates_df[STREAMER_ID_COL]:
        if streamer_id not in item_lookup:
            log(f"Streamer {streamer_id} not found in item embeddings, skipping.", QUIET_MODE)
            scores.append((streamer_id, float('-inf')))
            continue
        streamer_vec = item_embeddings[item_lookup[streamer_id]]
        streamer_vec = torch.tensor(streamer_vec, dtype=torch.float32).unsqueeze(0)  # shape: [1, emb_dim]
        with torch.no_grad():
            score = model(user_vec, streamer_vec).item()
        scores.append(score)
    
    candidates_df["score"] = scores
    topk_df = candidates_df.sort_values("score", ascending=False).head(topk)

    return topk_df


