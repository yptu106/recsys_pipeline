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

from ranker.mlp.mlp_ranker import MLPRanker
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

    # load the MLP model
    emb_dim = user_vec.shape[1]
    model = MLPRanker(input_dim=3 * emb_dim)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--retrieval-path", type=str, required=True, help="Path to JSON with top-k recall results")
    parser.add_argument("--user-emb-dir", type=str, default=None, help="Directory containing user embeddings and lookup")
    parser.add_argument("--user-log", type=str, required=True, help="User interaction log (parquet or csv)")
    parser.add_argument("--streamer-emb-dir", required=True, help="Directory containing streamer embeddings and lookup")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained MLP model")
    parser.add_argument("--topk", type=int, default=DEFAULT_K)
    parser.add_argument("--out-dir", type=str, default="data/ranked_results")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--print-topk", action="store_true", help="Print top-k results to console")
    args = parser.parse_args()

    global QUIET_MODE
    QUIET_MODE = args.quiet

    # log(f"› Loading retrieved candidates from {args.retrieval_path}", args.quiet)
    # with open(args.retrieval_path, "r", encoding="utf-8") as f:
    #     recs = json.load(f)

    # candidates_df = pd.DataFrame(recs)
    # assert STREAMER_ID_COL in candidates_df.columns

    # load user embeddings and lookup if provided
    if args.user_emb_dir:
        log(f"› Loading user embeddings from {args.user_emb_dir}", QUIET_MODE)
        user_embeddings, user_lookup = load_embeddings_and_lookup(args.user_emb_dir, USER_ID_COL)
    
    # load streamer embeddings and lookup
    log(f"› Loading streamer embeddings from {args.streamer_emb_dir}", QUIET_MODE)
    item_embeddings, item_lookup = load_embeddings_and_lookup(args.streamer_emb_dir, STREAMER_ID_COL)

    # set up user embedding fallback configuration
    item_emb_path, item_lookup_path = get_emb_paths(args.streamer_emb_dir)
    fallback_config = UserEmbFallbackConfig(
        emb_path=item_emb_path,
        lookup_path=item_lookup_path, 
        user_log_path=args.user_log,
        n_fallback=20,
        max_history=50
    )

    log(f"› Ranking candidates for user {args.user_id}", QUIET_MODE)

    topk_df = rank_user(
        user_id=args.user_id,
        retrieval_path=args.retrieval_path,
        user_embeddings=user_embeddings,
        user_lookup=user_lookup,
        item_embeddings=item_embeddings,
        item_lookup=item_lookup,
        model_path=args.model_path,
        fallback_config=fallback_config,
        topk=args.topk
    )

    if args.print_topk:
        print(f"🎯 Top-{args.topk} ranked streamers for user {args.user_id}:")
        print(topk_df[[STREAMER_ID_COL, "score"]].to_string(index=False))
    
    # Save results to output directory
    output_dir = pathlib.Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"user_{args.user_id}.json"
    topk_df[[STREAMER_ID_COL, "score"]].to_json(output_file, orient="records", force_ascii=False, indent=2)
    log(f"✅ Saved ranked result to: {output_file}", QUIET_MODE)

if __name__ == "__main__":
    main()