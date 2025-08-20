"""
run_ranker.py

This script runs the PyTorch-based ranker for a recommendation system.
It loads user IDs, item embeddings, and user embeddings, initializes the ranker,
and ranks items for each user based on their historical interactions.

Usage:
    python -m src.ranker.torch.services.run_ranker \
        --test-path <path_to_test_split_csv> \
        --retrieval-dir <path_to_retrieval_results> \
        --user-log-path <path_to_user_log> \
        --streamer-emb-dir <path_to_streamer_embeddings> \
        --out-dir <output_directory> \
        --topk <number_of_top_items> \
        --model-config <path_to_model_config_yaml> \
        --ckpt-path <path_to_model_checkpoint> \
        [--user-ids <list_of_user_ids>] \
        [--user-emb-dir <path_to_user_embeddings>] \
        [--debug-limit <number_of_users>] \
        [--batch-size <batch_size>]
"""

import yaml
import argparse
import numpy as np
import pandas as pd
import torch

from src.representations.store import EmbeddingStore, EmbeddingStoreConfig
from src.representations.user_embedder import UserEmbedder, UserEmbedderConfig
from src.ranker.torch.services.ranker import Ranker, RankerConfig
from src.ranker.utils.io import ensure_dir_exists, load_split
from src.config import USER_ID_COL, STREAMER_ID_COL

QUIET_MODE = False
DEFAULT_K = 100
MAX_HISTORY_LEN = 50  # Maximum length of user history to consider (only used for contextual models)

def parse_args():
    parser = argparse.ArgumentParser(description="Run Transformer Ranker")
    parser.add_argument("--test-path", required=True, type=str, help="Path to test split CSV file with user IDs")
    parser.add_argument("--user-ids", nargs="+", type=int, help="List of user IDs to rank. If not provided, will use test split.")
    parser.add_argument("--retrieval-dir", required=True, type=str, help="Directory containing retrieval results")
    parser.add_argument("--user-log-path", type=str, required=True, help="Path to user interaction log")
    parser.add_argument("--user-emb-dir", type=str, required=False, help="Directory containing user embeddings")
    parser.add_argument("--streamer-emb-dir", type=str, required=True, help="Directory containing streamer embeddings")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for ranked results")
    parser.add_argument("--topk", type=int, default=DEFAULT_K, help="Number of top items to retrieve per user")
    parser.add_argument("--debug-limit", type=int, default=None, help="Limit number of users for debugging purposes")
    parser.add_argument("--model-config", required=True, help="Path to YAML config file")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to model checkpoint file")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for ranking")
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_model(config, input_dim, model_path, device):
    model_name = config["model"]
    d_model = config.get("d_model", 256)
    if model_name == "mlp":
        from src.ranker.torch.models.mlp_ranker import MLPRanker
        model =  MLPRanker(input_dim=input_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model
    elif model_name == "transformer":
        from src.ranker.torch.models.transformer_ranker import TransformerRanker
        model = TransformerRanker(input_dim=input_dim, d_model=d_model, n_layers=config.get("n_layers", 2)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model
    elif model_name == "cross_interaction":
        from src.ranker.torch.models.cross_interaction import CrossInteractionRanker
        model = CrossInteractionRanker(input_dim=input_dim, d_model=d_model).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model
    elif model_name == "contextual":
        from src.ranker.torch.models.contextual_ranker import ContextualRanker
        model = ContextualRanker(input_dim=input_dim, proj_dim=d_model).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model
    elif model_name == "contextual_positional":
        from src.ranker.torch.models.contextual_positional_ranker import ContextualRanker
        model = ContextualRanker(input_dim=input_dim, proj_dim=d_model, max_history_len=config.get("max_history_len", MAX_HISTORY_LEN)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model
    else:
        raise ValueError(f"Unknown model type: {model_name}")

def main():
    args = parse_args()
    model_cfg = load_config(args.model_config)

    # Ensure output directory exists
    ensure_dir_exists(args.out_dir)

    # load inference user IDs
    if args.test_path:
        print(f"Loading user IDs from {args.test_path}...")
        user_ids = load_split(args.test_path)[USER_ID_COL].unique()
    elif args.user_ids:
        print(f"Loading user IDs from {args.user_ids}...")
        user_ids = list(args.user_ids)
    else:
        raise ValueError("Either 'test_path' or 'user_ids' must be provided in the config.")

    # for debugging purposes, limit user_ids
    if args.debug_limit:
        print(f"Limiting user IDs to {args.debug_limit} for debugging...")
        user_ids = user_ids[:args.debug_limit]

    # Load item embeddings and lookups
    print("Loading item store...")
    item_store_cfg = EmbeddingStoreConfig(
        data_dir=args.streamer_emb_dir,
        id_column=STREAMER_ID_COL,
        normalize=False,  # assumption: embeddings are already normalized
    )
    item_store = EmbeddingStore(item_store_cfg)
    print(f"Item store loaded with {item_store.count} items of dimension {item_store.dim}.")

    # Initialize user embedder
    print("Initializing user embedder...")
    user_embedder_cfg = UserEmbedderConfig(
        user_log_path=args.user_log_path,
        user_col=USER_ID_COL,
        item_col=STREAMER_ID_COL,
        max_history_len=MAX_HISTORY_LEN,
        pooling="mean",
        fallback_strategy="random",
        normalize=True, 
        rng_seed=42
    )
    user_embedder = UserEmbedder(user_embedder_cfg, item_store)

    if args.user_emb_dir:
        print("Loading user embeddings...")
        user_store_cfg = EmbeddingStoreConfig(
            data_dir=args.user_emb_dir,
            id_column=USER_ID_COL,
            normalize=False,  # assumption: embeddings are already normalized
        )
        user_store = EmbeddingStore(user_store_cfg)
        print(f"User store loaded with {user_store.count} users of dimension {user_store.dim}.")

    # Initialize ranker
    print("Initializing ranker...")
    model = build_model(
        config=model_cfg,
        input_dim=item_store.dim,
        model_path=args.ckpt_path,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    ranker_cfg = RankerConfig(
        contextual=True if "contextual" in model_cfg["model"] else False,
        max_history_len=MAX_HISTORY_LEN,
        item_id_col=STREAMER_ID_COL
    )

    ranker = Ranker(
        model=model,
        item_store=item_store,
        user_embedder=user_embedder,
        cfg=ranker_cfg,
        user_store=user_store if args.user_emb_dir else None,
    )

    # Rank users
    print(f"Ranking {len(user_ids)} users...")
    results = ranker.rank(
        user_ids, 
        args.retrieval_dir, 
        args.topk, 
        batch_size=args.batch_size,
    )

    # Save results
    ranker.dump_results(results, args.out_dir)

if __name__ == "__main__":
    main()