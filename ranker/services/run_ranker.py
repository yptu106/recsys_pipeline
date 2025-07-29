import yaml
import argparse
import numpy as np
import pandas as pd
import torch

from ranker.services.ranker import Ranker, UserEmbFallbackConfig
# from ranker.models.transformer_ranker import TransformerRanker, MLPRanker, CrossInteractionRanker, ContextualRanker
from ranker.utils.io import load_embedding_and_lookup, ensure_dir_exists, get_emb_paths, load_split
from src.config import USER_ID_COL, STREAMER_ID_COL

QUIET_MODE = False
DEFAULT_K = 100

def parse_args():
    parser = argparse.ArgumentParser(description="Run Transformer Ranker")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_model(model_name, model_path, input_dim, d_model, device):
    if model_name == "mlp":
        from ranker.models.mlp_ranker import MLPRanker
        model =  MLPRanker(input_dim=input_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model
    elif model_name == "transformer":
        from ranker.models.transformer_ranker import TransformerRanker
        model = TransformerRanker(input_dim=input_dim, d_model=d_model).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model
    elif model_name == "cross_interaction":
        from ranker.models.cross_interaction import CrossInteractionRanker
        model = CrossInteractionRanker(input_dim=input_dim, d_model=d_model).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model
    elif model_name == "contextual":
        from ranker.models.contextual_ranker import ContextualRanker
        model = ContextualRanker(input_dim=input_dim, proj_dim=d_model).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model
    else:
        raise ValueError(f"Unknown model type: {model_name}")

def main():
    args = parse_args()
    config = load_config(args.config)
    fallback_config = config["fallback"]

    # Ensure output directory exists
    ensure_dir_exists(config["out_dir"])

    # load inference user IDs
    if config.get("test_path"):
        print(f"Loading user IDs from {config['test_path']}...")
        user_ids = load_split(config["test_path"])[USER_ID_COL].unique()
    elif config.get("user_ids"):
        print(f"Loading user IDs from {config['user_ids']}...")
        user_ids = pd.read_csv(config["user_ids"])[USER_ID_COL].unique()
    else:
        raise ValueError("Either 'test_path' or 'user_ids' must be provided in the config.")

    # Load item embeddings and lookups
    print("Loading item embeddings and lookups...")
    item_embeddings, item_lookup = load_embedding_and_lookup(config["streamer_emb_dir"], STREAMER_ID_COL)
    print(f"Item embeddings shape: {item_embeddings.shape}")
    print(f"Item lookup size: {len(item_lookup)}")

    # Load user embeddings and lookups if provided
    user_embeddings, user_lookup = None, None
    if config.get("user_emb_dir"):
        print("Loading user embeddings and lookups...")
        user_embeddings, user_lookup = load_embedding_and_lookup(config["user_emb_dir"], USER_ID_COL)
        print(f"User embeddings shape: {user_embeddings.shape}")
        print(f"User lookup size: {len(user_lookup)}")
    
    # TODO: refactor
    # item_emb_path, item_lookup_path = get_emb_paths(streamer_emb_dir)
    fallback_config = UserEmbFallbackConfig(
        emb_path=fallback_config["emb_path"],
        lookup_path=fallback_config["lookup_path"],
        user_log_path=fallback_config["user_log_path"],
        n_fallback=fallback_config["n_fallback"],
        max_history=fallback_config["max_history"]
    )

    # Initialize ranker
    print("Initializing ranker...")
    model = build_model(
        model_name=config["model"],
        model_path=config["model_path"],
        input_dim=item_embeddings.shape[1],
        d_model=config["d_model"],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    ranker = Ranker(
        model=model,
        item_embeddings=item_embeddings,
        item_lookup=item_lookup,
        user_embeddings=user_embeddings, 
        user_lookup=user_lookup, 
        user_fallback_config=fallback_config
    )

    # Rank users
    print(f"Ranking {len(user_ids)} users...")
    results = ranker.rank(user_ids, config["retrieval_dir"], config["topk"])

    # Save results
    ranker.dump_results(results, config["out_dir"])

if __name__ == "__main__":
    main()