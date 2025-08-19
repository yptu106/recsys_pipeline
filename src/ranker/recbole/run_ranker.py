"""
run_ranker.py

This script runs the RecBole ranking process using a model checkpoint.
It expects a configuration file, a directory containing retrieval results,
and an output directory for the ranked results.

Usage:
    python src.ranker.recbole.run_ranker \
        --test-path <path_to_test_dataset> \
        --retrieval-dir <path_to_retrieval_results> \
        --out-dir <path_to_output_directory> \
        --model <model_name> \
        --recbole-config-path <path_to_recbole_config_file> \
        --checkpoint-path <path_to_model_checkpoint> \
        [--topk 100] \
        [--batch-size 256] \
        [--debug-limit <number_of_users>]

This script is designed to be run in the context of a RecBole project.
It requires the RecBole library to be installed and properly configured.

"""

import torch
import yaml
import argparse
import json
import torch
import pathlib
import pandas as pd
from tqdm import tqdm
from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation,
)
from recbole.utils import get_model
from src.config import USER_ID_COL, STREAMER_ID_COL
from src.ranker.recbole.ranker import RecBoleRanker
from src.ranker.utils.io import ensure_dir_exists, load_split


def parse_args():
    parser = argparse.ArgumentParser(description="Run Transformer Ranker")
    # directory and file paths
    parser.add_argument("--test-path", type=str, help="Path to the test dataset")
    parser.add_argument("--retrieval-dir", type=str, help="Directory containing retrieval results")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for results")
    # RecBole config
    parser.add_argument("--model", type=str, required=True, help="Model name to use for ranking")
    parser.add_argument("--recbole-config-path", type=str, required=True, help="Path to the RecBole configuration file")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to the model checkpoint")
    # inference options
    parser.add_argument("--topk", type=int, default=100, help="Number of top items to retrieve for each user")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for inference")
    # additional options
    parser.add_argument("--debug-limit", type=int, default=None, help="Limit the number of users for debugging")
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()

    # load inference user IDs
    print(f"Loading user IDs from {args.test_path}...")
    user_ids = load_split(args.test_path)[USER_ID_COL].unique()

    # for debugging purposes, limit user_ids
    if args.debug_limit:
        print(f"Limiting user IDs to {args.debug_limit} for debugging...")
        user_ids = user_ids[:args.debug_limit]

    print("Loading ranker...")
    # load config and recreate dataset
    recbole_config = Config(model=args.model, config_file_list=[args.recbole_config_path])
    dataset = create_dataset(recbole_config)

    if args.model == "SASRecF":
        train_data, _, _ = data_preparation(recbole_config, dataset)

        model_cls = get_model(recbole_config["model"])
        model = model_cls(recbole_config, train_data._dataset).to(recbole_config["device"])
    else:
        # load trained BPR model
        model_cls = get_model(recbole_config['model'])
        model = model_cls(recbole_config, dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint_path
    print(f"Loading model checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Initialize RecBoleRanker
    ranker = RecBoleRanker(model, dataset, recbole_config)

    # rank users
    print(f"Ranking {len(user_ids)} users...")
    results = ranker.rank(
        user_ids=user_ids,
        retrieval_dir=args.retrieval_dir,
        topk=args.topk,
        batch_size=args.batch_size
    )

    # Ensure output directory exists
    ensure_dir_exists(args.out_dir)

    # Save results
    ranker.dump_results(results, args.out_dir)

if __name__ == "__main__":
    main()