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
from ranker.recbole.services.ranker import RecBoleRanker
from ranker.utils.io import ensure_dir_exists, load_split


def parse_args():
    parser = argparse.ArgumentParser(description="Run Transformer Ranker")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    run_config = load_config(args.config)

    # Ensure output directory exists
    ensure_dir_exists(run_config["out_dir"])

    # load inference user IDs
    if run_config.get("test_path"):
        print(f"Loading user IDs from {run_config['test_path']}...")
        user_ids = load_split(run_config["test_path"])[USER_ID_COL].unique()
    elif run_config.get("user_ids"):
        print(f"Loading user IDs from {run_config['user_ids']}...")
        user_ids = pd.read_csv(run_config["user_ids"])[USER_ID_COL].unique()
    else:
        raise ValueError("Either 'test_path' or 'user_ids' must be provided in the run_config.")


    # for debugging purposes, limit user_ids
    if run_config.get("debug_limit"):
        print(f"Limiting user IDs to {run_config['debug_limit']} for debugging...")
        user_ids = user_ids[:run_config["debug_limit"]]


    print("Loading ranker...")
    # load config and recreate dataset
    recbole_config = Config(model=run_config["model"], config_file_list=[run_config["recbole_config_path"]])
    dataset = create_dataset(recbole_config)

    if run_config["model"] == "SASRecF":
        train_data, _, _ = data_preparation(recbole_config, dataset)

        model_cls = get_model(recbole_config["model"])
        model = model_cls(recbole_config, train_data._dataset).to(recbole_config["device"])
    else:
        # load trained BPR model
        model_cls = get_model(recbole_config['model'])
        model = model_cls(recbole_config, dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = run_config["checkpoint_path"]
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Initialize RecBoleRanker
    ranker = RecBoleRanker(model, dataset, recbole_config)

    # rank users
    print(f"Ranking {len(user_ids)} users...")
    results = ranker.rank(
        user_ids=user_ids,
        retrieval_dir=run_config["retrieval_dir"],
        topk=run_config.get("topk", 100)
    )

    # Save results
    ranker.dump_results(results, run_config["out_dir"])

if __name__ == "__main__":
    main()