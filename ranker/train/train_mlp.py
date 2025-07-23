import yaml
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from ranker.models.mlp_ranker import MLPRanker
from ranker.data.dataset import PairwiseRankingDataset
from ranker.data.triplet_sampling import sample_triplets
from ranker.utils.io import (
    load_split,
    load_embedding_and_lookup,
    ensure_dir_exists
)

from ranker.train.trainer import RankerTrainer
from ranker.utils.metrics import bpr_loss
from src.config import USER_ID_COL, STREAMER_ID_COL

MODEL_NAME = "mlp_ranker"

def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP Ranker (YAML config)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    config = load_config(args.config)

    # load training data
    train_df = load_split(config["train_split"])

    # Sample negatives for training
    print("› Constructing triplets...")
    streamers_in_train = set(train_df[STREAMER_ID_COL].unique())
    triplets = sample_triplets(train_df, streamers_in_train)
    print(f"    Sampled {len(triplets)} triplets for training.")

    # load embeddings and lookup tables
    print(f"Loading user embeddings from {config['user_emb_dir']}")
    user_embeddings, user_lookup = load_embedding_and_lookup(config["user_emb_dir"], USER_ID_COL)
    print(f"Loading streamer embeddings from {config['streamer_emb_dir']}")
    item_embeddings, item_lookup = load_embedding_and_lookup(config["streamer_emb_dir"], STREAMER_ID_COL)

    print(f"user embeddings shape: {user_embeddings.shape}")
    print(f"item embeddings shape: {item_embeddings.shape}")

    # prepare dataset and dataloader
    print("Preparing dataset and dataloader...")
    dataset = PairwiseRankingDataset(
        user_embeddings=user_embeddings,
        user_lookup=user_lookup,
        item_embeddings=item_embeddings,
        item_lookup=item_lookup,
        triplets=triplets
    )
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # initialize model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_dim = user_embeddings.shape[1]
    model = MLPRanker(input_dim=3 * input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # training
    ensure_dir_exists(config["out_dir"])
    model_path = f"{config['out_dir']}/{MODEL_NAME}.pth"
    trainer = RankerTrainer(
        model=model,
        device=device,
        optimizer=optimizer,
        loss_fn=bpr_loss,
        save_path=model_path,
        patience=config["patience"], 
        log_path=f"{config['out_dir']}/training.csv",
        config=config, 
        config_path=f"{config['out_dir']}/config.yaml"
    )
    trainer.train(dataloader, num_epochs=config["epochs"])

if __name__ == "__main__":
    main()