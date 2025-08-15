"""
run_trainer.py

Fine-tune encoder using triplet loss with YAML config.

Usage:
python -m src.encoder.train.run_trainer \
    --config <path_to_yaml_config>

"""

import yaml
import argparse
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, models

from src.config import USER_ID_COL, STREAMER_ID_COL
from src.encoder.utils.io import ensure_dir_exists
from src.encoder.train.trainer import EncoderTrainer
from src.encoder.utils.triplet_sampling import sample_triplets, build_triplets_from_val_df

MODEL_NAME = "MiniLM"

def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer Ranker (YAML config)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    config = load_config(args.config)

    # load item corpus
    item_df = pd.read_parquet(config["item_corpus"]) if config["item_corpus"].endswith('.parquet') else pd.read_csv(config["item_corpus"])
    item_id_to_text = dict(zip(item_df[STREAMER_ID_COL], item_df[config["text_col"]]))

    # load training interactions
    train_df = pd.read_parquet(config["train_interactions"]) if config["train_interactions"].endswith('.parquet') else pd.read_csv(config["train_interactions"])
    user_to_items = train_df.groupby(USER_ID_COL)[STREAMER_ID_COL].apply(list).to_dict() # {user_id: [item_ids]}
    all_items = set(item_df[STREAMER_ID_COL])

    # sample triplets
    triplets = sample_triplets(train_df, all_items)
    print(f"Sampled {len(triplets)} triplets for training")
    triplets = [t for t in triplets if t[0] in user_to_items and t[1] in item_id_to_text and t[2] in item_id_to_text]
    print(f"Filtered down to {len(triplets)} valid triplets")

    # load validation split
    if config.get("val_split_path") is None:
        print("No validation split provided, training without validation")
        val_triplets = None
    else:
        print(f"Loading validation split from {config['val_split_path']}")
        val_df = pd.read_parquet(config["val_split_path"]) if config["val_split_path"].endswith('.parquet') else pd.read_csv(config["val_split_path"])
        val_triplets = build_triplets_from_val_df(val_df)
        print(f"Loaded {len(val_triplets)} validation triplets")

    # # ---for development purposes---
    # triplets = triplets[:100]  # limit to 100 triplets for quick testing
    # val_triplets = val_triplets[:10]  # limit to 10 validation triplets

    if config["resume_from"]:
        print(f"Resuming training from checkpoint: {config['resume_from']}")
        encoder = SentenceTransformer(config['resume_from'])
    else:
        print("Initializing new encoder model")
        encoder_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        word_model = models.Transformer(encoder_name)
        pooling_model = models.Pooling(word_model.get_word_embedding_dimension())
        encoder = SentenceTransformer(modules=[word_model, pooling_model])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)

    loss_function = torch.nn.TripletMarginLoss(margin=config["margin"], p=2)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=config["lr"])

    ensure_dir_exists(config["out_dir"])
    model_path = f"{config['out_dir']}/{MODEL_NAME}"
    trainer = EncoderTrainer(
        encoder=encoder,
        device=device,
        optimizer=optimizer,
        loss_fn=loss_function,
        batch_size=config["batch_size"],
        save_path=model_path,
        patience=config["patience"],
        log_path=f"{config['out_dir']}/training.csv",
        config=config,
        config_path=f"{config['out_dir']}/config.yaml", 
        val_freq=config["val_freq"]
    )
    trainer.train(
        user_to_items=user_to_items,
        item_id_to_text=item_id_to_text,
        train_triplets=triplets,
        val_triplets=val_triplets,
        num_epochs=config["epochs"]
    )

if __name__ == "__main__":
    main()