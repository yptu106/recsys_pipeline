import yaml
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from ranker.data.dataset import PairwiseRankingDataset, PointwiseRankingDataset, ContextualPairwiseDataset
from ranker.data.triplet_sampling import sample_triplets, build_triplets_from_val_df
from ranker.data.triplet_sampling import (
    collect_retrieval_results,
    build_ranker_training_data,
    build_ranker_eval_data,
    build_pairwise_triplets_from_pointwise, 
    average_positives_per_user, 
    build_pointwise_samples, 
    build_user_log
)
from ranker.data.collate import listwise_collate, contextual_collate
from ranker.utils.io import (
    load_split,
    load_embedding_and_lookup,
    ensure_dir_exists
)

from ranker.train.trainer import RankerTrainer
from ranker.utils.metrics import bpr_loss, listwise_loss
from src.config import USER_ID_COL, STREAMER_ID_COL

PRE_TRAIN_EPOCHS = 100  # Default pretraining epochs

def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer Ranker (YAML config)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_model(config, input_dim, device):
    model_name = config["model"]
    d_model = config.get("d_model", 256)
    if model_name == "mlp":
        from ranker.models.mlp_ranker import MLPRanker
        return MLPRanker(input_dim=input_dim).to(device)
    elif model_name == "transformer":
        from ranker.models.transformer_ranker import TransformerRanker
        return TransformerRanker(input_dim=input_dim, d_model=d_model, n_layers=config.get("n_layers", 2)).to(device)
    elif model_name == "cross_interaction":
        from ranker.models.cross_interaction import CrossInteractionRanker
        return CrossInteractionRanker(input_dim=input_dim, d_model=d_model).to(device)
    elif model_name == "contextual":
        from ranker.models.contextual_ranker import ContextualRanker
        return ContextualRanker(input_dim=input_dim, proj_dim=d_model).to(device)
    elif model_name == "contextual_positional":
        from ranker.models.contextual_positional_ranker import ContextualRanker
        return ContextualRanker(input_dim=input_dim, proj_dim=d_model, max_history_len=config.get("max_history_len", 50)).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

def build_contextual_pairwise_dataloader(user_histories, item_embeddings, item_lookup, triplets, batch_size, shuffle):
    dataset = ContextualPairwiseDataset(
        user_histories=user_histories,
        item_embeddings=item_embeddings,
        item_lookup=item_lookup,
        triplets=triplets
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=contextual_collate)

def build_pairwise_dataloader(user_emb, user_lookup, item_emb, item_lookup, triplets, batch_size, shuffle):
    dataset = PairwiseRankingDataset(
        user_embeddings=user_emb,
        user_lookup=user_lookup,
        item_embeddings=item_emb,
        item_lookup=item_lookup,
        triplets=triplets
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def build_pointwise_dataloader(user_emb, user_lookup, item_emb, item_lookup, pairs, batch_size, shuffle):
    dataset = PointwiseRankingDataset(
        user_embeddings=user_emb,
        user_lookup=user_lookup,
        item_embeddings=item_emb,
        item_lookup=item_lookup,
        pairs=pairs
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=listwise_collate)

def main():
    args = parse_args()
    config = load_config(args.config)

    # load training and validation splits
    train_df = load_split(config["train_split"])
    val_df = load_split(config["val_split"])
    test_df = load_split(config["test_split"])

    val_items = val_df[val_df['label'] == 1].set_index('user_id')['streamer_id'].to_dict()
    test_items = test_df[test_df['label'] == 1].set_index('user_id')['streamer_id'].to_dict()
    # del val_df, test_df  # free memory

    # load embeddings and lookup tables
    print(f"Loading user embeddings from {config['user_emb_dir']}")
    user_embeddings, user_lookup = load_embedding_and_lookup(config["user_emb_dir"], USER_ID_COL)
    print(f"Loading streamer embeddings from {config['streamer_emb_dir']}")
    item_embeddings, item_lookup = load_embedding_and_lookup(config["streamer_emb_dir"], STREAMER_ID_COL)

    # initialize model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    input_dim = user_embeddings.shape[1]
    model = build_model(config, input_dim, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    ensure_dir_exists(config["out_dir"])

    trainer = RankerTrainer(
        model=model,
        device=device,
        optimizer=optimizer,
        save_path=f"{config['out_dir']}/{config['model']}.pth",
        patience=config["patience"], 
        log_path=f"{config['out_dir']}/training.csv",
        config=config, 
        config_path=f"{config['out_dir']}/config.yaml", 
        val_freq=config["val_freq"]
    )


    # ===== Stage 1: Pretrain on random negatives =====
    if config.get("pretrain", True):
        print("Stage 1: Pretraining with random negatives...")
        streamers_in_train = set(train_df[STREAMER_ID_COL].unique())
        pretrain_triplets = sample_triplets(train_df, streamers_in_train)
        val_triplets = build_triplets_from_val_df(val_df)

        if "contextual" in config["model"]: # check if we need to load user histories
            print("Using contextual pairwise dataset for pretraining...")
            user_histories = build_user_log(config["user_log_path"], max_history_len=config.get("max_history_len", 50))
            pretrain_loader = build_contextual_pairwise_dataloader(
                user_histories=user_histories,
                item_embeddings=item_embeddings,
                item_lookup=item_lookup,
                triplets=pretrain_triplets,
                batch_size=config["batch_size"],
                shuffle=True
            )
            val_loader = build_contextual_pairwise_dataloader(
                user_histories=user_histories,
                item_embeddings=item_embeddings,
                item_lookup=item_lookup,
                triplets=val_triplets,
                batch_size=config["batch_size"],
                shuffle=False
            )

        else:
            pretrain_loader = build_pairwise_dataloader(user_embeddings, user_lookup, item_embeddings, item_lookup, pretrain_triplets, config["batch_size"], shuffle=True)
            val_loader = build_pairwise_dataloader(user_embeddings, user_lookup, item_embeddings, item_lookup, val_triplets, config["batch_size"], shuffle=False)

        print(f"Pretraining with {len(pretrain_loader.dataset)} triplets...")
        print(f"Validation with {len(val_loader.dataset)} triplets...")
        trainer.train(
            pretrain_loader, 
            num_epochs=config.get("epochs", PRE_TRAIN_EPOCHS), 
            val_dataloader=val_loader, 
            # val_dataloader=None, # no validation during pretraining
            loss_fn=bpr_loss, 
            mode="contextual_pairwise" if "contextual" in config["model"] else "pairwise"
        )

    # ===== Stage 2: Fine-tune on retrieval-based negatives =====
    if config.get("skip_stage_2", False):
        print("Skipping Stage 2: Fine-tuning with retrieval-based hard negatives.")
        return
    print("Stage 2: Fine-tuning with retrieval-based hard negatives...")
    retrieval_results = collect_retrieval_results(config["retrieval_dir"])

    # build pointwise dataset from retrieval results
    train_df_hard = build_ranker_training_data(train_interactions_df=train_df, retrieval_results=retrieval_results, val_items=val_items, test_items=test_items)
    val_df_hard = build_ranker_eval_data(eval_items=val_items, retrieval_results=retrieval_results)

    if config.get("loss_type", "pairwise") == "pairwise":
        print("fine-tuning with pairwise loss...")
        train_triplets = build_pairwise_triplets_from_pointwise(train_df_hard)
        val_triplets = build_pairwise_triplets_from_pointwise(val_df_hard)

        if config["model"] == "contextual":
            user_histories = build_user_log(config["user_log_path"])
            train_loader = build_contextual_pairwise_dataloader(
                user_histories=user_histories,
                item_embeddings=item_embeddings,
                item_lookup=item_lookup,
                triplets=train_triplets,
                batch_size=config["batch_size"],
                shuffle=True
            )
            val_loader = build_contextual_pairwise_dataloader(
                user_histories=user_histories,
                item_embeddings=item_embeddings,
                item_lookup=item_lookup,
                triplets=val_triplets,
                batch_size=config["batch_size"],
                shuffle=False
            )
        else:
            train_loader = build_pairwise_dataloader(user_embeddings, user_lookup, item_embeddings, item_lookup, train_triplets, config["batch_size"], shuffle=True)
            val_loader = build_pairwise_dataloader(user_embeddings, user_lookup, item_embeddings, item_lookup, val_triplets, config["batch_size"], shuffle=False)

        trainer.train(
            train_loader,
            num_epochs=config["epochs"],
            val_dataloader=val_loader,
            loss_fn=bpr_loss,
            mode="contextual_pairwise" if "contextual" in config["model"] else "pairwise"
        )
    else:
        print("fine-tuning with pointwise loss...")
        train_pairs = build_pointwise_samples(train_df_hard)
        val_pairs = build_pointwise_samples(val_df_hard)

        train_loader = build_pointwise_dataloader(user_embeddings, user_lookup, item_embeddings, item_lookup, train_pairs, config["batch_size"], shuffle=True)
        val_loader = build_pointwise_dataloader(user_embeddings, user_lookup, item_embeddings, item_lookup, val_pairs, config["batch_size"], shuffle=False)

        trainer.train(
            train_loader,
            num_epochs=config["epochs"],
            val_dataloader=val_loader,
            loss_fn=listwise_loss,
            mode="pointwise"
        )

if __name__ == "__main__":
    main()