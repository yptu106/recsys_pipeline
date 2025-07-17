# ranker/train/train_mlp.py

import pathlib
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from ranker.mlp.mlp_ranker import MLPRanker
from ranker.mlp.pairwise_dataset import PairwiseRankingDataset

from src.config import USER_ID_COL, STREAMER_ID_COL
from src.services.retrieval import get_emb_paths

np.random.seed(42)

def sample_triplets(train_df, all_streamers):
    """Assumes train_df only contains positive interactions."""
    triplets = [] # (user_id, pos_item_id, neg_item_id)
    for user_id, group in train_df.groupby(USER_ID_COL):
        pos_items = set(group[STREAMER_ID_COL])
        neg_candidates = list(all_streamers - pos_items)
        if not neg_candidates:
            continue

        # Sample one negative for each positive item
        for pos_item in pos_items:
            neg_item = np.random.choice(neg_candidates)
            triplets.append((user_id, pos_item, neg_item))
    return triplets

def bpr_loss(pos_score, neg_score):
    return -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8))

def train_mlp_ranker(
        user_embeddings, user_lookup, 
        item_embeddings, item_lookup, 
        triplets,
        epochs=10, batch_size=512, lr=1e-3, patience=5,
        out_dir="../models"
    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare dataset
    dataset = PairwiseRankingDataset(
        user_embeddings=user_embeddings,
        user_lookup=user_lookup,
        item_embeddings=item_embeddings,
        item_lookup=item_lookup,
        triplets=triplets
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"user embeddings shape: {user_embeddings.shape}")
    print(f"item embeddings shape: {item_embeddings.shape}")

    # Init model
    emb_dim = user_embeddings.shape[1]
    model = MLPRanker(input_dim=3 * emb_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_model_path = pathlib.Path(out_dir) / "mlp_ranker_best.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for u, pos_i, neg_i in tqdm(dataloader, desc=f"Epoch {epoch}"):
            u, pos_i, neg_i = u.to(device), pos_i.to(device), neg_i.to(device)

            pos_score = model(u, pos_i)
            neg_score = model(u, neg_i)

            loss = bpr_loss(pos_score, neg_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch}] Loss = {avg_loss:.4f}")

        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"   New best model saved at epoch {epoch} with loss {best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping triggered. No improvement for {patience_counter} epochs.")
                break

    print(f"Training finished. Best loss: {best_loss:.4f} at epoch {best_epoch}.")
    print(f"Best model saved to {best_model_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-emb-dir", required=True, help="Directory containing user embeddings and lookup")
    parser.add_argument("--streamer-emb-dir", required=True, help="Directory containing streamer embeddings and lookup")
    parser.add_argument("--train-split", required=True, help="Path to the training split parquet or csv file")
    parser.add_argument("--validation-split", required=True, help="Path to the validation split parquet or csv file")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dir", default="ranker/models", help="Output directory for the trained model")
    args = parser.parse_args()

    # load positive pairs from training split
    train_df = pd.read_parquet(args.train_split) if args.train_split.endswith('.parquet') else pd.read_csv(args.train_split)
    
    # Sample negatives for training 
    print("› Constructing triplets...")
    streamers_in_train = set(train_df[STREAMER_ID_COL].unique())
    triplets = sample_triplets(train_df, streamers_in_train)
    print(f"    Sampled {len(triplets)} triplets for training.")

    # load user and item embeddings
    print(f"Loads user embeddings from {args.user_emb_dir}")
    print(f"Loads streamer embeddings from {args.streamer_emb_dir}")
    user_emb_path, user_lookup_path = get_emb_paths(args.user_emb_dir)
    item_emb_path, item_lookup_path = get_emb_paths(args.streamer_emb_dir)

    user_embeddings = np.load(user_emb_path)
    item_embeddings = np.load(item_emb_path)

    # load lookup tables and convert to dicts
    user_lookup = pd.read_parquet(user_lookup_path).reset_index(drop=True)
    item_lookup = pd.read_parquet(item_lookup_path).reset_index(drop=True)
    user_lookup = dict(zip(user_lookup[USER_ID_COL], user_lookup.index))
    item_lookup = dict(zip(item_lookup[STREAMER_ID_COL], item_lookup.index))

    # Ensure output directory exists
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Train the MLP ranker
    print("Training MLP ranker...")
    train_mlp_ranker(
        user_embeddings=user_embeddings,
        user_lookup=user_lookup,
        item_embeddings=item_embeddings,
        item_lookup=item_lookup,
        triplets=triplets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_dir=args.out_dir
    )

if __name__ == "__main__":
    main()