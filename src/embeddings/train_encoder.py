import os
import argparse
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, models

from src.config import USER_ID_COL, STREAMER_ID_COL


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

def build_triplets_from_val_df(val_df):
    """Build (user_id, pos_item_id, neg_item_id) from validation DataFrame with label column."""
    triplets = []
    for user_id, group in val_df.groupby(USER_ID_COL):
        pos_items = group[group["label"] == 1][STREAMER_ID_COL].tolist()
        neg_items = group[group["label"] == 0][STREAMER_ID_COL].tolist()

        if not pos_items or not neg_items:
            continue
        
        pos_item = pos_items[0]  # only one positive per user
        neg_item = random.choice(neg_items)
        
        triplets.append((user_id, pos_item, neg_item))
    
    return triplets

def build_user_embedding_cache(encoder, user_to_items, item_id_to_text):
    """Builds a cache of user embeddings by averaging item embeddings."""
    user_embeddings = {}
    for user_id, item_ids in user_to_items.items():
        item_texts = [item_id_to_text[item_id] for item_id in item_ids if item_id in item_id_to_text]
        if not item_texts:
            continue
        user_embedding = encoder.encode(item_texts, convert_to_tensor=True, show_progress_bar=False)
        user_embedding = torch.mean(user_embedding, dim=0)
        user_embeddings[user_id] = user_embedding
    return user_embeddings

class TripletDatasetWithUserEmbeddings(Dataset):
    def __init__(self, user_embeddings, item_id_to_text, triplets):
        self.user_embeddings = user_embeddings # {user_id: embedding}
        self.item_id_to_text = item_id_to_text
        self.triplets = triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        user_id, pos_item_id, neg_item_id = self.triplets[idx]

        # get user embedding
        user_emb = self.user_embeddings[user_id]
        pos_item_text = self.item_id_to_text[pos_item_id]
        neg_item_text = self.item_id_to_text[neg_item_id]

        return InputExample(texts=[user_emb, pos_item_text, neg_item_text])


class DynamicUserEmbeddingDataset(Dataset):
    def __init__(self, user_to_items, item_id_to_text, triplets, encoder):
        self.user_to_items = user_to_items
        self.item_id_to_text = item_id_to_text
        self.triplets = triplets
        self.encoder = encoder
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        user_id, pos_item_id, neg_item_id = self.triplets[idx]

        # get user's interacted item texts
        user_items = self.user_to_items.get(user_id, [])
        user_item_texts = [self.item_id_to_text[item_id] for item_id in user_items if item_id in self.item_id_to_text]

        # encode user via mean pooling of item embeddings
        user_embedding = self.encoder.encode(user_item_texts, convert_to_tensor=True, show_progress_bar=False)
        user_embedding = torch.mean(user_embedding, dim=0)

        pos_item_text = self.item_id_to_text[pos_item_id]
        neg_item_text = self.item_id_to_text[neg_item_id]

        return InputExample(texts=[user_embedding, pos_item_text, neg_item_text])

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a bi-encoder for item retrieval")
    parser.add_argument("--item-corpus", required=True, help="Path to item corpus CSV or parquet file")
    parser.add_argument("--train-interactions", required=True, help="Path to training interactions CSV or parquet file")
    parser.add_argument("--val-split-path", type=str, required=True, help="Path to validation split CSV or parquet file")
    parser.add_argument("--output-dir", required=True, help="Directory to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--margin", type=float, default=0.2, help="Margin for triplet loss")
    args = parser.parse_args()

    # load item corpus
    item_df = pd.read_parquet(args.item_corpus) if args.item_corpus.endswith('.parquet') else pd.read_csv(args.item_corpus)
    item_id_to_text = dict(zip(item_df[STREAMER_ID_COL], item_df["item_sentence"]))

    # load training interactions
    train_df = pd.read_parquet(args.train_interactions) if args.train_interactions.endswith('.parquet') else pd.read_csv(args.train_interactions)
    user_to_items = train_df.groupby(USER_ID_COL)[STREAMER_ID_COL].apply(list).to_dict() # {user_id: [item_ids]}
    all_items = set(item_df[STREAMER_ID_COL])

    # sample triplets
    triplets = sample_triplets(train_df, all_items)
    print(f"Sampled {len(triplets)} triplets for training")
    triplets = [t for t in triplets if t[0] in user_to_items and t[1] in item_id_to_text and t[2] in item_id_to_text]
    print(f"Filtered down to {len(triplets)} valid triplets")

    # load validation split
    val_df = pd.read_parquet(args.val_split_path) if args.val_split_path.endswith('.parquet') else pd.read_csv(args.val_split_path)
    val_triplets = build_triplets_from_val_df(val_df)
    print(f"Loaded {len(val_triplets)} validation triplets")

    # define shared encoder to be trained
    encoder_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    word_model = models.Transformer(encoder_name)
    pooling_model = models.Pooling(word_model.get_word_embedding_dimension())
    encoder = SentenceTransformer(modules=[word_model, pooling_model])

    # wrap dataset
    dataset = DynamicUserEmbeddingDataset(user_to_items, item_id_to_text, triplets, encoder)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # define loss function
    loss_function = losses.TripletLoss(
        model=encoder, 
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=args.margin
    )

    # training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)

    os.makedirs(args.output_dir, exist_ok=True)
    best_model_path = os.path.join(args.output_dir, "item_encoder.pt")

    print("› Training encoder with triplet loss...")
    encoder.fit(
        train_objectives=[(dataloader, loss_function)],
        epochs=args.epochs,
        warmup_steps=100,
        output_path=best_model_path,
        show_progress_bar=True
    )

    print(f"Training complete. Best model saved to {best_model_path}")

if __name__ == "__main__":
    main()