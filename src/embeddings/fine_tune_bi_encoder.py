from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import models
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch
import random
import os
import argparse

from src.config import USER_ID_COL, STREAMER_ID_COL

# ---------- Load Item Encoder ----------
encoder_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
item_encoder = SentenceTransformer(encoder_name)

# ---------- Load Item Corpus ----------
item_df = pd.read_csv("data/item_corpus.csv")  # [streamer_id, item_text]
item_id_to_text = dict(zip(item_df["streamer_id"], item_df["item_text"]))

# ---------- Load Train Interactions ----------
train_df = pd.read_csv("data/train_interactions.csv")  # [user_id, streamer_id]
user_to_items = train_df.groupby("user_id")["streamer_id"].apply(list).to_dict()

# ---------- Precompute Item Embeddings ----------
print("› Encoding all items...")
item_ids = list(item_id_to_text.keys())
item_texts = [item_id_to_text[i] for i in item_ids]
item_embeddings = item_encoder.encode(item_texts, convert_to_tensor=True, show_progress_bar=True)
item_emb_dict = {item_id: emb for item_id, emb in zip(item_ids, item_embeddings)}

# ---------- Build User Embeddings ----------
print("› Building user embeddings...")
user_emb_dict = {}
for user_id, item_ids in user_to_items.items():
    valid_embs = [item_emb_dict[i] for i in item_ids if i in item_emb_dict]
    if valid_embs:
        user_emb_dict[user_id] = torch.stack(valid_embs).mean(dim=0)
print(f"Total users with valid embeddings: {len(user_emb_dict)}")

# ---------- Prepare Positive Pairs ----------
examples = []
for user_id, pos_item_ids in user_to_items.items():
    if user_id not in user_emb_dict:
        continue
    user_emb = user_emb_dict[user_id]
    for item_id in pos_item_ids:
        if item_id in item_id_to_text:
            examples.append(InputExample(texts=[user_emb, item_id_to_text[item_id]]))

# ---------- Custom Dataset ----------
class StaticUserEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

train_dataset = StaticUserEmbeddingDataset(examples)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ---------- Define Bi-Encoder Model ----------
word_embedding_model = models.Transformer(encoder_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

bi_encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# ---------- Contrastive Loss ----------
train_loss = losses.MultipleNegativesRankingLoss(model=bi_encoder)

# ---------- Fine-Tune ----------
output_dir = "output/fine_tuned_item_encoder"
os.makedirs(output_dir, exist_ok=True)

print("› Fine-tuning bi-encoder...")
bi_encoder.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path=output_dir
)

print(f"Model saved to {output_dir}")


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

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a bi-encoder for item retrieval")
    parser.add_argument("--item-corpus", required=True, help="Path to item corpus CSV or parquet file")
    parser.add_argument("--train-interactions", required=True, help="Path to training interactions CSV or parquet file")
    parser.add_argument("--output-dir", required=True, help="Directory to save the fine-tuned model")
    args = parser.parse_args()

    encoder_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    item_encoder = SentenceTransformer(encoder_name)
    
    # load item corpus
    item_df = pd.read_parquet(args.item_corpus) if args.item_corpus.endswith('.parquet') else pd.read_csv(args.item_corpus)
    item_id_to_text = dict(zip(item_df[STREAMER_ID_COL], item_df["item_sentence"]))

    # load training interactions
    train_df = pd.read_parquet(args.train_interactions) if args.train_interactions.endswith('.parquet') else pd.read_csv(args.train_interactions)
    user_to_items = train_df.groupby(USER_ID_COL)[STREAMER_ID_COL].apply(list).to_dict() # {user_id: [item_ids]}

    # precompute item embeddings
    print("› Encoding all items...")
    item_ids = list(item_id_to_text.keys())
    item_texts = [item_id_to_text[i] for i in item_ids]
    item_embeddings = item_encoder.encode(item_texts, convert_to_tensor=True, show_progress_bar=True)
    item_id_to_emb = {item_id: emb for item_id, emb in zip(item_ids, item_embeddings)}

    # build user embeddings
    print("› Building user embeddings...")
    user_id_to_emb = {}
    for user_id, item_ids in user_to_items.items():
        valid_embs = [item_id_to_emb[i] for i in item_ids if i in item_id_to_emb]
        if valid_embs:
            user_id_to_emb[user_id] = torch.stack(valid_embs).mean(dim=0)
    print(f"Total users with valid embeddings: {len(user_id_to_emb)}")

