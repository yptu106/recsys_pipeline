import os
import argparse
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
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
    all_texts = []
    all_user_ids = []

    for user_id, item_ids in user_to_items.items():
        for item_id in item_ids:
            if item_id in item_id_to_text:
                all_texts.append(item_id_to_text[item_id]) # item text
                all_user_ids.append(user_id)               # user_id owning the item text

    print(f"› Encoding {len(all_texts)} item texts for {len(user_to_items)} users...")
    all_embeddings = encoder.encode(all_texts, convert_to_tensor=True, show_progress_bar=True, batch_size=128)

    emb_sum = defaultdict(lambda: torch.zeros_like(all_embeddings[0]))
    emb_count = defaultdict(int)

    for uid, emb in zip(all_user_ids, all_embeddings):
        emb_sum[uid] += emb
        emb_count[uid] += 1

    user_embeddings = {uid: emb_sum[uid] / emb_count[uid] for uid in emb_sum}
    return user_embeddings

def encode_texts(model, texts, device):
    tokens = model.tokenize(texts)  # {'input_ids': ..., 'attention_mask': ...}
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.set_grad_enabled(True):
        outputs = model(tokens)
    return outputs['sentence_embedding']  # [batch_size, hidden_dim]

def evaluate_validation_loss(user_emb_cache, encoder, loss_fn, val_triplets, item_id_to_text, user_to_items, batch_size=32, device="cpu"):
    """Evaluates average triplet loss on the validation set."""
    encoder.eval()

    # Prepare validation dataset and loader
    val_dataset = TripletDatasetWithUserEmbeddings(user_emb_cache, item_id_to_text, val_triplets, device=device)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    total_val_loss = 0.0
    with torch.no_grad():
        for user_emb_batch, pos_text_batch, neg_text_batch in tqdm(val_loader, desc="Validating"):
            user_emb_batch = user_emb_batch.to(device)

            pos_emb_batch = encode_texts(encoder, list(pos_text_batch), device)
            neg_emb_batch = encode_texts(encoder, list(neg_text_batch), device)

            loss = loss_fn(user_emb_batch, pos_emb_batch, neg_emb_batch)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    return avg_val_loss


class TripletDatasetWithUserEmbeddings(Dataset):
    def __init__(self, user_embeddings, item_id_to_text, triplets, device="cpu"):
        self.user_embeddings = user_embeddings # {user_id: embedding}
        self.item_id_to_text = item_id_to_text
        self.triplets = triplets
        self.device = device

    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        user_id, pos_item_id, neg_item_id = self.triplets[idx]

        # get user embedding
        user_emb = self.user_embeddings[user_id].to(self.device)
        pos_item_text = self.item_id_to_text[pos_item_id]
        neg_item_text = self.item_id_to_text[neg_item_id]

        return user_emb, pos_item_text, neg_item_text

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
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the optimizer")
    parser.add_argument("--margin", type=float, default=0.2, help="Margin for triplet loss")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to a checkpoint to resume training from")
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
    
    # # ---for development purposes---
    # triplets = triplets[:1000]  # limit to 1000 triplets for quick testing
    # val_triplets = val_triplets[:100]  # limit to 100 validation trip
    
    if args.resume_from:
        print(f"Resuming training from checkpoint: {args.resume_from}")
        encoder = SentenceTransformer(args.resume_from)
    else:
        # define shared encoder to be trained
        encoder_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        word_model = models.Transformer(encoder_name)
        pooling_model = models.Pooling(word_model.get_word_embedding_dimension())
        encoder = SentenceTransformer(modules=[word_model, pooling_model])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)

    # # define loss function
    # loss_function = losses.TripletLoss(
    #     model=encoder, 
    #     distance_metric=losses.TripletDistanceMetric.COSINE,
    #     triplet_margin=args.margin
    # )

    # sentence transformers' Triplet loss expects triplets to be all texts
    # use `torch.nn.TripletMarginLoss` instead
    loss_function = torch.nn.TripletMarginLoss(margin=args.margin, p=2)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)

    print("› Training encoder with triplet loss...")
    best_loss = float('inf')
    best_epoch = 0
    patience = args.patience
    patience_counter = 0
    for epoch in range(1, args.epochs + 1):
        print(f"\n› Epoch {epoch}/{args.epochs}")

        user_embeddings = build_user_embedding_cache(encoder, user_to_items, item_id_to_text)
        dataset = TripletDatasetWithUserEmbeddings(user_embeddings, item_id_to_text, triplets, device=device)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        print(f"Training on {len(dataloader)} batches with batch size {args.batch_size}")

        total_loss = 0.0
        encoder.train()

        for step, (user_emb_batch, pos_text_batch, neg_text_batch) in enumerate(tqdm(dataloader, desc="Training")):
            # move user embedding to device
            user_emb_batch = user_emb_batch.to(device)

            # encode positive and negative item texts
            pos_emb_batch = encode_texts(encoder, list(pos_text_batch), device)
            neg_emb_batch = encode_texts(encoder, list(neg_text_batch), device)

            loss = loss_function(user_emb_batch, pos_emb_batch, neg_emb_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")

        # TODO: fix error in validation loss evaluation
        # val_loss = evaluate_validation_loss(user_embeddings, encoder, loss_function, val_triplets, item_id_to_text, user_to_items, batch_size=args.batch_size)
        # print(f"› Epoch {epoch} validation loss: {val_loss:.4f}")

        # early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            patience_counter = 0

            best_model_path = os.path.join(args.output_dir, f"best_model")
            encoder.save(best_model_path)
            print(f"New best model saved at epoch {epoch} with loss {best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break


    print(f"Training complete. Best model saved to {best_model_path}")

if __name__ == "__main__":
    main()