import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Union
from collections import defaultdict

class PairwiseRankingDataset(Dataset):
    def __init__(
            self, 
            user_embeddings: np.ndarray, 
            user_lookup: dict, 
            item_embeddings: np.ndarray, 
            item_lookup: dict, 
            triplets: Union[list, None] = None
        ):
        self.user_embeddings = user_embeddings  # [num_users, dim]
        self.user_lookup = user_lookup # dict: user_id -> row_id in user_embeddings
        self.item_embeddings = item_embeddings  # [num_items, dim]
        self.item_lookup = item_lookup # dict: item_id -> row_id in item_embeddings
        self.triplets = triplets if triplets is not None else [] # list of (user_id, pos_item_id, neg_item_id)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        user_id, pos_item_id, neg_item_id = self.triplets[idx]
        
        # Convert the user and item ids to their corresponding embeddings index
        user_idx = self.user_lookup[user_id]
        pos_idx = self.item_lookup[pos_item_id]
        neg_idx = self.item_lookup[neg_item_id]
        
        # Get the embeddings
        user_emb = torch.tensor(self.user_embeddings[user_idx], dtype=torch.float32)
        pos_emb = torch.tensor(self.item_embeddings[pos_idx], dtype=torch.float32)
        neg_emb = torch.tensor(self.item_embeddings[neg_idx], dtype=torch.float32)

        return user_emb, pos_emb, neg_emb

class PointwiseRankingDataset(Dataset):
    def __init__(
        self,
        user_embeddings: np.ndarray,
        user_lookup: dict,
        item_embeddings: np.ndarray,
        item_lookup: dict,
        pairs: list[tuple[int, int, int]]  # (user_id, item_id, label)
    ):
        self.user_embeddings = user_embeddings
        self.user_lookup = user_lookup
        self.item_embeddings = item_embeddings
        self.item_lookup = item_lookup
        # self.pairs = pairs  # list of (user_id, item_id, label)

        # group by user_id
        self.user_to_items = defaultdict(list)
        for user_id, item_id, label in pairs:
            self.user_to_items[user_id].append((item_id, label))
        
        # Store unique user list for indexing
        self.user_ids = list(self.user_to_items.keys())

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        """
        should return:
        - user_emb: (K, D) # repeated user embedding for K candidates
        - item_emb: (K, D) # candidate item embeddings
        - label: (K,) # relevance labels for each candidate
        where K is the number of candidates per user (e.g., 1 for pointwise, >1 for listwise)
        """
        user_id = self.user_ids[idx]
        item_label_list = self.user_to_items[user_id]

        user_idx = self.user_lookup[user_id]
        user_emb = torch.tensor(self.user_embeddings[user_idx], dtype=torch.float32)

        item_embs = []
        labels = []

        for item_id, label in item_label_list:
            item_idx = self.item_lookup[item_id]
            item_embs.append(self.item_embeddings[item_idx])
            labels.append(label)

        item_embs = torch.tensor(np.stack(item_embs), dtype=torch.float32)  # (K, D)
        labels = torch.tensor(labels, dtype=torch.float32)                  # (K,)

        # Broadcast user embedding to match item list
        user_embs = user_emb.unsqueeze(0).expand(len(item_label_list), -1)  # (K, D)

        return user_embs, item_embs, labels

class ContextualPairwiseDataset(Dataset):
    def __init__(self, user_histories, item_embeddings, item_lookup, triplets):
        self.user_histories = user_histories
        self.item_embeddings = item_embeddings
        self.item_lookup = item_lookup
        self.triplets = triplets  # (user_id, pos_item_id, neg_item_id)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        user_id, pos_item_id, neg_item_id = self.triplets[idx]
        history_ids = self.user_histories[user_id]  # List of item_ids in user's history

        pos_emb = torch.tensor(self.item_embeddings[self.item_lookup[pos_item_id]], dtype=torch.float32)
        neg_emb = torch.tensor(self.item_embeddings[self.item_lookup[neg_item_id]], dtype=torch.float32)

        # Convert history item_ids to embeddings, shape (H, D) where H is history length
        if not history_ids:
            history_emb = torch.zeros((1, self.item_embeddings.shape[1]), dtype=torch.float32)
        else:
            history_emb = torch.stack([
                torch.tensor(self.item_embeddings[self.item_lookup[h]], dtype=torch.float32)
                for h in history_ids
            ])

        return history_emb, pos_emb, neg_emb
