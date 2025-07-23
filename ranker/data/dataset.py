import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Union

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

