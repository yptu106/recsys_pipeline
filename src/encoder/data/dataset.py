import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Union

class TripletDatasetWithUserEmbeddings(Dataset):
    def __init__(
        self, 
        user_embeddings: np.ndarray, 
        item_id_to_text: dict,
        triplets: Union[list, None] = None
    ):
        self.user_embeddings = user_embeddings # {user_id: embedding}
        self.item_id_to_text = item_id_to_text # {item_id: text}
        self.triplets = triplets if triplets is not None else [] # list of (user_id, pos_item_id, neg_item_id)

    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        user_id, pos_item_id, neg_item_id = self.triplets[idx]

        # get user embedding
        user_emb = self.user_embeddings[user_id]

        # get item texts
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
