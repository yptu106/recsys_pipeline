import numpy as np
import pandas as pd
import random

np.random.seed(42)
random.seed(42)

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
        neg_item = random.choice(neg_items) # sample one negative
        
        triplets.append((user_id, pos_item, neg_item))
    
    return triplets