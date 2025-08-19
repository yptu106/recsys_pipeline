import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from collections import defaultdict

from src.config import USER_ID_COL, STREAMER_ID_COL

random.seed(42)
np.random.seed(42)

def sample_triplets(train_df, all_streamers):
    """Assumes train_df only contains positive interactions."""
    triplets = [] # (user_id, pos_item_id, neg_item_id)
    for user_id, group in tqdm(train_df.groupby(USER_ID_COL), desc="Sampling triplets"):
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

def collect_retrieval_results(retrieval_dir):
    """
    Collects retrieval results from JSON files in the specified directory.
    
    Args:
        retrieval_dir (str): Path to the directory containing retrieval result files.
        
    Returns:
        dict: user_id to set of item_ids mapping.
    """
    retrieval_results = dict()
    
    for file in tqdm(os.listdir(retrieval_dir), desc="Loading retrieval results"):
        if file.endswith('.json'):
            user_id = int(file.split('_')[1].replace('.json', ''))  # Assuming the file name format is 'user_<user_id>.json'
            retrieval_results[user_id] = set()

            with open(os.path.join(retrieval_dir, file), 'r') as f:
                data = json.load(f)
                for item in data:
                    retrieval_results[user_id].add(item[STREAMER_ID_COL])

    return retrieval_results

def build_user_log(user_log_path, user_id_col=USER_ID_COL, item_id_col=STREAMER_ID_COL, max_history_len=100):
    """
    Builds a user log dictionary from the user log file.

    Args:
        user_log_path (str): Path to the user log file (Parquet or CSV format).

    Returns:
        dict: user_id to set of item_ids mapping.
    """
    user_log = defaultdict(list)

    if user_log_path.endswith(".parquet"):
        df = pd.read_parquet(user_log_path)
    else:
        df = pd.read_csv(user_log_path)

    for _, row in df.iterrows():
        user_id = row[user_id_col]
        item_id = row[item_id_col]
        user_log[user_id].append(item_id)
    
    # Truncate histories to max_history_len
    for user_id in user_log:
        if len(user_log[user_id]) > max_history_len:
            user_log[user_id] = user_log[user_id][-max_history_len:]

    return user_log

def build_ranker_training_data(train_interactions_df, retrieval_results, val_items, test_items):
    """
    Builds pointwise training data for the ranking model.

    Args:
        train_interactions_df (pd.DataFrame): Columns = ['user_id', 'item_id']
        retrieval_results (dict): {user_id: {item_id1, item_id2, ..., item_id500}}
        val_items (dict): {user_id: val_item} (held-out validation item)
        test_items (dict): {user_id: test_item} (held-out test item)

    Returns:
        pd.DataFrame: Columns = ['user_id', 'item_id', 'label']
    """

    # Build fast lookup for training positives
    train_pos_dict = (
        train_interactions_df.groupby(USER_ID_COL)[STREAMER_ID_COL]
        .apply(set)
        .to_dict()
    )

    training_data = []

    for user_id, retrieved_items in tqdm(retrieval_results.items(), desc="Building training data"):
        if user_id not in train_pos_dict:
            continue  # skip users without training data

        train_pos = train_pos_dict[user_id]
        val_item = val_items.get(user_id)
        test_item = test_items.get(user_id)

        for item_id in retrieved_items:
            # Exclude held-out val/test positives
            if item_id == val_item or item_id == test_item:
                continue

            label = 1 if item_id in train_pos else 0
            training_data.append((user_id, item_id, label))

    df = pd.DataFrame(training_data, columns=[USER_ID_COL, STREAMER_ID_COL, "label"])

    return df

def build_ranker_eval_data(eval_items, retrieval_results):
    """
    Builds pointwise eval (validation or test) data for the ranking model.

    Args:
        eval_items (dict): {user_id: held-out item_id}  (for val or test)
        retrieval_results (dict): {user_id: list of retrieved item_ids}

    Returns:
        pd.DataFrame: Columns = ['user_id', 'item_id', 'label']
    """
    eval_data = []

    for user_id, retrieved_items in tqdm(retrieval_results.items(), desc="Building eval data"):
        gt_item = eval_items.get(user_id)
        if gt_item is None:
            continue

        for item_id in retrieved_items:
            label = 1 if item_id == gt_item else 0
            eval_data.append((user_id, item_id, label))

    return pd.DataFrame(eval_data, columns=[USER_ID_COL, STREAMER_ID_COL, "label"])

def build_pairwise_triplets_from_pointwise(df, user_col=USER_ID_COL, item_col=STREAMER_ID_COL, label_col="label"):
    """
    Builds (user_id, pos_item_id, neg_item_id) triplets from a pointwise-labeled DataFrame.
    Assumes only 0/1 labels.

    Returns:
        List of (user_id, pos_item_id, neg_item_id)
    """
    triplets = []

    for user_id, group in tqdm(df.groupby(user_col), desc="Building triplets"):
        pos_items = group[group[label_col] == 1][item_col].tolist()
        neg_items = group[group[label_col] == 0][item_col].tolist()

        if not pos_items or not neg_items:
            continue

        # one triplet per positive item
        for pos_item in pos_items:
            neg_item = random.choice(neg_items)
            triplets.append((user_id, pos_item, neg_item))

    return triplets

def average_positives_per_user(df, user_col="user_id", label_col="label", pos_label=1):
    """
    Compute the average number of positive items per user.

    Args:
        df (pd.DataFrame): Pointwise labeled DataFrame with columns like ['user_id', 'item_id', 'label']
        user_col (str): Column name for user ID
        label_col (str): Column name for label (binary: 1 for positive, 0 for negative)
        pos_label (int): Value representing a positive label (default: 1)

    Returns:
        float: Average number of positive items per user (users with at least one positive)
    """
    pos_counts = (
        df[df[label_col] == pos_label]
        .groupby(user_col)
        .size()
    )

    return pos_counts.mean()


def build_pointwise_samples(df, user_col=USER_ID_COL, item_col=STREAMER_ID_COL, label_col="label"):
    """
    Convert a DataFrame with columns ['user_id', 'streamer_id', 'label']
    into a list of (user_id, streamer_id, label) tuples.

    Args:
        df (pd.DataFrame): Input DataFrame with columns 'user_id', 'streamer_id', 'label'.

    Returns:
        List[Tuple[int, int, int]]: List of (user_id, streamer_id, label) triplets.
    """
    return list(df[['user_id', 'streamer_id', 'label']].itertuples(index=False, name=None))