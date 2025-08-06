import pathlib
import pandas as pd
import numpy as np
import numpy as np
import random

np.random.seed(42)
random.seed(42)

from src.config import USER_ID_COL, STREAMER_ID_COL

filter_type = "donate"
interactions_file = f"data/splits/donate/w_ts_heavy/train.parquet"
item_features_file = "features/ranker/lightgbm/item.parquet"
user_features_file = "features/ranker/lightgbm/user.parquet"
out_dir = "data/livestream_w_ts_heavy"
dataset_name = "livestream_w_ts_heavy"

def process_interactions(df: pd.DataFrame, out_dir: str) -> None:
    """
    Process interactions DataFrame to ensure it has the correct columns and types.
    """

    # # Convert event_time to UNIX timestamp (float seconds)
    # if "event_time" in df.columns:
    #     print("process with ts!")
    #     df["timestamp"] = pd.to_datetime(df["event_time"]).astype(int) / 1e9
    #     df = df.drop(columns=["event_time"])

    # Filter out rows with missing user or streamer IDs
    df = df.dropna(subset=[USER_ID_COL, STREAMER_ID_COL])

    # Save as tab-separated file with correct RecBole header
    out_path = f"{out_dir}/{dataset_name}.inter"
    if "timestamp" in df.columns:
        print("process with ts!")
        df[[USER_ID_COL, STREAMER_ID_COL, "timestamp"]].to_csv(
            out_path,
            sep="\t",
            index=False,
            header=[f"{USER_ID_COL}:token", f"{STREAMER_ID_COL}:token", "timestamp:float"]
        )
    else:
        df[[USER_ID_COL, STREAMER_ID_COL]].to_csv(
            out_path,
            sep="\t",
            index=False,
            header=[f"{USER_ID_COL}:token", f"{STREAMER_ID_COL}:token"]
        )

def process_item_features(df: pd.DataFrame, out_dir: str) -> None:
    """
    Process item features DataFrame and save in RecBole .item atomic format.
    Assumes all columns except item_id are continuous float fields.
    """
    df = df.reset_index()

    # Create header: assume item_id is token, rest are float
    header = [f"{STREAMER_ID_COL}:token"] + [f"{col}:float" for col in df.columns if col != STREAMER_ID_COL]

    # Save as tab-separated file
    out_path = f"{out_dir}/{dataset_name}.item"
    df.to_csv(
        out_path,
        sep="\t",
        index=False,
        header=header
    )
    

def process_user_features(df: pd.DataFrame, out_dir: str) -> None:
    """
    Process user features DataFrame and save in RecBole .user atomic format.
    Assumes all columns except user_id are continuous float fields.
    """
    df = df.reset_index()

    # Create header: assume user_id is token, rest are float
    header = [f"{USER_ID_COL}:token"] + [f"{col}:float" for col in df.columns if col != USER_ID_COL]

    # Save as tab-separated file
    out_path = f"{out_dir}/{dataset_name}.user"
    df.to_csv(
        out_path,
        sep="\t",
        index=False,
        header=header
    )

def build_pointwise(df):
    """
    Convert evaluation data into a DataFrame with columns [USER_ID_COL, STREAMER_ID_COL, "label"].
    
    Args:
        eval_data (list): List of tuples (user_id, streamer_id, label).
    
    Returns:
        pd.DataFrame: DataFrame with columns [USER_ID_COL, STREAMER_ID_COL, "label"].
    """
    df["label"] = 1.0  # Default label for pointwise data

    # Generate one negative sample per positive
    all_items = df["streamer_id"].unique()
    user_to_items = df.groupby("user_id")["streamer_id"].apply(set)

    neg_samples = []
    for user, pos_items in user_to_items.items():
        while True:
            neg_item = np.random.choice(all_items)
            if neg_item not in pos_items:
                neg_samples.append([user, neg_item, 0.0])
                break

    neg_df = pd.DataFrame(neg_samples, columns=["user_id", "streamer_id", "label"])
    df = pd.concat([df, neg_df], ignore_index=True)

    out_path = f"{out_dir}/{dataset_name}_pointwise.inter"
    df.to_csv(
        out_path,
        sep="\t",
        index=False,
        header=[f"{USER_ID_COL}:token", f"{STREAMER_ID_COL}:token", "label:float"]
    )

def main():
    # ensure output directory exists
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Load interactions
    df_interactions = pd.read_parquet(interactions_file)
    process_interactions(df_interactions, out_dir)

    # Load item features
    df_item_feat = pd.read_parquet(item_features_file)
    process_item_features(df_item_feat, out_dir)

    # Load user features
    df_user_feat = pd.read_parquet(user_features_file)
    process_user_features(df_user_feat, out_dir)

    # # random sample for labeling
    # df_interactions = pd.read_parquet(interactions_file)
    # build_pointwise(df_interactions)

if __name__ == "__main__":
    main()