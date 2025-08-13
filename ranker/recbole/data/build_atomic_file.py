import pathlib
import pandas as pd
import numpy as np
import numpy as np
import random
import argparse

np.random.seed(42)
random.seed(42)

from src.config import USER_ID_COL, STREAMER_ID_COL

# filter_type = "donate"
# interactions_file = f"data/splits/donate/w_ts_heavy/train.parquet"
# item_features_file = "features/ranker/lightgbm/item.parquet"
# user_features_file = "features/ranker/lightgbm/user.parquet"
# out_dir = "data/livestream_w_ts_heavy"
# dataset_name = "livestream_w_ts_heavy"

def process_interactions(
    df: pd.DataFrame, 
    out_dir: str, 
    dataset_name: str, 
) -> None:
    """
    Process interactions DataFrame to ensure it has the correct columns and types.
    """

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

def process_item_features(df: pd.DataFrame, out_dir: str, dataset_name: str) -> None:
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
    

def process_user_features(df: pd.DataFrame, out_dir: str, dataset_name: str) -> None:
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

def build_pointwise(df, out_dir: str, dataset_name: str) -> None:
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

def write_item_with_embeddings(
    df_interactions: pd.DataFrame,
    out_dir: str,
    dataset_name: str,
    item_embeddings: np.ndarray,
    item_lookup: pd.DataFrame,          # expects a single column: 'streamer_id'
    emb_feature_name: str = "side_emb",
) -> None:
    """
    Create <dataset>.item with one row per streamer_id that appears in interactions
    and exists in the lookup. The embedding is written as a space-separated float_seq.

    Assumptions:
      - item_lookup has a single column 'streamer_id'
      - The row order of item_lookup matches the row order of item_embeddings
        (i.e., row i in item_lookup maps to item_embeddings[i])
    """
    # --- basic checks ---
    if "streamer_id" not in item_lookup.columns:
        raise ValueError("item_lookup must contain a 'streamer_id' column.")

    if not isinstance(item_embeddings, np.ndarray):
        raise TypeError("item_embeddings must be a numpy ndarray.")
    if item_embeddings.ndim != 2:
        raise ValueError(f"item_embeddings must be 2-D, got shape {item_embeddings.shape}.")
    print(f"Item embeddings shape: {item_embeddings.shape}")

    # ensure unique ids in the lookup (keep first if duplicates)
    lookup = (
        item_lookup[["streamer_id"]]
        .astype({"streamer_id": "string"})
        .drop_duplicates(ignore_index=True)
    )

    # row index == embedding row
    lk = pd.Series(
        data=np.arange(len(lookup), dtype=int),
        index=lookup["streamer_id"],
        name="idx",
    )

    # shape check
    if item_embeddings.shape[0] != len(lookup):
        raise ValueError(
            f"Embeddings rows ({item_embeddings.shape[0]}) != unique streamer_ids in lookup ({len(lookup)}). "
            "Ensure embeddings.npy and lookup rows align 1:1."
        )

    # --- Only include items that appear in interactions & exist in lookup ---
    items = pd.Index(df_interactions[STREAMER_ID_COL].astype("string").unique())
    reindexed = lk.reindex(items)

    if reindexed.isna().any():
        missing = reindexed[reindexed.isna()].index.tolist()
        print(f"[write_item_with_embeddings] Warning: {len(missing)} streamer_ids in interactions not in lookup; skipping first 5: {missing[:5]}")
    ok = reindexed.dropna().astype(int)

    # gather embeddings in the same order as 'ok'
    embs = item_embeddings[ok.to_numpy()]

    # convert to float_seq strings
    def _to_float_seq(vec: np.ndarray) -> str:
        return " ".join(f"{x:.6f}" for x in vec.tolist())

    # def _to_float_seq(vec):
    #     v = np.asarray(vec, dtype=float)
    #     s = " ".join(f"{x:.6f}" for x in v)
    #     # collapse any accidental multiple spaces
    #     return " ".join(s.split(" "))

    item_df = pd.DataFrame(
        {
            STREAMER_ID_COL: ok.index.astype("string"),
            emb_feature_name: [_to_float_seq(v) for v in embs],
        }
    )

    # write .item
    out_path = f"{out_dir}/{dataset_name}.item"
    item_df.to_csv(
        out_path,
        sep="\t",
        index=False,
        header=[f"{STREAMER_ID_COL}:token", f"{emb_feature_name}:float_seq"],
    )


def main():
    parser = argparse.ArgumentParser(description="Build atomic files for RecBole")
    parser.add_argument("--interactions_file", type=str, required=True, help="Path to interactions file")
    parser.add_argument("--item_features_file", type=str, default=None, help="Path to item features file")
    parser.add_argument("--user_features_file", type=str, default=None, help="Path to user features file")
    parser.add_argument("--add_item_emb", action='store_true', help="Whether to add item embeddings to interactions file")
    parser.add_argument("--item_emb_dir", type=str, default=None,
                        help="Directory containing embeddings.npy and lookup.parquet")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for atomic files")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    args = parser.parse_args()

    if args.add_item_emb and not args.item_emb_dir:
        parser.error("--item_emb_dir is required when --add_item_emb is set")

    # ensure output directory exists
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.out_dir}")

    # Load interactions
    df_interactions = pd.read_parquet(args.interactions_file)
    process_interactions(df_interactions, args.out_dir, args.dataset_name)

    # Load item features
    if args.item_features_file:
        df_item_feat = pd.read_parquet(args.item_features_file)
        process_item_features(df_item_feat, args.out_dir, args.dataset_name)

    # Load user features
    if args.user_features_file:
        df_user_feat = pd.read_parquet(args.user_features_file)
        process_user_features(df_user_feat, args.out_dir, args.dataset_name)
    

    # write .item with embeddings
    if args.add_item_emb:
        emb_path = pathlib.Path(args.item_emb_dir) / "embeddings.npy"
        lookup_path = pathlib.Path(args.item_emb_dir) / "lookup.parquet"
        if not emb_path.exists() or not lookup_path.exists():
            raise FileNotFoundError(f"Item embeddings or lookup file not found in {args.item_emb_dir}")
        
        print(f"Loading item embeddings from {emb_path} and {lookup_path}")
        item_embeddings = np.load(emb_path)
        item_lookup = pd.read_parquet(lookup_path)
        item_lookup.reset_index(drop=True, inplace=True)
        print(f"Item embeddings shape: {item_embeddings.shape}, Lookup shape: {item_lookup.shape}")

        write_item_with_embeddings(
            df_interactions,
            args.out_dir,
            args.dataset_name,
            item_embeddings,
            item_lookup,
        )


    # # random sample for labeling
    # df_interactions = pd.read_parquet(interactions_file)
    # build_pointwise(df_interactions)

if __name__ == "__main__":
    main()