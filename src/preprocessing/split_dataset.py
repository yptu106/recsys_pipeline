import argparse
import pathlib
import numpy as np
import pandas as pd

np.random.seed(42)  # Set seed for reproducibility

from src.config import USER_ID_COL, STREAMER_ID_COL

def _uniform_random_lko_split(
    df: pd.DataFrame,
    neg_per_pos: int = 100,
    val_k: int = 1, 
    test_k: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split user-item interactions into train, validation, and test sets using a uniform random leave-k-out strategy"""
    
    # user_id -> set(unique streamer_ids)
    user_streamer_df = df.groupby(USER_ID_COL).agg({
        STREAMER_ID_COL: set
    }).rename(columns={
        STREAMER_ID_COL: "unique_streamers"
    })

    streamer_set = set(df[STREAMER_ID_COL].unique())

    train_samples, val_samples, test_samples = [], [], []
    user_streamer_dict = user_streamer_df["unique_streamers"].to_dict()

    for user_id, pos_streamers in user_streamer_dict.items():
        pos_streamers = list(pos_streamers)
        n_pos = len(pos_streamers)

        np.random.shuffle(pos_streamers)

        # Assign `test_k` positives to test
        # - Only if n_pos ≥ test_k + 1 (i.e., must leave at least one for train)
        # - If not enough, assign nothing to test or val — keep all in train

        # Assign `val_k` positives to validation
        # - Only if remaining after test ≥ val_k + 1 (again, leave at least one for train)
        # - If not enough, skip val

        # All other interactions go to train

        # default all to train
        test_pos, val_pos, train_pos = [], [], pos_streamers
        # check if we can assign test_k while keeping >= 1 for train
        if n_pos >= test_k + 1:
            test_pos = pos_streamers[:test_k]
            remaining = pos_streamers[test_k:]

            # check if we can assign full val_k and still keep >= 1 for train
            if len(remaining) >= val_k + 1:
                val_pos = remaining[:val_k]
                train_pos = remaining[val_k:]
            else:
                val_pos = []
                train_pos = remaining  # all remaining go to train

        # Train positives
        train_samples.extend([
            {USER_ID_COL: user_id, STREAMER_ID_COL: p, "label": 1} for p in train_pos
        ])

        # Validation positives and negatives
        neg_candidates_val = list(streamer_set - set(pos_streamers))
        if len(neg_candidates_val) < neg_per_pos:
            raise ValueError(f"Not enough negative candidates for user {user_id} (val). Required: {neg_per_pos}, Available: {len(neg_candidates_val)}")
        for p in val_pos:
            val_samples.append({USER_ID_COL: user_id, STREAMER_ID_COL: p, "label": 1})
            negs = np.random.choice(neg_candidates_val, neg_per_pos, replace=False)
            val_samples.extend([
                {USER_ID_COL: user_id, STREAMER_ID_COL: neg, "label": 0} for neg in negs
            ])

        # Test positives and negatives
        neg_candidates_test = list(streamer_set - set(pos_streamers))
        if len(neg_candidates_test) < neg_per_pos:
            raise ValueError(f"Not enough negative candidates for user {user_id} (test). Required: {neg_per_pos}, Available: {len(neg_candidates_test)}")
        for p in test_pos:
            test_samples.append({USER_ID_COL: user_id, STREAMER_ID_COL: p, "label": 1})
            negs = np.random.choice(neg_candidates_test, neg_per_pos, replace=False)
            test_samples.extend([
                {USER_ID_COL: user_id, STREAMER_ID_COL: neg, "label": 0} for neg in negs
            ])
        
    train_df = pd.DataFrame(train_samples)
    val_df = pd.DataFrame(val_samples)
    test_df = pd.DataFrame(test_samples)

    return train_df, val_df, test_df

def split_dataset(
    df: pd.DataFrame, 
    strategy: str = "uniform_random_lko",
    neg_per_pos: int = 100,
    val_k: int = 1,
    test_k: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split user-item interactions into train, validation, and test sets.

    df columns required: 
        USER_ID_COL: int64
        STREAMER_ID_COL: int64
    All rows are assumed to be positive interactions. 
    """

    if strategy == "uniform_random_lko":
        train_df, val_df, test_df = _uniform_random_lko_split(df, neg_per_pos, val_k, test_k)
        return train_df, val_df, test_df
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
def filter_rows_in_subset(df_all: pd.DataFrame, df_subset: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """
    Return only the rows in df that are also present in subset, based on the given keys.
    """
    result = df_all.merge(
        df_subset[keys],
        on=keys,
        how="inner"
    )
    return result

def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactions", required=True, 
                        help="Path to the interaction parquet file")
    parser.add_argument(
        "--streamer-lookup",
        required=True,
        help="Path to the streamer embedding lookup parquet file (used to filter interactions to only streamers with embeddings)"
    )
    parser.add_argument(
        "--filter-missing-streamers",
        default=True, action="store_true",
        help="If set, filter out interactions with streamers not in the embedding lookup. If not set, keep all interactions."
    )
    parser.add_argument("--out_dir", required=True, help="Output directory for splits")
    parser.add_argument("--strategy", default="uniform_random_lko", choices=["uniform_random_lko"])
    parser.add_argument("--neg_per_pos", type=int, default=100)
    parser.add_argument("--val_k", type=int, default=1, help="Number of positive interactions to leave out for validation")
    parser.add_argument("--test_k", type=int, default=1, help="Number of positive interactions to leave out for testing")
    args = parser.parse_args()

    df = pd.read_parquet(args.interactions, columns=[USER_ID_COL, STREAMER_ID_COL])

    # filter out interactions with streamers not in the embedding lookup
    if args.filter_missing_streamers:
        streamer_ids_with_embeddings = set(pd.read_parquet(args.streamer_lookup)["streamer_id"].to_list())
        before = len(df)
        df = df[df[STREAMER_ID_COL].isin(streamer_ids_with_embeddings)]
        after = len(df)
        print(f"Filtered out {before - after} interactions with streamers not in the embedding lookup ({before} → {after})")

    train_df, val_df, test_df = split_dataset(
        df, 
        strategy=args.strategy,
        neg_per_pos=args.neg_per_pos,
        val_k=args.val_k,
        test_k=args.test_k
    )

    # filter interactions to only those in train_df
    df_filtered_train = filter_rows_in_subset(df, train_df, [USER_ID_COL, STREAMER_ID_COL])

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(out_dir / "train.parquet", compression="snappy")
    val_df.to_parquet(out_dir / "val.parquet", compression="snappy")
    test_df.to_parquet(out_dir / "test.parquet", compression="snappy")
    df_filtered_train.to_parquet(out_dir / "interactions_train.parquet", compression="snappy")
    print("✓ wrote splits to", out_dir)

if __name__ == "__main__":
    _cli()