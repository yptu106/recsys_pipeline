import argparse
import pathlib
import numpy as np
import pandas as pd

np.random.seed(42)  # Set seed for reproducibility

from src.config import USER_ID_COL, STREAMER_ID_COL

def _uniform_random_loo_split(
    df: pd.DataFrame,
    neg_per_pos: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split user-item interactions into train, validation, and test sets using a uniform random leave-one-out strategy"""
    
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
        if n_pos < 2:
            # Exclude users with only 1 interacted streamer (though they may have multiple interactions with that streamer)
            continue

        np.random.shuffle(pos_streamers)
        if n_pos == 2:
            # 1 train, 1 test
            train_pos = pos_streamers[0]
            test_pos = pos_streamers[1]
            train_samples.append({USER_ID_COL: user_id, STREAMER_ID_COL: train_pos, "label": 1})
            # Test positive
            test_samples.append({USER_ID_COL: user_id, STREAMER_ID_COL: test_pos, "label": 1})
            # Test negatives
            neg_candidates = list(streamer_set - set(pos_streamers))
            if len(neg_candidates) < neg_per_pos:
                raise ValueError(f"Not enough negative candidates for user {user_id}. Required: {neg_per_pos}, Available: {len(neg_candidates)}")
            negs = np.random.choice(neg_candidates, neg_per_pos, replace=False)
            test_samples.extend([
                {USER_ID_COL: user_id, STREAMER_ID_COL: neg, "label": 0} for neg in negs
            ])
        else:
            # >=3: 1 train, 1 val, 1 test, rest train
            train_pos = pos_streamers[0]
            val_pos = pos_streamers[1]
            test_pos = pos_streamers[2]
            # Remaining positives (if any)
            for p in [train_pos] + pos_streamers[3:]:
                train_samples.append({USER_ID_COL: user_id, STREAMER_ID_COL: p, "label": 1})
            # Validation positive
            val_samples.append({USER_ID_COL: user_id, STREAMER_ID_COL: val_pos, "label": 1})
            # Validation negatives
            neg_candidates_val = list(streamer_set - set(pos_streamers))
            if len(neg_candidates_val) < neg_per_pos:
                raise ValueError(f"Not enough negative candidates for user {user_id} (val). Required: {neg_per_pos}, Available: {len(neg_candidates_val)}")
            negs_val = np.random.choice(neg_candidates_val, neg_per_pos, replace=False)
            val_samples.extend([
                {USER_ID_COL: user_id, STREAMER_ID_COL: neg, "label": 0} for neg in negs_val
            ])
            # Test positive
            test_samples.append({USER_ID_COL: user_id, STREAMER_ID_COL: test_pos, "label": 1})
            # Test negatives
            neg_candidates_test = list(streamer_set - set(pos_streamers))
            if len(neg_candidates_test) < neg_per_pos:
                raise ValueError(f"Not enough negative candidates for user {user_id} (test). Required: {neg_per_pos}, Available: {len(neg_candidates_test)}")
            negs_test = np.random.choice(neg_candidates_test, neg_per_pos, replace=False)
            test_samples.extend([
                {USER_ID_COL: user_id, STREAMER_ID_COL: neg, "label": 0} for neg in negs_test
            ])

    train_df = pd.DataFrame(train_samples)
    val_df = pd.DataFrame(val_samples)
    test_df = pd.DataFrame(test_samples)
    return train_df, val_df, test_df

def split_dataset(
    df: pd.DataFrame, 
    strategy: str = "uniform_random_loo",
    neg_per_pos: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split user-item interactions into train, validation, and test sets.

    df columns required: 
        USER_ID_COL: int64
        STREAMER_ID_COL: int64
    All rows are assumed to be positive interactions. 
    """

    if strategy == "uniform_random_loo":
        train_df, val_df, test_df = _uniform_random_loo_split(df, neg_per_pos)
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
    parser.add_argument("--strategy", default="uniform_random_loo", choices=["uniform_random_loo"])
    parser.add_argument("--neg_per_pos", type=int, default=100)
    args = parser.parse_args()

    df = pd.read_parquet(args.interactions, columns=[USER_ID_COL, STREAMER_ID_COL])

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