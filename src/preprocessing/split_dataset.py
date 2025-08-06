import argparse
import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

np.random.seed(42)  # Set seed for reproducibility
rng = np.random.default_rng(seed=42)

from src.config import USER_ID_COL, STREAMER_ID_COL, TIMESTAMP_COL

OPTIONAL_COLS = [TIMESTAMP_COL]

def get_extra_cols(df: pd.DataFrame) -> list[str]:
    """Return list of columns that are both optional and present."""
    return [c for c in OPTIONAL_COLS if c in df.columns]

def _sample_negatives(
    pos_df: pd.DataFrame,
    neg_per_pos: int,
    all_items: set, 
    user_to_items: dict[int, set],
) -> pd.DataFrame:
    """Return a DataFrame of negatives matching row-for-row the positives."""
    if neg_per_pos == 0 or pos_df.empty:
        return pd.DataFrame(columns=pos_df.columns)
    
    neg_rows = []
    for _, row in pos_df.iterrows():
        user_id = row[USER_ID_COL]
        timestamp = row[TIMESTAMP_COL] if TIMESTAMP_COL in row else np.nan
        candidates = list(all_items - user_to_items[user_id])

        # if the catalog is smaller than requested, fall back to replacement
        replace = len(candidates) < neg_per_pos
        neg_items = np.random.choice(candidates, neg_per_pos, replace=replace)

        neg_rows.append(
            pd.DataFrame(
                {
                    USER_ID_COL: user_id,
                    STREAMER_ID_COL: neg_items,
                    TIMESTAMP_COL: timestamp, 
                    "label": 0,  # negative samples, 
                }
            )
        )

    return pd.concat(neg_rows, ignore_index=True)

def _batch_sample_negatives(
    pos_df: pd.DataFrame, 
    neg_per_pos: int,
    all_items: set, 
    user_to_items: dict[int, set],
) -> pd.DataFrame:
    """Sample negatives for a batch of positive interactions."""
    if neg_per_pos == 0 or pos_df.empty:
        return pd.DataFrame(columns=pos_df.columns)

    # group positives by user to batch-choice
    neg_frames = []
    for user_id, group in pos_df.groupby(USER_ID_COL):
        n_pos = len(group)
        
        unseen = np.array(list(all_items - user_to_items[user_id]))
        replace = len(unseen) < neg_per_pos * n_pos
        neg_items = np.random.choice(
            unseen,
            size = n_pos * neg_per_pos,
            replace=replace
        )

        neg_df = pd.DataFrame({
            USER_ID_COL: np.repeat(user_id, neg_items.size),
            STREAMER_ID_COL: neg_items,
            TIMESTAMP_COL: np.repeat(group[TIMESTAMP_COL].values, neg_per_pos),
            "label": 0  # negative samples
        })
        neg_frames.append(neg_df)

    return pd.concat(neg_frames, ignore_index=True)

def _donation_based_split(
    df: pd.DataFrame,
    neg_per_pos: int = 100,
    val_k: int = 1, 
    test_k: int = 1,
    top_k: int = 5, 
    min_streamers_for_eval: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """doesn't aggregate multiple interactions with the same streamer"""

    assert "prod_total" in df.columns or "consume_cnt" in df.columns, \
        "DataFrame must contain either 'prod_total' or 'consume_cnt' for donation-based splitting."
    
    # user_id -> set(unique streamer_ids)
    # get the unique streamers that each user has interacted with
    user_streamer_df = df.groupby(USER_ID_COL).agg({
        STREAMER_ID_COL: set
    }).rename(columns={
        STREAMER_ID_COL: "unique_streamers"
    })
    user_streamer_dict = user_streamer_df["unique_streamers"].to_dict()

    # create a set of all unique streamer_ids in the dataset
    streamer_set = set(df[STREAMER_ID_COL].unique())

    # aggregate by user_id and streamer_id, summing the donation amounts
    agg = df.groupby([USER_ID_COL, STREAMER_ID_COL], as_index=False).agg({
        DONATE: "sum"
    })

    # compute per-user rank in place
    agg["rank"] = agg.groupby(USER_ID_COL)[DONATE].rank(method="first", ascending=False)
    print(f"Number of users with donate records: {agg[USER_ID_COL].nunique()}")

    qualified = agg[agg["rank"] <= top_k].groupby(USER_ID_COL).filter(lambda x: len(x) >= min_streamers_for_eval)
    print(f"Number of users with at least {min_streamers_for_eval} streamers in top {top_k}: {qualified[USER_ID].nunique()}")

    # sample one test positive and one val positive per qualified user
    test_pos = qualified.groupby(USER_ID_COL, group_keys=False).apply(lambda g: g.sample(n=1, random_state=rng))
    # sample one validation from the remaining streamers
    remaining = qualified[~qualified.index.isin(test_pos.index)]
    val_pos = remaining.groupby(USER_ID_COL, group_keys=False).apply(lambda g: g.sample(n=1, random_state=rng))

    test_pos["label"] = 1  # ensure test positives are labeled as 1
    val_pos["label"] = 1  # ensure val positives are labeled as 1

    test_pos = test_pos[[USER_ID_COL, STREAMER_ID_COL, "label"]]
    val_pos = val_pos[[USER_ID_COL, STREAMER_ID_COL, "label"]]

    drop_pairs = set(pd.concat([test_pos, val_pos], ignore_index=True).set_index([USER_ID, STREAMER_ID_COL]).index)

    train_df = df[~df.set_index([USER_ID_COL, STREAMER_ID_COL]).index.isin(drop_pairs)].reset_index(drop=True)
    train_df["label"] = 1  # ensure train positives are labeled as 1
    train_df = train_df[[USER_ID_COL, STREAMER_ID_COL, "label"]]

    val_df = pd.concat([val_pos, _batch_sample_negatives(
        val_pos, neg_per_pos, streamer_set, user_streamer_dict
    )])
    test_df = pd.concat([test_pos, _batch_sample_negatives(
        test_pos, neg_per_pos, streamer_set, user_streamer_dict
    )])

    return train_df, val_df, test_df


def _time_based_split(
    df: pd.DataFrame,
    neg_per_pos: int = 100,
    val_k: int = 1, 
    test_k: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-based split with per-positive negative sampling.
    - newest `test_k`   → test
    - next-newest `val_k` (only if this still leaves ≥1 for train) → val
    - remaining         → train
    For every positive in every split, add `neg_per_pos` negatives that
    the user has never interacted with, sharing the positive’s timestamp.
    """

    assert TIMESTAMP_COL in df.columns, f"DataFrame must contain a '{TIMESTAMP_COL}' column for time-based splitting."

    # retain all the positive interactions with their timestamps
    pos_df = df[[USER_ID_COL, STREAMER_ID_COL, TIMESTAMP_COL]].copy()
    pos_df.sort_values([USER_ID_COL, TIMESTAMP_COL], inplace=True) # sort by user and timestamp
    pos_df["label"] = 1  # all interactions are positive

    # user_id -> set(unique streamer_ids)
    # get the unique streamers that each user has interacted with
    user_streamer_df = pos_df.groupby(USER_ID_COL).agg({
        STREAMER_ID_COL: set
    }).rename(columns={
        STREAMER_ID_COL: "unique_streamers"
    })
    user_streamer_dict = user_streamer_df["unique_streamers"].to_dict()

    # create a set of all unique streamer_ids in the dataset
    streamer_set = set(df[STREAMER_ID_COL].unique())
    
    train_samples, val_samples, test_samples = [], [], []

    # split each user's interactions into train, validation, and test sets
    for uid, grp in tqdm(pos_df.groupby(USER_ID_COL, sort=False), desc="building positive splits", total=len(user_streamer_dict)):
        n_pos = len(grp)
        test_pos, val_pos, train_pos = [], [], grp

        # sample test positives
        if n_pos >= test_k + 1:
            test_pos = grp.iloc[-test_k:]
            remaining = grp.iloc[:-test_k]
        else: # not enough to spare >= 1 train positive
            test_pos = grp.iloc[0:0] # no test positives, keep all in train
            remaining = grp

        # sample validation positives
        if len(remaining) >= val_k + 1:
            val_pos = remaining.iloc[-val_k:]
            train_pos = remaining.iloc[:-val_k]
        else:  # not enough to spare >= 1 train positive
            val_pos = remaining.iloc[0:0]  # no validation positives, keep all in train
            train_pos = remaining

        # add train positives
        train_samples.append(train_pos)
        val_samples.append(val_pos)
        test_samples.append(test_pos)

        # add negatives
        # train_samples.append(_batch_sample_negatives(
        #     train_pos, neg_per_pos,
        #     streamer_set, user_streamer_dict
        # ))
        val_samples.append(_batch_sample_negatives(
            val_pos, neg_per_pos,
            streamer_set, user_streamer_dict
        ))
        test_samples.append(_batch_sample_negatives(
            test_pos, neg_per_pos,
            streamer_set, user_streamer_dict
        ))

    # concatenate all samples into DataFrames
    train_df = pd.concat(train_samples, ignore_index=True)
    train_df["label"] = 1  # ensure train positives are labeled as 1
    train_df = train_df[[USER_ID_COL, STREAMER_ID_COL, TIMESTAMP_COL, "label"]]
    val_df = pd.concat(val_samples, ignore_index=True)
    test_df = pd.concat(test_samples, ignore_index=True)

    return train_df, val_df, test_df

def _uniform_random_lko_split(
    df: pd.DataFrame,
    neg_per_pos: int = 100,
    val_k: int = 1, 
    test_k: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split user-item interactions into train, validation, and test sets using a uniform random leave-k-out strategy"""
    pass

# TODO: DO NOT COLLAPSE TO ONE INTERACTION PER USER-STREAMER PAIR
def _uniform_random_lko_split_deduplicate(
    df: pd.DataFrame,
    neg_per_pos: int = 100,
    val_k: int = 1, 
    test_k: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Each user's multiple interactions with a streamer are collapsed to one interaction.
    """
    
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
    elif strategy == "time_based":
        train_df, val_df, test_df = _time_based_split(df, neg_per_pos, val_k, test_k)
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

def split_repeat_novel(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the positive interactions in the test set into:
        - repeat_df: interactions where the user has seen the streamer in training
        - novel_df: interactions where the user has NOT seen the streamer in training

    Args:
        train_df (pd.DataFrame): Training interactions (must include 'user_id', 'streamer_id', 'label')
        test_df (pd.DataFrame): Testing interactions (must include 'user_id', 'streamer_id', 'label')

    Returns:
        repeat_df (pd.DataFrame): Positive test interactions where streamer was seen in training
        novel_df (pd.DataFrame): Positive test interactions where streamer was NOT seen in training
    """
    # Build user → set of streamers in train set (only positive interactions)
    user_to_train_streamers = defaultdict(set)
    for row in train_df.itertuples():
        if row.label == 1:
            user_to_train_streamers[row.user_id].add(row.streamer_id)

    # Extract positive test interactions
    test_pos_df = test_df[test_df["label"] == 1].copy()

    # Flag each row as repeat or novel
    test_pos_df["is_repeat"] = test_pos_df.apply(
        lambda row: row.streamer_id in user_to_train_streamers[row.user_id],
        axis=1
    )

    # Split
    repeat_df = test_pos_df[test_pos_df["is_repeat"] == True].drop(columns="is_repeat")
    novel_df = test_pos_df[test_pos_df["is_repeat"] == False].drop(columns="is_repeat")

    return repeat_df, novel_df

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
    parser.add_argument("--filter-too-few-streamers",
                        default=False, action="store_true",
                        help="If set, filter out users with too few streamers in the embedding lookup. If not set, keep all users.")
    parser.add_argument("--split-repeat-novel", default=False, action="store_true",
                        help="If set, split the test set into repeat and novel interactions.")
    parser.add_argument("--out_dir", required=True, help="Output directory for splits")
    parser.add_argument("--strategy", default="uniform_random_lko", choices=["uniform_random_lko", "time_based"],)
    parser.add_argument("--neg_per_pos", type=int, default=100)
    parser.add_argument("--val_k", type=int, default=1, help="Number of positive interactions to leave out for validation")
    parser.add_argument("--test_k", type=int, default=1, help="Number of positive interactions to leave out for testing")
    args = parser.parse_args()

    # df = pd.read_parquet(args.interactions, columns=[USER_ID_COL, STREAMER_ID_COL])
    df = pd.read_parquet(args.interactions)
    extra_cols = get_extra_cols(df)
    df = df[[USER_ID_COL, STREAMER_ID_COL] + extra_cols]

    # filter out interactions with streamers not in the embedding lookup
    if args.filter_missing_streamers:
        streamer_ids_with_embeddings = set(pd.read_parquet(args.streamer_lookup)["streamer_id"].to_list())
        before = len(df)
        df = df[df[STREAMER_ID_COL].isin(streamer_ids_with_embeddings)]
        after = len(df)
        print(f"Filtered out {before - after} interactions with streamers not in the embedding lookup ({before} → {after})")

    # filter out users with too few streamers in the embedding lookup
    if args.filter_too_few_streamers:
        n_interactions_required = 5
        user_streamer_counts = df.groupby("user_id")["streamer_id"].nunique().reset_index(name="num_streamers")
        before = len(df)
        df = df[df[USER_ID_COL].isin(user_streamer_counts[user_streamer_counts["num_streamers"] >= n_interactions_required][USER_ID_COL])]
        after = len(df)
        print(f"Filtered out {before - after} interactions from users with too few streamers ({before} → {after})")

    train_df, val_df, test_df = split_dataset(
        df, 
        strategy=args.strategy,
        neg_per_pos=args.neg_per_pos,
        val_k=args.val_k,
        test_k=args.test_k
    )
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

    # filter interactions to only those in train_df
    df_filtered_train = filter_rows_in_subset(df, train_df, [USER_ID_COL, STREAMER_ID_COL] + extra_cols)

    if args.split_repeat_novel:
        repeat_df, novel_df = split_repeat_novel(train_df, test_df)
        print(f"Repeat interactions in test set: {len(repeat_df)}")
        print(f"Novel interactions in test set: {len(novel_df)}")
        # save repeat and novel splits
        out_dir = pathlib.Path(args.out_dir) / "repeat_novel"
        out_dir.mkdir(parents=True, exist_ok=True)
        repeat_df.to_parquet(out_dir / "repeat.parquet", compression="snappy")
        novel_df.to_parquet(out_dir / "novel.parquet", compression="snappy")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(out_dir / "train.parquet", compression="snappy")
    val_df.to_parquet(out_dir / "val.parquet", compression="snappy")
    test_df.to_parquet(out_dir / "test.parquet", compression="snappy")
    df_filtered_train.to_parquet(out_dir / "interactions_train.parquet", compression="snappy")
    print("✓ wrote splits to", out_dir)

if __name__ == "__main__":
    _cli()