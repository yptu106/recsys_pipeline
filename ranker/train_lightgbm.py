import pandas as pd
import numpy as np
import argparse
import pathlib
from lightgbm import LGBMRanker, early_stopping
from sklearn.preprocessing import StandardScaler
from joblib import dump

from src.config import USER_ID_COL, STREAMER_ID_COL

N_NEGATIVES = 100

FEATURE_COLS = [
    # user features
    'u_watch_tot', 'u_watch_cnt', 'u_gift_cnt', 'u_gift_amt', 'u_follow_cnt',
    # item features
    'i_watch_tot', 'i_watch_cnt', 'i_unique_user', 'i_live_cnt',
    'i_followers', 'i_gift_amt', 'i_watch_avg', 'i_pop_z'
]

def sample_negatives(train_df, all_streamers, n_neg=N_NEGATIVES):
    neg_samples = []
    for user_id, group in train_df.groupby(USER_ID_COL):
        pos_streamers = set(group[STREAMER_ID_COL])
        neg_candidates = list(all_streamers - pos_streamers)
        sampled_negs = np.random.choice(neg_candidates, min(n_neg, len(neg_candidates)), replace=False)
        for neg in sampled_negs:
            neg_samples.append({
                USER_ID_COL: user_id, 
                STREAMER_ID_COL: neg, 
                "label": 0
            })
    return pd.DataFrame(neg_samples)

def load_and_merge_features(df, user_df, item_df):
    df = df.merge(user_df, on=USER_ID_COL, how="left")
    df = df.merge(item_df, on=STREAMER_ID_COL, how="left")
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=str, required=True, help="Directory containing train/val/test splits")
    parser.add_argument("--feature-dir", type=str, default="features/ranker")
    parser.add_argument("--output-dir", type=str, default="ranker/models")
    args = parser.parse_args()

    split_dir = pathlib.Path(args.split_dir)
    feature_dir = pathlib.Path(args.feature_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data splits
    # - train_df: all positive interactions
    # - val_df: 1 posiitve and N negatives per user
    # - test_df: 1 positive and N negatives per user
    print("› Loading train/val/test splits...")
    train_df = pd.read_parquet(split_dir / f"train.parquet")
    val_df = pd.read_parquet(split_dir / f"val.parquet")
    test_df = pd.read_parquet(split_dir / f"test.parquet")

    # Load prebuilt features
    print("› Loading user/item features...")
    user_df = pd.read_parquet(feature_dir / f"user.parquet")
    item_df = pd.read_parquet(feature_dir / f"item.parquet")

    # Sample negatives for training 
    print("› Sampling negative interactions...")
    streamers_in_train = set(train_df["streamer_id"].unique())
    neg_df = sample_negatives(train_df, streamers_in_train)
    train_df = pd.concat([train_df, neg_df], ignore_index=True)

    # Join features
    print("› Merging features...")
    train_df = load_and_merge_features(train_df, user_df, item_df)
    val_df = load_and_merge_features(val_df, user_df, item_df)
    test_df = load_and_merge_features(test_df, user_df, item_df)

    # Handle cold-start cases
    val_df.fillna(val_df.mean(numeric_only=True), inplace=True)
    test_df.fillna(test_df.mean(numeric_only=True), inplace=True)

    # Sort by user for group-based ranking (so group order = row order)
    train_df.sort_values("user_id", inplace=True)
    val_df.sort_values("user_id", inplace=True)

    # Group sizes
    train_group = train_df.groupby("user_id").size().to_numpy()
    val_group = val_df.groupby("user_id").size().to_numpy()

    # Extract features
    scaler = StandardScaler().fit(train_df[FEATURE_COLS])

    X_train = scaler.transform(train_df[FEATURE_COLS])
    X_val = scaler.transform(val_df[FEATURE_COLS])

    y_train = train_df["label"].values
    y_val = val_df["label"].values

    # Train ranker
    print("› Training LightGBM ranker...")
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[10, 20, 50],
        num_leaves=63,
        n_estimators=100,
        learning_rate=0.05,
    )

    callbacks = [early_stopping(stopping_rounds=30, verbose=True)]

    model.fit(
        X_train, y_train,
        group=train_group,
        eval_set=[(X_val, y_val)],
        eval_group=[val_group],
        callbacks=callbacks,
    )

    # Save model and scaler
    model_path = output_dir / f"lgbm.txt"
    scaler_path = output_dir / f"scaler.joblib"
    model.booster_.save_model(str(model_path))
    dump(scaler, scaler_path)

    print(f"✅ Model saved to {model_path}")
    print(f"✅ Scaler saved to {scaler_path}")

if __name__ == "__main__":
    main()
