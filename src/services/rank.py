"""
rank.py

Lightweight ranking module for user-to-streamer retrieval using LightGBM.

Usage:
python -m src.services.rank \
    --user-id 12345 \
    --retrieval-path data/retrieval_results/user_12345.json \
    --feature-dir features/ranker \
    --model-dir ranker/models \
    --topk 100 \
    --out-dir results/ranked
"""

import pandas as pd
import numpy as np
import argparse
import pathlib
import json
from lightgbm import Booster
from joblib import load

from src.config import USER_ID_COL, STREAMER_ID_COL

DEFAULT_K = 100 # Default number of candidates to retrieve

FEATURE_COLS = [
    'u_watch_tot', 'u_watch_cnt', 'u_gift_cnt', 'u_gift_amt', 'u_follow_cnt',
    'i_watch_tot', 'i_watch_cnt', 'i_unique_user', 'i_live_cnt',
    'i_followers', 'i_gift_amt', 'i_watch_avg', 'i_pop_z'
]

def log(message: str, quiet: bool):
    if not quiet:
        print(message)

def load_features(user_id: int, candidates: pd.DataFrame, user_feats: pd.DataFrame, item_feats: pd.DataFrame) -> pd.DataFrame:
    candidates = candidates.copy()
    candidates[USER_ID_COL] = user_id

    user_feats = user_feats.reset_index()
    item_feats = item_feats.reset_index()

    # Join user/item features
    df = candidates.merge(user_feats, on="user_id", how="left")
    df = df.merge(item_feats, on="streamer_id", how="left")
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    return df

def rank_user(
    user_id: int, 
    retrieval_path: str,
    user_feats: pd.DataFrame,
    item_feats: pd.DataFrame,
    model: Booster,
    scaler,
    topk: int = DEFAULT_K,
    out_dir: str = "data/ranked_results",
    quiet: bool = False,
    print_topk: bool = False
) -> None:
    log(f"› Loading retrieved candidates from {retrieval_path}", quiet)
    with open(retrieval_path, "r", encoding="utf-8") as f:
        recs = json.load(f)

    candidates_df = pd.DataFrame(recs)
    assert "streamer_id" in candidates_df.columns

    log("› Building feature matrix...", quiet)
    df = load_features(user_id, candidates_df, user_feats, item_feats)
    X = scaler.transform(df[FEATURE_COLS])
    scores = model.predict(X)

    df["score"] = scores
    topk_df = df.sort_values("score", ascending=False).head(topk)

    if print_topk:
        print(f"🎯 Top-{topk} ranked streamers for user {user_id}:")
        print(topk_df[["streamer_id", "score"]].to_string(index=False))

    output_dir = pathlib.Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"user_{user_id}.json"
    topk_df[["streamer_id", "score"]].to_json(output_file, orient="records", force_ascii=False, indent=2)
    log(f"✅ Saved ranked result to: {output_file}", quiet)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--retrieval-path", type=str, required=True, help="Path to JSON with top-k recall results")
    parser.add_argument("--feature-dir", type=str, default="features/ranker")
    parser.add_argument("--model-dir", type=str, default="ranker/models")
    parser.add_argument("--topk", type=int, default=DEFAULT_K)
    parser.add_argument("--out-dir", type=str, default="data/ranked_results")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--print-topk", action="store_true", help="Print top-k results to console")
    args = parser.parse_args()

    # Load features and model once
    log("› Loading user/item features and model...", args.quiet)
    user_feats = pd.read_parquet(f"{args.feature_dir}/user.parquet")
    item_feats = pd.read_parquet(f"{args.feature_dir}/item.parquet")
    model = Booster(model_file=f"{args.model_dir}/lgbm.txt")
    scaler = load(f"{args.model_dir}/scaler.joblib")

    rank_user(
        user_id=args.user_id,
        retrieval_path=args.retrieval_path,
        user_feats=user_feats,
        item_feats=item_feats,
        model=model,
        scaler=scaler,
        topk=args.topk,
        out_dir=args.out_dir,
        quiet=args.quiet,
        print_topk=args.print_topk
    )

if __name__ == "__main__":
    main()
