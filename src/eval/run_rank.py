import argparse
import pathlib
import subprocess
import pandas as pd
from tqdm import tqdm
from lightgbm import Booster
from joblib import load

from src.services.rank import rank_user

DEFAULT_K = 100 # Default number of candidates to retrieve

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", required=True, help="Parquet/CSV file with test interactions")
    parser.add_argument("--retrieval-dir", required=True, help="Directory with per-user retrieval JSONs")
    parser.add_argument("--feature-dir", default="features/ranker")
    parser.add_argument("--model-dir", default="ranker/models")
    parser.add_argument("--topk", type=int, default=DEFAULT_K)
    parser.add_argument("--out-dir", default="results/ranked")
    args = parser.parse_args()

    # Setup
    output_dir = pathlib.Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_dir = pathlib.Path(args.retrieval_dir)

    test_df = pd.read_parquet(args.test_path) if args.test_path.endswith(".parquet") else pd.read_csv(args.test_path)
    user_ids = test_df["user_id"].unique()

    # Load features and model once
    print("› Loading user/item features and model...")
    user_feats = pd.read_parquet(f"{args.feature_dir}/user.parquet")
    item_feats = pd.read_parquet(f"{args.feature_dir}/item.parquet")
    model = Booster(model_file=f"{args.model_dir}/lgbm.txt")
    scaler = load(f"{args.model_dir}/scaler.joblib")

    print(f"› Running ranking for {len(user_ids)} users...")

    for user_id in tqdm(user_ids, desc="Ranking users"):
        rank_user(
            user_id=user_id, 
            retrieval_path=str(retrieval_dir / f"user_{user_id}.json"),
            user_feats=user_feats,
            item_feats=item_feats,
            model=model,
            scaler=scaler,
            topk=args.topk,
            out_dir=args.out_dir,
            quiet=True,
            print_topk=False
        )

if __name__ == "__main__":
    main()