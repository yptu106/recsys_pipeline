import argparse
from pathlib import Path
import pandas as pd
from reranker.lgbm_ranker import LGBMRanker

def main():
    parser = argparse.ArgumentParser(description="Run LGBM Ranker")
    parser.add_argument("--test-path", type=Path, required=True, help="Path to the test dataset")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to the LGBM model file")
    parser.add_argument("--stack_feature_path", type=Path, required=True, help="Path to the stack features parquet file")
    parser.add_argument("--retrieval_dir", type=Path, required=True, help="Directory containing retrieval JSON files")
    parser.add_argument("--topk", type=int, default=100, help="Number of top items to return")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for results")
    args = parser.parse_args()

    test_df = pd.read_parquet(args.test_path)
    user_ids = test_df["user_id"].unique()

    ranker = LGBMRanker(model_path=args.model_path, stack_feature_path=args.stack_feature_path)
    results = ranker.rank(user_ids=user_ids, retrieval_dir=args.retrieval_dir, topk=args.topk)
    ranker.dump_results(results, out_dir=args.out_dir)

if __name__ == "__main__":
    main()
