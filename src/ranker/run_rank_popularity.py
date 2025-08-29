import argparse
import pathlib
import pandas as pd
from tqdm import tqdm

from src.ranker.rank_pop import rank_user
from src.config import USER_ID_COL, STREAMER_ID_COL

DEFAULT_K = 100  # Default number of candidates to retrieve

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", required=True, help="Parquet/CSV file with test interactions")
    parser.add_argument("--pop-streamers-path", required=True, help="Path to popular streamers CSV or parquet file")
    parser.add_argument("--topk", type=int, default=DEFAULT_K, help="Number of top streamers to retrieve")
    parser.add_argument("--out-dir", default="results/ranked/popularity", help="Output directory for ranked results")
    args = parser.parse_args()

    # Setup
    output_dir = pathlib.Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_parquet(args.test_path) if args.test_path.endswith(".parquet") else pd.read_csv(args.test_path)
    user_ids = test_df[USER_ID_COL].unique()

    # Load popular streamers
    pop_streamers_df = pd.read_parquet(args.pop_streamers_path) if args.pop_streamers_path.endswith(".parquet") else pd.read_csv(args.pop_streamers_path)
    pop_streamers = pop_streamers_df["streamer_id"].tolist()

    print(f"› Loaded {len(pop_streamers)} popular streamers")

    for user_id in tqdm(user_ids, desc="Ranking users by popularity"):
        topk_df = rank_user(
            user_id=user_id,
            pop_streamers_list=pop_streamers,
            topk=args.topk
        )

        # Save results to output directory
        output_file = output_dir / f"user_{user_id}.json"
        topk_df[[STREAMER_ID_COL, "rank"]].to_json(output_file, orient="records", force_ascii=False, indent=2)

if __name__ == "__main__":
    main()