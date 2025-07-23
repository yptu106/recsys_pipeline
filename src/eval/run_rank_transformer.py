import argparse
import pathlib
import pandas as pd
from tqdm import tqdm
from collections import namedtuple

UserEmbFallbackConfig = namedtuple(
    "UserEmbFallbackConfig",
    ["emb_path", "lookup_path", "user_log_path", "n_fallback", "max_history"]
)

from src.services.retrieval import get_emb_paths
from src.services.rank_transformer import rank_user, load_embeddings_and_lookup
from src.config import USER_ID_COL, STREAMER_ID_COL

DEFAULT_K = 100  # Default number of candidates to retrieve

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", required=True, help="Parquet/CSV file with test interactions")
    parser.add_argument("--retrieval-dir", required=True, help="Directory with per-user retrieval JSONs")
    parser.add_argument("--user-emb-dir", type=str, default=None, help="Directory containing user embeddings and lookup")
    parser.add_argument("--user-log", type=str, required=True, help="User interaction log (parquet or csv)")
    parser.add_argument("--streamer-emb-dir", required=True, help="Directory containing streamer embeddings and lookup")
    parser.add_argument("--model-path", default="Path to the trained MLP model")
    parser.add_argument("--topk", type=int, default=DEFAULT_K)
    parser.add_argument("--out-dir", default="results/ranked")
    args = parser.parse_args()

    # Setup
    output_dir = pathlib.Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_dir = pathlib.Path(args.retrieval_dir)

    test_df = pd.read_parquet(args.test_path) if args.test_path.endswith(".parquet") else pd.read_csv(args.test_path)
    user_ids = test_df[USER_ID_COL].unique()

    # Load user and item embeddings and lookups
    if args.user_emb_dir:
        user_embeddings, user_lookup = load_embeddings_and_lookup(args.user_emb_dir, "user_id")
    
    item_embeddings, item_lookup = load_embeddings_and_lookup(args.streamer_emb_dir, "streamer_id")

    # Set up user embedding fallback configuration
    item_emb_path, item_lookup_path = get_emb_paths(args.streamer_emb_dir)
    fallback_config = UserEmbFallbackConfig(
        emb_path=item_emb_path,
        lookup_path=item_lookup_path, 
        user_log_path=args.user_log,
        n_fallback=20,
        max_history=50
    )


    print(f"› Running ranking for {len(user_ids)} users...")
    for user_id in tqdm(user_ids, desc="Ranking users"):
        topk_df = rank_user(
            user_id=user_id,
            retrieval_path=str(retrieval_dir / f"user_{user_id}.json"),
            user_embeddings=user_embeddings,
            user_lookup=user_lookup,
            item_embeddings=item_embeddings,
            item_lookup=item_lookup,
            model_path=args.model_path,
            fallback_config=fallback_config,
            topk=args.topk
        )
        # Save results to output directory
        output_file = output_dir / f"user_{user_id}.json"
        topk_df[[STREAMER_ID_COL, "score"]].to_json(output_file, orient="records", force_ascii=False, indent=2)

if __name__ == "__main__":
    main()