import argparse
from pathlib import Path
import pandas as pd
from reranker.reranker import Reranker
from reranker.base import BaseRerankStrategy, NoOpStrategy
from reranker.strategies.mmr import MMRStrategy
from reranker.utils import EmbeddingStore, load_embedding_and_lookup

def main():
    parser = argparse.ArgumentParser(description="Run Reranker with MMR Strategy")
    parser.add_argument("--test-path", type=Path, required=True, help="Path to the test dataset")
    parser.add_argument("--strategy", type=str, choices=["mmr", "noop"], default="mmr", help="Rerank strategy to use")
    parser.add_argument("--lambda_", type=float, default=0.7, help="Lambda parameter for MMR strategy")
    parser.add_argument("--embedding-dir", type=Path, default=None, help="Directory containing embeddings and lookup")
    parser.add_argument("--ranked-dir", type=Path, required=True, help="Directory containing ranked JSON files")
    parser.add_argument("--topk", type=int, default=50, help="Number of top items to return")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for results")
    args = parser.parse_args()

    test_df = pd.read_parquet(args.test_path)
    user_ids = test_df["user_id"].unique()

    if args.strategy == "mmr":
        if args.embedding_dir is None:
            raise ValueError("Embedding directory must be specified for MMR strategy.")
        
        embeddings, lookup = load_embedding_and_lookup(args.embedding_dir, id_col="streamer_id")
        embedding_store = EmbeddingStore(embeddings, lookup)
        strategy = MMRStrategy(embedding_store=embedding_store, lambda_=args.lambda_, top_n=args.topk)
    else:
        strategy = NoOpStrategy()
    
    reranker = Reranker(strategy=strategy)
    results = reranker.rerank(user_ids=user_ids, ranked_dir=args.ranked_dir, topk=args.topk)

    reranker.dump_results(results, out_dir=args.out_dir)

if __name__ == "__main__":
    main()