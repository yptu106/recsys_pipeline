"""train_tag2vec.py
Train Tag2Vec embeddings from a CSV of streamers.

Run:
    python -m embeddings.train_tag2vec \
        --csv data/raw/streamers.csv \
        --out embeddings/tag2vec.bin
"""
from __future__ import annotations

import argparse
import pathlib

from .utils import (
    build_bipartite_graph,
    export_tag_vectors,
    generate_random_walks,
    load_streamer_df,
    train_tag2vec_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Tag2Vec embeddings from streamer CSV.")
    parser.add_argument("--csv", required=True, help="Path to the streamer CSV file")
    parser.add_argument("--out", default="embeddings/tag2vec.bin", help="Output KeyedVectors path")
    parser.add_argument("--vector_size", type=int, default=64, help="Embedding dimension size")
    parser.add_argument("--window", type=int, default=5, help="Word2Vec context window size")
    parser.add_argument("--negative", type=int, default=15, help="Negative samples for Word2Vec")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--workers", type=int, default=4, help="Worker threads")
    parser.add_argument("--num_walks", type=int, default=10, help="Random walks per graph node")
    parser.add_argument("--walk_length", type=int, default=16, help="Length of each random walk")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load and build graph
    df = load_streamer_df(args.csv)
    G = build_bipartite_graph(df)

    # Generate random walks
    walks = generate_random_walks(
        G,
        num_walks=args.num_walks,
        walk_length=args.walk_length,
        seed=args.seed,
    )

    # Train Word2Vec
    model = train_tag2vec_model(
        walks,
        vector_size=args.vector_size,
        window=args.window,
        negative=args.negative,
        epochs=args.epochs,
        workers=args.workers,
    )

    # Export tag‑only vectors
    export_tag_vectors(model, str(out_path))

    print(
        f"✓ Trained Tag2Vec ({model.vector_size}‑d) on {len(walks):,} walks "
        f"from {G.number_of_nodes()} nodes → {out_path}"
    )


if __name__ == "__main__":
    main()
