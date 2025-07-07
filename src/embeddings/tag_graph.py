"""tag_graph.py
Builds (and optionally persists) the bipartite *streamer ↔ tag* graph.

Usage (module):
    python -m embeddings.tag_graph \
        --csv data/raw/streamers.csv \
        --out embeddings/debug_graph.gpickle
"""
from __future__ import annotations

import argparse
import pathlib

import networkx as nx
import pickle

from .utils import build_bipartite_graph, load_streamer_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct the streamer–tag graph")
    parser.add_argument("--csv", required=True, help="Input CSV with a 'tags' column")
    parser.add_argument(
        "--out",
        required=True,
        help="Destination file ( .gpickle | .graphml ). Parents will be created.",
    )
    args = parser.parse_args()

    df = load_streamer_df(args.csv)
    G = build_bipartite_graph(df)

    dst = pathlib.Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix == ".graphml":
        nx.write_graphml(G, dst)
    else:
        with open(dst, "wb") as f:
            pickle.dump(G, f)

    print(
        f"Graph saved → {dst}  (nodes={G.number_of_nodes():,}, edges={G.number_of_edges():,})"
    )


if __name__ == "__main__":
    main()
