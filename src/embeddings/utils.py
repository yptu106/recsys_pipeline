"""
Utility functions for Tag2Vec training and streamer–tag graph construction.

All helpers here are **framework‑agnostic** and pure‑Python so they can be
re‑used by notebooks, batch scripts, or future Spark UDFs.
"""
from __future__ import annotations

import random
from typing import List

import networkx as nx
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_streamer_df(csv_path: str, tag_col: str = "tags") -> pd.DataFrame:
    """Load the streamer CSV and ensure *tags* is a Python list.

    Parameters
    ----------
    csv_path : str
        Path to the original CSV.
    tag_col : str, optional
        Column name that contains the list of tags, by default "tags".
    """
    def _conv(x):
        # Expect a string representation of a list, e.g. "['k-pop', 'dance']"
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return []
        try:
            return eval(x)
        except Exception:
            return []

    return pd.read_csv(csv_path, converters={tag_col: _conv})


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_bipartite_graph(
    df: pd.DataFrame,
    tag_col: str = "tags",
    id_col: str = "pfid",
    streamer_prefix: str = "s_",
) -> nx.Graph:
    """Return an undirected bipartite graph (streamers ↔ tags)."""
    G = nx.Graph()
    for _, row in df.iterrows():
        s_node = f"{streamer_prefix}{row[id_col]}"
        G.add_node(s_node, bipartite="streamer")
        for tag in row[tag_col]:
            G.add_node(ｆｆtag, bipartite="tag")
            G.add_edge(s_node, tag, weight=1.0)
    return G


# ---------------------------------------------------------------------------
# Random‑walk sampler (Node2Vec style)
# ---------------------------------------------------------------------------

def generate_random_walks(
    G: nx.Graph,
    num_walks: int = 10,
    walk_length: int = 16,
    seed: int | None = None,
) -> List[List[str]]:
    """Generate *num_walks* random walks of length *walk_length* per node."""
    rnd = random.Random(seed)
    nodes = list(G.nodes())
    walks: List[List[str]] = []

    for _ in range(num_walks):
        rnd.shuffle(nodes)
        for node in nodes:
            walk = [node]
            while len(walk) < walk_length:
                neighbours = list(G[walk[-1]])
                if not neighbours:
                    break
                walk.append(rnd.choice(neighbours))
            walks.append(walk)
    return walks


# ---------------------------------------------------------------------------
# Training & export helpers
# ---------------------------------------------------------------------------

def train_tag2vec_model(
    walks: List[List[str]],
    vector_size: int = 64,
    window: int = 5,
    negative: int = 15,
    epochs: int = 5,
    workers: int = 4,
) -> Word2Vec:
    """Train a skip‑gram Word2Vec model on the sampled walks."""
    model = Word2Vec(
        sentences=walks,
        vector_size=vector_size,
        window=window,
        sg=1,  # skip‑gram
        negative=negative,
        min_count=1,
        workers=workers,
        epochs=epochs,
    )
    return model


def export_tag_vectors(
    model: Word2Vec,
    output_path: str,
    streamer_prefix: str = "s_",
) -> None:
    """Strip streamer nodes and save **only tag vectors** to KeyedVectors."""
    tag_keys = [k for k in model.wv.key_to_index if not k.startswith(streamer_prefix)]
    tag_vecs = model.wv.vectors[[model.wv.get_index(k) for k in tag_keys]]

    kv = KeyedVectors(vector_size=model.vector_size)
    kv.add_vectors(tag_keys, tag_vecs)
    kv.save(output_path)


__all__ = [
    "load_streamer_df",
    "build_bipartite_graph",
    "generate_random_walks",
    "train_tag2vec_model",
    "export_tag_vectors",
]
