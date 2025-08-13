import pathlib
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Union

class EmbeddingStore:
    """
    Backed by a 2D array (numpy) and an id->row lookup (dict).
    Args:
        - embeddings: np.ndarray of shape (N, D)
        - lookup: dict[id -> row_index in embeddings]
        - normalize: L2-normalize each row
    """

    def __init__(
        self, 
        embeddings: np.ndarray,
        lookup: Dict[int, int],
        normalize: bool = True
    ): 
        self.embeddings = embeddings # shape (N, D)
        self.lookup = lookup # id -> row index mapping
        if normalize:
            self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)

        self.size = embeddings.shape[0] # number of items
        self.dim = embeddings.shape[1]  # embedding dimension

    def has(self, id: int) -> bool:
        """
        Check if the store contains the given id.
        Returns:
            bool: True if id exists, False otherwise.
        """
        return id in self.lookup
    
    def get_embedding(self, id: int) -> np.ndarray:
        """
        Get the embedding for a given id.
        Returns:
            np.ndarray of shape (D,) or None if id not found.
        """
        if not self.has(id):
            return None
        row_index = self.lookup.get(id)
        return self.embeddings[row_index]

    def get_many_embeddings(self, ids: list[int]) -> np.ndarray:
        """Return (len(ids), D) matrix aligned to ids. Raises KeyError if any id missing."""
        rows = []
        for id in ids:
            if not self.has(id):
                raise KeyError(f"Embedding for id {id} not found in store.")
            rows.append(self.lookup[id])
        return self.embeddings[rows]


def get_emb_paths(dir_path: Union[str, pathlib.Path]) -> tuple[str, str]:
    dir_path = pathlib.Path(dir_path)
    emb_path = next(dir_path.glob("*.npy"))
    lookup_path = next(dir_path.glob("*.parquet"))
    return str(emb_path), str(lookup_path)

def load_embeddings(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    else:
        raise ValueError(f"Unsupported file format for embeddings: {path}")

def load_lookup_table(path: str, id_col: str) -> dict:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format for lookup table: {path}")
    
    df = df.reset_index(drop=True)

    if id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' not found in the lookup table.")
    if df[id_col].isnull().any():
        raise ValueError(f"Column '{id_col}' contains null values.")
    if df[id_col].duplicated().any():
        raise ValueError(f"Column '{id_col}' contains duplicate values.")
    
    return dict(zip(df[id_col], df.index))

def load_embedding_and_lookup(emb_dir: str, id_col: str) -> tuple[np.ndarray, dict]:
    emb_path, lookup_path = get_emb_paths(emb_dir)
    emb = load_embeddings(emb_path)
    lookup = load_lookup_table(lookup_path, id_col)
    return emb, lookup