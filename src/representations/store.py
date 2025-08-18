# src/representations/item/loaders.py
from __future__ import annotations
import pathlib, json
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Iterable, List, Optional, Tuple

from src.representations.utils import load_embedding_and_lookup

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class EmbeddingStoreConfig:
    data_dir: Path | str # Directory containing embeddings and lookup table
    id_column: Optional[str] = None # if None, use first column
    normalize: bool = False        # Whether to normalize embeddings

class EmbeddingStore:
    """Centralized access to item embeddings and ID <-> row index mapping."""

    def __init__(self, cfg: EmbeddingStoreConfig):
        self.cfg = cfg
        # Load matrix of embeddings and ID→row mapping
        self.emb_matrix, self.id_to_row = load_embedding_and_lookup(
            emb_dir=cfg.data_dir,
            id_col=cfg.id_column
        )

        if cfg.normalize:
            self.emb_matrix = self.emb_matrix / np.linalg.norm(
                self.emb_matrix, axis=1, keepdims=True
            )
        
    @property
    def dim(self) -> int:
        """Return the dimensionality of the embeddings."""
        return int(self.emb_matrix.shape[1])
    
    @property
    def count(self) -> int:
        """Return the number of embeddings."""
        return int(self.emb_matrix.shape[0])

    @property
    def all_ids(self) -> List[int]:
        """Return all available item IDs."""
        return list(self.id_to_row.keys())
    
    def get_vector(self, item_id: int) -> np.ndarray:
        """Get the embedding vector for a single item ID."""
        if item_id not in self.id_to_row:
            raise KeyError(f"Item ID {item_id} not found in lookup table.")
        
        row_index = self.id_to_row[item_id]
        return self.emb_matrix[row_index]

    def get_vectors(self, item_ids: Iterable[int]) -> np.ndarray:
        """Get embedding vectors for a collection of item IDs."""
        indices = [
            self.id_to_row[item_id]
            for item_id in item_ids
            if item_id in self.id_to_row
        ]
        return self.emb_matrix[indices]

    