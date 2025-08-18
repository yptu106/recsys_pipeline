from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict
import pathlib

import numpy as np
import pandas as pd

from src.config import USER_ID_COL, STREAMER_ID_COL
from src.representations.store import EmbeddingStoreConfig, EmbeddingStore

@dataclass(frozen=True)
class UserEmbedderConfig:
    user_log_path: Path | str
    user_col: str = USER_ID_COL
    item_col: str = STREAMER_ID_COL
    ts_col: Optional[str] = None       # if provided, sort by timestamp
    max_history_len: int = 50
    pooling: str = "mean"              # "mean" | "recency"
    fallback_strategy: str = "random"  # "random" | "zero"
    normalize: bool = True
    rng_seed: int = 42

class UserEmbedder:
    """
    Builds user representations from interaction logs:
      - `get_history_vectors`: returns a 2D array of shape (H, D)
        where H is the number of interactions and D is the embedding dimension.
      - `get_user_vector`: returns a 1D array of shape (D,).
        This is a pooled vector based on the user's interaction history.
    Consumes an EmbeddingStore that serves entity vectors (items, or even users later).
    """
    def __init__(self, cfg: UserEmbedderConfig, item_store: EmbeddingStore):
        self.cfg = cfg
        self.item_store = item_store
        self.user_history_lookup = self._build_user_log(self.cfg.user_log_path)
        self._rng = np.random.default_rng(cfg.rng_seed)
    
    def get_user_vector(self, user_id: int) -> np.ndarray:
        """
        Get the pooled embedding vector for a user based on their interaction history.
        Returns a 1D array of shape (D,) where D is the embedding dimension.
        """
        history = self.get_history_vectors(user_id) # (H, D) or (0, D) if no history
        if history.size == 0:
            return self._fallback_embedding(strategy=self.cfg.fallback_strategy)
        return self._pool_history(history)

    def get_history_vectors(self, user_id: int) -> np.ndarray:
        """
        Get the history vectors for a user, based on their interaction history.
        Returns a 2D array of shape (H, D) where H is the number of interactions
        and D is the embedding dimension.
        If the user has no history, returns an empty array of shape (0, D).
        """
        if user_id not in self.user_history_lookup:
            return np.empty((0, self.item_store.dim), dtype=np.float32)
    
        item_ids = self.user_history_lookup[user_id]
        vecs = self.item_store.get_vectors(item_ids) # expected shape (H, D)
        return vecs

    # ======== Private Methods ========
    def _pool_history(self, history: List[np.ndarray]) -> np.ndarray:
        """
        Pool the history vectors into a single vector.
        Supports 'mean' and 'recency' pooling strategies.
        """
        if self.cfg.pooling == "mean":
            v = np.mean(history, axis=0)
            if self.cfg.normalize:
                v /= np.linalg.norm(v) + 1e-9
            return v
        else:
            raise ValueError(f"Unknown pooling strategy: {self.cfg.pooling}")

    def _build_user_log(self, user_log_path: str) -> Dict[int, List[int]]:
        """
        Builds a user log dictionary from the user log file.

        Args:
            user_log_path (str): Path to the user log file (Parquet or CSV format).

        Returns:
            dict: user_id to set of item_ids mapping.
        """
        user_log = defaultdict(list)

        if user_log_path.endswith(".parquet"):
            df = pd.read_parquet(user_log_path)
        else:
            df = pd.read_csv(user_log_path)

        for _, row in df.iterrows():
            user_id = row[self.cfg.user_col]
            item_id = row[self.cfg.item_col]
            user_log[user_id].append(item_id)
        
        # Truncate histories to max_history
        # assumes df is ordered by recency
        for user_id in user_log:
            if len(user_log[user_id]) > self.cfg.max_history_len:
                user_log[user_id] = user_log[user_id][-self.cfg.max_history_len:]
        
        return user_log

    def _fallback_embedding(self, strategy: str = "random") -> np.ndarray:
        """Generate fallback embeddings based on the specified strategy."""
        if strategy == "random":
            sampled_ids = self._sample_random_streamer_ids()
            return np.mean(self.item_store.get_vectors(sampled_ids), axis=0)
        elif strategy == "zero":
            return np.zeros(self.item_store.dim)
        else:
            raise ValueError(f"Unknown fallback strategy: {strategy}")

    def _sample_random_streamer_ids(self, n: int = 20) -> list:
        """Randomly sample n item IDs for fallback."""
        all_ids = self.item_store.all_ids
        return self._rng.choice(all_ids, size=n, replace=False).tolist()