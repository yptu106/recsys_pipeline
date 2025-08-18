from __future__ import annotations
import json
import faiss
from tqdm import tqdm
import pandas as pd
import numpy as np
import pathlib
from typing import List, Union, Dict

from src.representations.user_embedder import UserEmbedder

DEFAULT_K = 500  # Default number of candidates to retrieve

class Retriever:
    def __init__(
        self, 
        user_embedder: UserEmbedder,
        lookup_path: Union[str, pathlib.Path],
        index_path: Union[str, pathlib.Path],
        k: int = DEFAULT_K,
    ): 
        self.user_embedder = user_embedder
        self.index = self._load_faiss(index_path)
        self.lookup = self._load_lookup(lookup_path)
        self.k = k

    def retrieve_one(self, user_id: int) -> List[dict[str, object]]:

        user_vec = self.user_embedder.get_user_vector(user_id)
        
        # D: distances (cosine similarity), I: indices of the nearest neighbors
        D, I = self.index.search(user_vec.reshape(1, -1), self.k)
        
        # Prepare the results
        out = self.lookup.iloc[I[0]].copy()
        out["score"] = D[0]
        out = out.sort_values("score", ascending=False)

        return out.to_dict("records")

    def retrieve_many(
        self, 
        user_ids: Union[int, List[int]], 
        output_dir: Union[str, pathlib.Path]
    ) -> None:

        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(user_ids, int):
            user_ids = [user_ids]
        
        for user_id in tqdm(user_ids, desc="Retrieving"):
            recs = self.retrieve_one(user_id)
            self.dump_result(user_id, recs, output_dir)

    def dump_result(
        self, 
        user_id: int, 
        recs: List[dict[str, object]], 
        output_dir: Union[str, pathlib.Path]
    ) -> None:
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / f"user_{user_id}.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)

    def _load_faiss(self, index_path: str) -> faiss.Index:
        """Load the FAISS index from disk.
        Cached after first load for efficiency.
        """
        return faiss.read_index(str(index_path))
    
    def _load_lookup(self, lookup_path: str) -> pd.DataFrame:
        """Load the streamer lookup table (row-aligned with embeddings).
        Cached after first load for efficiency.
        """
        df = pd.read_parquet(lookup_path)
        df.reset_index(drop=True, inplace=True)
        return df