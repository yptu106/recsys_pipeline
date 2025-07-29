import torch
import json
import numpy as np
import pandas as pd
import pathlib
from tqdm import tqdm
from typing import Union, List, Optional
from collections import namedtuple

from ranker.utils.io import load_embedding_and_lookup, ensure_dir_exists, get_emb_paths
from src.services.retrieval import user_embedding
from src.config import USER_ID_COL, STREAMER_ID_COL

UserEmbFallbackConfig = namedtuple(
    "UserEmbFallbackConfig",
    ["emb_path", "lookup_path", "user_log_path", "n_fallback", "max_history"]
)

class Ranker:
    def __init__(
        self,
        model: torch.nn.Module,
        item_embeddings: np.ndarray,
        item_lookup: dict[int, int],
        user_embeddings: Optional[np.ndarray] = None,
        user_lookup: Optional[dict[int, int]] = None,
        user_fallback_config: Optional[UserEmbFallbackConfig] = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = model
        self.model.to(self.device)
        self.model.eval()

        # Load embeddings
        self.item_embeddings = item_embeddings
        self.item_lookup = item_lookup

        self.user_embeddings = None
        self.user_lookup = None
        if user_embeddings is not None and user_lookup is not None:
            self.user_embeddings = user_embeddings
            self.user_lookup = user_lookup

        self.fallback_config = user_fallback_config

    def get_user_embedding(self, user_id: int) -> torch.Tensor:
        if self.user_lookup and user_id in self.user_lookup:
            vec = self.user_embeddings[self.user_lookup[user_id]]
        elif self.fallback_config:
            vec = user_embedding(
                user_id,
                emb_path=self.fallback_config.emb_path,
                lookup_path=self.fallback_config.lookup_path,
                user_log_path=self.fallback_config.user_log_path,
                n_fallback=self.fallback_config.n_fallback,
                max_history=self.fallback_config.max_history,
            )
        else:
            raise ValueError(f"User {user_id} not found and no fallback config provided.")
        return torch.tensor(vec, dtype=torch.float32)

    def get_item_embedding(self, item_id: int) -> Optional[torch.Tensor]:
        return torch.tensor(self.item_embeddings[self.item_lookup[item_id]], dtype=torch.float32) if item_id in self.item_lookup else None

    def _rank_batch(self, user_to_candidates: dict[int, list[int]], topk: int, batch_size: int = 256) -> dict[int, pd.DataFrame]:
        """
        user_cnt: U
        candidates_per_user: K
        embedding_dim: d
        """
        user_embeddings = [] # shape: [U*K, d]
        item_embeddings = [] # shape: [U*K, d]
        index_pairs = [] # to track which user_id corresponds to which item_id after model inference

        for user_id, candidate_ids in tqdm(user_to_candidates.items(), desc="Preparing user-item pairs"):
            user_emb = self.get_user_embedding(user_id)
            if user_emb is None:
                print(f"User {user_id} has no valid embedding. Skipping.")
                continue
            
            for sid in candidate_ids: 
                item_emb = self.get_item_embedding(sid)
                if item_emb is None:
                    print(f"Item {sid} has no valid embedding. Skipping.")
                    continue
                user_embeddings.append(user_emb) # repeat for each candidate
                item_embeddings.append(item_emb)
                index_pairs.append((user_id, sid))
        
        if not index_pairs:
            print("No valid user-item pairs to rank. Returning empty results.")
            return {}

        results = {user_id: [] for user_id in user_to_candidates.keys()}

        num_pairs = len(index_pairs)
        for i in tqdm(range(0, num_pairs, batch_size), desc="Scoring user-item pairs"):
            user_tensor_batch = torch.stack(user_embeddings[i:i+batch_size]).to(self.device) # shape: [batch_size, d]
            item_tensor_batch = torch.stack(item_embeddings[i:i+batch_size]).to(self.device) # shape: [batch_size, d]

            with torch.no_grad():
                scores = self.model(user_tensor_batch, item_tensor_batch).squeeze(-1).cpu().numpy()

            for (user_id, item_id), score in zip(index_pairs[i:i+batch_size], scores):
                results[user_id].append((item_id, score))
        
        # Convert to DataFrame and sort by score
        for user_id, items in results.items():
            df = pd.DataFrame(items, columns=[STREAMER_ID_COL, "score"])
            df = df.sort_values("score", ascending=False).head(topk)
            results[user_id] = df.reset_index(drop=True)
        
        return results

    def rank(
        self,
        user_ids: Union[int, List[int]],
        retrieval_dir: Union[str, pathlib.Path],
        topk: int = 100,
        batch_size: int = 256,
    ) -> dict[int, pd.DataFrame]:
        """
        - load all the user embeddings
        - load each user's candidate streamer IDs from their corresponding JSON
        - collect the embeddings for those candidate items
        - pass the packed `user_tensor` and `candidate_tensors` to `_rank_batch`
        """

        if isinstance(user_ids, int):
            user_ids = [user_ids]

        # TODO: stream ranking
        user_to_candidates = {}
        for i, user_id in tqdm(enumerate(user_ids), desc="Loading user retrievals", total=len(user_ids)):
            try:
                with open(pathlib.Path(retrieval_dir) / f"user_{user_id}.json", "r") as f:
                    recs = json.load(f)
                candidate_ids = pd.DataFrame(recs)[STREAMER_ID_COL].tolist()
                valid_candidates = [
                    sid for sid in candidate_ids if sid in self.item_lookup
                ]
                if not valid_candidates:
                    print(f"No valid candidates for user {user_id}. Skipping.")
                    continue
                user_to_candidates[user_id] = valid_candidates
            except FileNotFoundError:
                print(f"Retrieval file for user {user_id} not found. Skipping.")
                continue
        
        return self._rank_batch(user_to_candidates, topk=topk, batch_size=batch_size)

    def dump_results(
        self,
        results: dict[int, pd.DataFrame],
        out_dir: Union[str, pathlib.Path],
    ):
        """
        Save ranked results to the specified output directory.
        Each user's results are saved as a json file.
        """
        out_dir = pathlib.Path(out_dir)
        ensure_dir_exists(out_dir)

        for user_id, df in tqdm(results.items(), desc="Saving results"):
            out_file = out_dir / f"user_{user_id}.json"
            df[[STREAMER_ID_COL, "score"]].to_json(out_file, orient="records", force_ascii=False, indent=2)
        
