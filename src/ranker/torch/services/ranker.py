from __future__ import annotations
from dataclasses import dataclass
import torch
import json
import numpy as np
import pandas as pd
import pathlib
from tqdm import tqdm
from typing import Union, List, Optional, Generator, Tuple
from collections import namedtuple

from src.ranker.utils.io import load_embedding_and_lookup, ensure_dir_exists, get_emb_paths
from src.representations.store import EmbeddingStoreConfig, EmbeddingStore
from src.representations.user_embedder import UserEmbedderConfig, UserEmbedder
from src.config import USER_ID_COL, STREAMER_ID_COL


@dataclass
class RankerConfig:
    contextual: bool = False # True => model takes (B, H, D); False => model takes (B, D)
    max_history_len: int = 50 # Maximum length of user history to consider
    item_id_col: str = STREAMER_ID_COL

class Ranker:
    def __init__(
        self,
        model: torch.nn.Module,
        item_store: EmbeddingStore,
        user_embedder: UserEmbedder,
        cfg: RankerConfig,
        user_store: Optional[EmbeddingStore] = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = model
        self.model.to(self.device)
        self.model.eval()

        self.item_store = item_store
        self.user_embedder = user_embedder
        self.user_store = user_store
        self.cfg = cfg

        self._user_cache: Dict[int, torch.Tensor] = {}  # Cache for user embeddings
        self._item_cache: Dict[int, torch.Tensor] = {}  # Cache for item embeddings

    def get_user_representation(self, user_id: int) -> torch.Tensor:
        """
        (H_i, D) if contextual; (D,) if static.
        """
        if user_id in self._user_cache:
            return self._user_cache[user_id]
        
        if self.cfg.contextual:
            seq = self.user_embedder.get_history_vectors(user_id) # (H_i, D)
            if seq.shape[0] == 0:
                # fallback if user has no history
                # TODO: since the fallback is pooled from sampled items, we might need to change the cold sequence to 
                # return n_fallback items instead of a single vector
                vec = self.user_embedder.get_user_vector(user_id)  # (D,)
                out = torch.tensor(vec, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, D)
            else:
                # truncate to max history length
                if seq.shape[0] > self.cfg.max_history_len:
                    seq = seq[-self.cfg.max_history_len:] # assuming last interactions are more relevant
                out = torch.tensor(seq, dtype=torch.float32, device=self.device)
        else:
            if self.user_store:
                vec = self.user_store.get_vector(user_id)  # (D,)
            else:
                vec = self.user_embedder.get_user_vector(user_id)  # (D,)
            out = torch.tensor(vec, dtype=torch.float32, device=self.device)
        
        self._user_cache[user_id] = out
        return out

    def get_item_embedding(self, item_id: int) -> Optional[torch.Tensor]:
        """Return (D,) or None if id not found."""
        if item_id in self._item_cache:
            return self._item_cache[item_id]
        
        if not self.item_store.has(item_id):
            print(f"Item {item_id} not found in item store.")
            return None
        
        # Get the embedding vector for the item
        vec = self.item_store.get_vector(item_id) # np.ndarray(D,)
        
        t = torch.tensor(vec, device=self.device, dtype=torch.float32)
        self._item_cache[item_id] = t

        return t

    def _pad_user_histories(self, history_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Pads a list of (H_i, D) user histories to shape (B, H_max, D).
        """
        max_len = max(h.shape[0] for h in history_list)
        emb_dim = history_list[0].shape[1]

        # print(f"Padding user histories to max length {max_len} with embedding dimension {emb_dim}")

        padded = torch.zeros(len(history_list), max_len, emb_dim) # (B, H_max, D)

        # Fill padded tensor with actual history embeddings
        for i, h in enumerate(history_list):
            padded[i, :h.shape[0]] = h

        return padded

    def _build_user_item_batches(self, user_to_candidates: dict[int, list[int]]) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
        """
        Returns:
            - user_tensor: (B*K, D) or (B*K, H, D)
            - item_tensor: (B*K, D)
            - index_pairs: list of (user_id, item_id)
            - K: number of candidates per user
        """
        user_list = []
        item_list = []
        index_pairs = []

        for user_id, candidate_ids in tqdm(user_to_candidates.items(), desc="Preparing user-item pairs"):
            user_rep = self.get_user_representation(user_id)
            if user_rep is None:
                print(f"User {user_id} has no valid embedding. Skipping.")
                continue

            for sid in candidate_ids:
                item_emb = self.get_item_embedding(sid)
                if item_emb is None:
                    print(f"Item {sid} has no valid embedding. Skipping.")
                    continue

                user_list.append(user_rep)
                item_list.append(item_emb)
                index_pairs.append((user_id, sid))
        
        print(f"Total valid user: {len(user_list)}, valid item pairs: {len(index_pairs)}")
        print(f"Sampled user shape: {user_list[0].shape if user_list else 'N/A'}, item shape: {item_list[0].shape if item_list else 'N/A'}")

        # Detect if contextual input (list of [H_i, D])
        if isinstance(user_list[0], torch.Tensor) and user_list[0].dim() == 2:
            user_tensor = self._pad_user_histories(user_list)  # (B, H, D)
        else:
            user_tensor = torch.stack(user_list)  # (B, D)

        item_tensor = torch.stack(item_list)  # (B, D)

        print(f"[_build_user_item_batches]User tensor shape: {user_tensor.shape}, Item tensor shape: {item_tensor.shape}, Index pairs: {len(index_pairs)}")

        return user_tensor, item_tensor, index_pairs

    def iter_user_item_batches(
        self, 
        user_to_candidates: dict[int, list[int]],
        batch_size: int = 256
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]], None, None]:
        """
        Yields batches of (user_tensor_batch, item_tensor_batch, index_pairs) for ranking to avoid memory overflow. 
        """
        user_list, item_list, index_pairs = [], [], []

        for user_id, candidate_ids in tqdm(user_to_candidates.items(), desc="Preparing batches"):
            user_rep = self.get_user_representation(user_id)
            if user_rep is None:
                print(f"User {user_id} has no valid embedding. Skipping.")
                continue

            for sid in candidate_ids:
                item_emb = self.get_item_embedding(sid)
                if item_emb is None:
                    print(f"Item {sid} has no valid embedding. Skipping.")
                    continue

                user_list.append(user_rep)
                item_list.append(item_emb)
                index_pairs.append((user_id, sid))

                if len(user_list) >= batch_size:
                    yield self._process_batch(user_list, item_list, index_pairs)
                    user_list, item_list, index_pairs = [], [], []
        
        # final leftover batch
        if user_list:
            yield self._process_batch(user_list, item_list, index_pairs)

    def _process_batch(
        self,
        user_list: list[torch.Tensor],
        item_list: list[torch.Tensor],
        index_pairs: list[tuple[int, int]]
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
        """
        Converts one batch into padded tensors. 
        """
        if user_list[0].dim() == 2:  # contextual: list of (H_i, D)
            user_tensor = self._pad_user_histories(user_list)  # (B, H_max, D)
        else:
            user_tensor = torch.stack(user_list)  # (B, D)

        item_tensor = torch.stack(item_list)  # (B, D)
        return user_tensor, item_tensor, index_pairs

    def _rank_batch(self, user_to_candidates: dict[int, list[int]], topk: int, batch_size: int = 256) -> dict[int, pd.DataFrame]:
        """
        Ranks candidates for each user and returns top-k results.
        Supports both static and contextual user embeddings.
        """
        # isn't process at batched-level
        # user_inputs, item_inputs, index_pairs = self._build_user_item_batches(user_to_candidates)

        # if not index_pairs:
        #     print("No valid user-item pairs to rank. Returning empty results.")
        #     return {}

        results = {user_id: [] for user_id in user_to_candidates.keys()}
        total_pairs = 0
        # num_pairs = len(index_pairs)
        # print(f"Total user-item pairs to score: {num_pairs}")

        # print(f"Batch size: {batch_size}, Total batches: {num_pairs // batch_size + (1 if num_pairs % batch_size > 0 else 0)}")
        # for i in tqdm(range(0, num_pairs, batch_size), desc="Scoring user-item pairs"):
        for user_tensor_batch, item_tensor_batch, index_pairs in self.iter_user_item_batches(user_to_candidates, batch_size):
            # user_tensor_batch = torch.stack(user_inputs[i:i+batch_size]).to(self.device) # shape: [batch_size, d]
            # item_tensor_batch = torch.stack(item_inputs[i:i+batch_size]).to(self.device) # shape: [batch_size, d]
            # user_tensor_batch = user_inputs[i:i+batch_size].to(self.device)  # (B, H, D) or (B, D)
            # item_tensor_batch = item_inputs[i:i+batch_size].to(self.device)  # (B, D)
            user_tensor_batch = user_tensor_batch.to(self.device)  # (B, H, D) or (B, D)
            item_tensor_batch = item_tensor_batch.to(self.device)  # (B, D)

            # print(f"Processing batch of size {user_tensor_batch.shape[0]}: User tensor shape {user_tensor_batch.shape}, Item tensor shape {item_tensor_batch.shape}")

            with torch.no_grad():
                scores = self.model(user_tensor_batch, item_tensor_batch).squeeze(-1).cpu().numpy()

            # for (user_id, item_id), score in zip(index_pairs[i:i+batch_size], scores):
            #     results[user_id].append((item_id, score))
            for (user_id, item_id), score in zip(index_pairs, scores):
                results[user_id].append((item_id, score))
            
            total_pairs += len(index_pairs)
        
        print(f"Total user-item pairs scored: {total_pairs}")
        
        # Convert to DataFrame and sort by score
        for user_id, items in results.items():
            df = pd.DataFrame(items, columns=[self.cfg.item_id_col, "score"])
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
                candidate_ids = pd.DataFrame(recs)[self.cfg.item_id_col].tolist()
                # filter out invalid candidates
                valid_candidates = [
                    sid for sid in candidate_ids if self.item_store.has(sid)
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
            df[[self.cfg.item_id_col, "score"]].to_json(out_file, orient="records", force_ascii=False, indent=2)
        
