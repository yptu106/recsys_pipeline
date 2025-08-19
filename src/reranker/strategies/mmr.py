import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional
from src.reranker.base import BaseRerankStrategy
from src.representations.store import EmbeddingStore

class MMRStrategy(BaseRerankStrategy):
    """
    Maximal Marginal Relevance (MMR) reranking strategy.
    Reranks items based on diversity and relevance.
    """
    def __init__(
        self, 
        embedding_store: EmbeddingStore,
        lambda_: float = 0.7, 
        top_n: int = 50, 
        id_col: str = "streamer_id", 
        score_col: str = "score"
    ):
        if not (0.0 <= lambda_ <= 1.0):
            raise ValueError("lambda_ must be in [0.0, 1.0]")
        self.embedding_store = embedding_store
        self.lambda_ = lambda_
        self.top_n = top_n
        self.id_col = id_col
        self.score_col = score_col

        print(f"[MMRStrategy] Initialized with lambda_={self.lambda_}")
    
    def apply(self, df: pd.DataFrame, context: Optional[Dict[int, Any]] = None) -> pd.DataFrame:
        """
        `context`: placeholder for future use (not used in MMR).
        - e.g., could be used to pin certain items
        """
        if df.empty:
            return df
        
        # pull ids, scores, and embeddings
        ids = df[self.id_col].tolist() # list of ids
        scores = df[self.score_col].to_numpy(np.float32) # (K,) original scores output by the ranker
        E = self.embedding_store.get_vectors(ids) # (K, D)

        # rows L2-normalized
        EPS = 1e-8
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        E = E / np.clip(norms, EPS, None)  # normalize embeddings

        # normalize per-user relevance scores to [0, 1]
        relevance_scores = scores.copy()
        relevance_scores = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min() + EPS)

        K = len(ids)
        top_n = min(self.top_n, K)  # ensure top_n does not exceed available items

        # initialize MMR variables
        selected = []
        remaining = np.arange(K).tolist()  # indices of remaining items

        first = int(np.argmax(relevance_scores))  # index of the first item to select
        selected.append(first) 
        remaining.remove(first) # original indices of candidates that haven't been selected yet

        while len(selected) < top_n and remaining:
            # compute MMR scores for remaining items
            S = E[selected]     # (s, D)
            R = E[remaining]    # (r, D)

            # consine = dot since rows are normalized
            # - R @ S.T: entry (i, j) is the dot product between remaining item i and selected item j
            # - .max(axis=1): for each remaining item, find the max similarity with any selected item
            # - clip(-1, 1): ensure values are in [-1, 1] range
            max_sims = (R @ S.T).max(axis=1).clip(-1, 1)  # (r,)

            ## normalize per iteration
            # max_sims = (max_sims - max_sims.min()) / (max_sims.max() - max_sims.min() + EPS)
            # print(f"[MMRStrategy] Max sims for remaining items: {max_sims}")

            ## z-score per iteration
            mu = float(np.mean(max_sims))
            sd = float(np.std(max_sims))
            if sd < EPS:
                # no diversity, all remaining items are similar
                z = np.zeros_like(max_sims)
            else:
                z = (max_sims - mu) / sd # center and scale
                z = np.clip(z, -3.0, 3.0)  # clip to avoid extreme values
            
            # use z as the penalty (centered: negative = diverse, positive = redundant)
            penalty = z

            mmr = self.lambda_ * relevance_scores[remaining] - (1.0 - self.lambda_) * penalty
            # print(f"[MMRStrategy] MMR scores for remaining items: {mmr}")

            # select the item with the highest MMR score
            picked_pos = np.argmax(mmr) # index in remaining
            picked = remaining[picked_pos] # index in original df
            selected.append(picked)
            remaining.pop(picked_pos)

            # print(f"[MMRStrategy] Selected item {ids[picked]} with MMR score {mmr[picked_pos]}, and sims {max_sims[picked_pos]}")

        # print(selected)
        # print([ids[i] for i in selected])
        # return reordered df (scores unchanged; only order changed)
        out = pd.DataFrame({
            self.id_col: [ids[i] for i in selected],
            self.score_col: [float(scores[i]) for i in selected],
        })
        
        return out.reset_index(drop=True)