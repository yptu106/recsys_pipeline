import json
import pathlib
import pandas as pd
from typing import Dict, List, Optional, Union
from tqdm import tqdm

from src.reranker.base import BaseRerankStrategy

def walk_ranked_jsons(ranked_dir: pathlib.Path):
    """
    Walks through the ranked directory and yields user_id and json path. 
    """
    for json_path in ranked_dir.glob("user_*.json"):
        user_id = int(json_path.stem.split("_")[1])
        yield user_id, json_path

class Reranker:
    """
    Orchestrates:
        - loading per-user ranked JSON (expects [{id_col, score_col}, ...])
        - applying the given strategy with optional per-user `context`
        - returning/writing reranked results
    """
    def __init__(self, strategy: BaseRerankStrategy):
        self.strategy = strategy

    def rerank_user(self, user_id: int, ranked_json_path: Union[str, pathlib.Path], topk: int = 100, context: Optional[dict] = None) -> pd.DataFrame:
        with open(ranked_json_path, "r") as f:
            raw = json.load(f)
        df = pd.DataFrame(raw)
        if df.empty:
            return df

        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        # print(f"[Reranker] df before reranking for user {user_id}:")
        # print(df.head(5))
        reranked_df = self.strategy.apply(df, context=context)
        # print(f"[Reranker] df after reranking for user {user_id}:")
        # print(reranked_df.head(5))

        return reranked_df.head(topk).reset_index(drop=True)

    def rerank(
        self, 
        user_ids: Union[int, List[int]],
        ranked_dir: Union[str, pathlib.Path],
        topk: int = 100,
        context_by_user: Optional[dict] = None
    ) -> dict[int, pd.DataFrame]:
        if isinstance(user_ids, int):
            user_ids = [user_ids]

        results = {}
        ranked_dir = pathlib.Path(ranked_dir)
        for uid in tqdm(user_ids, desc="Reranking users"):
            json_path = ranked_dir / f"user_{uid}.json"
            if json_path.exists():
                ctx = context_by_user.get(uid) if context_by_user else None
                results[uid] = self.rerank_user(uid, json_path, topk=topk, context=ctx)
            else:
                print(f"[Reranker] No ranked data found for user {uid} at {json_path}.")
                results[uid] = pd.DataFrame(columns=["streamer_id", "score"])
        
        return results

    def dump_results(
        self,
        results: dict[int, pd.DataFrame],
        out_dir: Union[str, pathlib.Path],
    ):
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for user_id, df in tqdm(results.items(), desc="Dumping results"):
            out_file = out_dir / f"user_{user_id}.json"
            df[["streamer_id", "score"]].to_json(out_file, orient="records", force_ascii=False, indent=2)