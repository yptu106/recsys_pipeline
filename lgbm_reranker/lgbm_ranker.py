# services/lgbm_ranker.py
import json, pathlib
import lightgbm as lgb
import pandas as pd
from typing import Union
from tqdm import tqdm

# MODEL = lgb.Booster(model_file="reranker/models/lgbm.txt")
# FEATURES = ["retr_score","bpr_score","sasrec_score","trans_score","is_repeat"]
# STACK_FEATURES = pd.read_parquet("reranker/train.parquet")

# def rerank(user_id: int, candidates_df: pd.DataFrame) -> pd.DataFrame:
#     X = candidates_df[FEATURES]
#     score = MODEL.predict(X, num_iteration=MODEL.best_iteration)
#     candidates_df["final_score"] = score
#     return candidates_df.sort_values("final_score", ascending=False)


class LGBMRanker:
    def __init__(self, model_path: Union[str, pathlib.Path], stack_feature_path: Union[str, pathlib.Path]):
        self.model = lgb.Booster(model_file=str(model_path))
        self.stack_features = pd.read_parquet(stack_feature_path)
        self.feature_cols = ["retr_score", "bpr_score", "sasrec_score", "trans_score", "is_repeat"]

    def rank_user(self, user_id: int, retrieval_json: Union[str, pathlib.Path], topk: int = 100) -> pd.DataFrame:
        with open(retrieval_json, "r") as f:
            raw = json.load(f)
        raw_df = pd.DataFrame(raw)

        # join with precomputed stack features
        streamer_ids = raw_df["streamer_id"].tolist()
        feats = self.stack_features[
            (self.stack_features["user_id"] == user_id) &
            (self.stack_features["streamer_id"].isin(streamer_ids))
        ].copy()

        if feats.empty:
            print(f"[LGBMRanker] No features found for user {user_id}.")
            return pd.DataFrame(columns=["streamer_id", "score"])
        
        # Replace retr_score with the one from raw input
        # feats = feats.drop(columns=["retr_score"], errors="ignore")
        feats = feats.merge(raw_df[["streamer_id", "score"]], on="streamer_id", how="left")

        # Predict
        feats["score"] = self.model.predict(feats[self.feature_cols], num_iteration=self.model.best_iteration)
        return feats[["streamer_id", "score"]].sort_values("score", ascending=False).head(topk).reset_index(drop=True)

    def rank(
        self,
        user_ids: Union[int, list[int]],
        retrieval_dir: Union[str, pathlib.Path],
        topk: int = 100,
    ) -> dict[int, pd.DataFrame]:
        if isinstance(user_ids, int):
            user_ids = [user_ids]

        results = {}
        for uid in tqdm(user_ids, desc="Ranking users"):
            json_path = pathlib.Path(retrieval_dir) / f"user_{uid}.json"
            if json_path.exists():
                results[uid] = self.rank_user(uid, json_path, topk=topk)
            else:
                print(f"[LGBMRanker] Missing retrieval file for user {uid}")
        return results

    def dump_results(
        self,
        results: dict[int, pd.DataFrame],
        out_dir: Union[str, pathlib.Path],
    ):
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for user_id, df in results.items():
            out_file = out_dir / f"user_{user_id}.json"
            df[["streamer_id", "score"]].to_json(out_file, orient="records", force_ascii=False, indent=2)