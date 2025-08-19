import torch
import json
import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union, List
from recbole.data.interaction import Interaction
from src.config import USER_ID_COL, STREAMER_ID_COL
from src.ranker.utils.io import ensure_dir_exists

class InteractionBuilder:
    """
    Construct an Interaction object that satisfies RecBole's requirements.
    Handles ID-only, sequential, and feature-based models.
    """
    def __init__(self, dataset, model, recbole_config, device):
        self.dataset = dataset
        self.cfg = recbole_config
        self.device = device

        # TODO: detect context fields

        # # detect sequential fields
        # self.seq_field = self.cfg["ITEM_SEQ"] if "ITEM_SEQ" in self.cfg else None
        # self.len_field = self.cfg["ITEM_LIST_LENGTH_FIELD"] if "ITEM_LIST_LENGTH_FIELD" in self.cfg else None
        # self.max_L = self.cfg["MAX_ITEM_LIST_LENGTH"] if "ITEM_LIST_LENGTH_FIELD" in self.cfg else None

        if hasattr(model, "ITEM_SEQ"):
            self.seq_field = model.ITEM_SEQ                   # 'streamer_id_list'
            self.len_field = self.cfg["ITEM_LIST_LENGTH_FIELD"]
            self.max_L     = self.cfg["MAX_ITEM_LIST_LENGTH"]
            self.user_history = self._build_histories()
        else:
            self.seq_field = self.len_field = None
            self.max_L     = 0
            self.user_history = {}

        print(f"InteractionBuilder: seq_field={self.seq_field}, len_field={self.len_field}, max_L={self.max_L}")

        # # pre-build per-user history in interal ids
        # self.user_history = {}
        # if self.seq_field:
        #     self.user_history = self._build_histories()

    # def _build_histories(self):
    #     """
    #     Returns
    #     -------
    #     dict
    #         {internal_uid_int: torch.LongTensor([iid1, iid2, ...])}
    #         Sequence is already truncated to the last `max_L` items.
    #     """
    #     inter = self.dataset.inter_feat.sort_values(self.cfg["TIME_FIELD"])
    #     user_history = {}
    #     for uid, rows in inter.groupby(self.cfg["USER_ID_FIELD"], sort=False):
    #         seq = rows[self.cfg["ITEM_ID_FIELD"]].tolist()[-self.max_L:]
    #         user_history[uid] = torch.tensor(seq, dtype=torch.long)
    #     return user_history

    def _build_histories(self):
        inter = self.dataset.inter_feat
        uid_f, iid_f, t_f = self.cfg["USER_ID_FIELD"], self.cfg["ITEM_ID_FIELD"], self.cfg["TIME_FIELD"]

        # branch 1: RecBole Interaction -> pandas (if available)
        if hasattr(inter, "to_pandas"):
            df = inter.to_pandas()
            df = df.sort_values(t_f)
            return {
                int(uid): torch.as_tensor(g[iid_f].to_numpy()[-self.max_L:], dtype=torch.long)
                for uid, g in df.groupby(uid_f, sort=False)
            }

        # branch 2: Already a pandas DataFrame
        if isinstance(inter, pd.DataFrame):
            df = inter.sort_values(t_f)
            return {
                int(uid): torch.as_tensor(g[iid_f].to_numpy()[-self.max_L:], dtype=torch.long)
                for uid, g in df.groupby(uid_f, sort=False)
            }

        # branch 3: Pure Interaction (no pandas ops)
        def to_np(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        u = to_np(inter[uid_f])
        i = to_np(inter[iid_f])
        t = to_np(inter[t_f])

        order = np.argsort(t, kind="stable")
        u, i = u[order], i[order]

        user_history = {}
        for uid, iid in zip(u, i):
            lst = user_history.setdefault(int(uid), [])
            lst.append(int(iid))
            if len(lst) > self.max_L:
                lst.pop(0)  # keep only the most recent max_L

        return {uid: torch.tensor(seq, dtype=torch.long) for uid, seq in user_history.items()}
    
    def make_batch(self, uid: int, iid_list: List[int], history: torch.Tensor):
        """
        Build one batched Interaction:
            • user tensor  shape (B,)
            • item tensor  shape (B,)
            • (optional) streamer_id_list shape (B , L)
            • (optional) item_length      shape (B,)
        """
        if history is None or history.numel() == 0:
            history = torch.zeros(0, dtype=torch.long)

        B = len(iid_list)
        data = {
            self.cfg["USER_ID_FIELD"]: torch.full((B,), uid, dtype=torch.long, device=self.device),
            self.cfg["ITEM_ID_FIELD"]: torch.tensor(iid_list, dtype=torch.long, device=self.device),
        }

        if self.seq_field:
            # sequential model
            pad_len = self.max_L - len(history)
            seq = torch.cat(
                [torch.zeros(pad_len, dtype=torch.long), history]
            ) # left-pad to max_L
            data[self.seq_field] = seq.unsqueeze(0).repeat(B, 1).to(self.device)
            data[self.len_field] = torch.full(
                (B,), len(history), dtype=torch.long, device=self.device
            )
        
        return Interaction(data)

class RecBoleRanker:
    def __init__(self, model, dataset, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config = config # recbole config
        self.model = model
        self.dataset = dataset
        self.model.to(self.device)
        self.model.eval()
        self.interaction_builder = InteractionBuilder(dataset, self.model, config, self.device)

    def rank(
        self,
        user_ids: Union[int, List[int]],
        retrieval_dir: Union[str, pathlib.Path],
        topk: int = 100,
        batch_size: int = 256,
    ) -> dict[int, pd.DataFrame]:
        """
        Rank top-k items from retrieval candidates using RecBole model.
        """
        if isinstance(user_ids, int):
            user_ids = [user_ids]

        user_to_candidates = {}
        for user_id in tqdm(user_ids, desc="Loading user retrievals"):
            try:
                with open(pathlib.Path(retrieval_dir) / f"user_{user_id}.json", "r") as f:
                    recs = json.load(f)
                candidate_ids = pd.DataFrame(recs)[STREAMER_ID_COL].tolist()
                if not candidate_ids:
                    print(f"No candidates found for user {user_id}")
                    continue
                user_to_candidates[user_id] = candidate_ids
            except FileNotFoundError:
                print(f"Retrieval file for user {user_id} not found. Skipping.")
                continue

        return self._rank_batch(user_to_candidates, topk=topk, batch_size=batch_size)

    def _rank_batch(self, user_to_candidates: dict[int, list[int]], topk: int, batch_size: int = 256) -> dict[int, pd.DataFrame]:
        results, skipped_users = {}, set()

        # check if the model is sequential
        is_sequential = self.interaction_builder.seq_field is not None

        for user_id, candidate_ids in tqdm(user_to_candidates.items(), desc="Scoring user-item pairs"):
            # try to map user ID to internal ID
            try:
                internal_uid = self.dataset.token2id(self.config["USER_ID_FIELD"], str(user_id))
            except ValueError: # model not trained on this user
                skipped_users.add(user_id)
                continue
        
            # get user's history
            if is_sequential:
                history = self.interaction_builder.user_history.get(internal_uid, torch.zeros(0, dtype=torch.long))
            else:
                history = torch.zeros(0, dtype=torch.long)  # no history for ID-only models
        
            # map candidate IDs to internal IDs
            internal_iid_list = []
            valid_item_ids = []
            for item_id in candidate_ids:
                try:
                    internal_iid = self.dataset.token2id(self.config["ITEM_ID_FIELD"], str(item_id))
                    internal_iid_list.append(internal_iid)
                    valid_item_ids.append(item_id)
                except ValueError:
                    # print(f"[WARN] Item {item_id} not in vocab — skipping.")
                    continue
        
            if not internal_iid_list:
                # print(f"No valid candidates for user {user_id}")
                skipped_users.add(user_id)
                continue
                
            # batch scoring
            scores = []
            for i in range(0, len(internal_iid_list), batch_size):
                batch_iids = internal_iid_list[i: i + batch_size]
                interaction = self.interaction_builder.make_batch(internal_uid, batch_iids, history)
                with torch.no_grad():
                    batch_scores = self.model.predict(interaction).cpu().numpy()
                scores.extend(batch_scores)
            
            # create DataFrame of results
            df = pd.DataFrame({
                STREAMER_ID_COL: valid_item_ids,
                "score": scores
            }).sort_values("score", ascending=False).head(topk).reset_index(drop=True)

            results[user_id] = df
        
        if skipped_users:
            print(f"Skipped {len(skipped_users)} users due to missing data or vocab issues.")
        
        return results

    # def _rank_batch(self, user_to_candidates: dict[int, list[int]], topk: int) -> dict[int, pd.DataFrame]:
    #     results = {}
    #     candidate_skipped = set()
    #     for user_id, candidate_ids in tqdm(user_to_candidates.items(), desc="Scoring user-item pairs"):
    #         # Map to RecBole's internal token IDs
    #         uid = self.dataset.token2id("user_id", str(user_id))
    #         # iid_list = [self.dataset.token2id("item_id", str(item_id)) for item_id in item_ids]

    #         # Convert candidate item IDs to internal IDs
    #         iid_list = []
    #         valid_item_ids = []
    #         for item_id in candidate_ids:
    #             try:
    #                 iid = self.dataset.token2id(STREAMER_ID_COL, str(item_id))
    #                 if iid is not None:
    #                     iid_list.append(iid)
    #                     valid_item_ids.append(item_id)
    #             except ValueError:
    #                 candidate_skipped.add(item_id)
    #                 # print(f"[WARN] Item {item_id} not in vocab — skipping.")

    #         if not iid_list:
    #             print(f"No valid candidates for user {user_id}")
    #             continue

    #         # Sanity check
    #         if uid is None or any(i is None for i in iid_list):
    #             raise ValueError("User or item not found in the dataset vocab")

    #         user_tensor = torch.tensor([uid] * len(iid_list), device=self.device)
    #         item_tensor = torch.tensor(iid_list, device=self.device)
    #         interaction = Interaction({
    #             self.config['USER_ID_FIELD']: user_tensor,
    #             self.config['ITEM_ID_FIELD']: item_tensor
    #         })

    #         with torch.no_grad():
    #             scores = self.model.predict(interaction).cpu().numpy()

    #         # Map back to original candidate_ids (raw string IDs if needed)
    #         df = pd.DataFrame({
    #             STREAMER_ID_COL: valid_item_ids,
    #             "score": scores
    #         }).sort_values("score", ascending=False).reset_index(drop=True)

    #         df = df.sort_values("score", ascending=False).head(topk)
    #         results[user_id] = df.reset_index(drop=True)

    #     print(f"Skipped {len(candidate_skipped)} items not found in the dataset vocab.")

    #     return results

    def dump_results(
        self,
        results: dict[int, pd.DataFrame],
        out_dir: Union[str, pathlib.Path],
    ):
        """
        Save ranked results to the specified output directory.
        Each user's results are saved as a JSON file.
        """
        out_dir = pathlib.Path(out_dir)
        ensure_dir_exists(out_dir)

        for user_id, df in tqdm(results.items(), desc="Saving results"):
            out_file = out_dir / f"user_{user_id}.json"
            df[[STREAMER_ID_COL, "score"]].to_json(out_file, orient="records", force_ascii=False, indent=2)
