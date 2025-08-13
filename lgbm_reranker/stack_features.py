# scripts/dump_stack_features.py
import pathlib, json, pandas as pd, numpy as np
import pickle
from functools import lru_cache
from tqdm import tqdm

SRC_DIR   = pathlib.Path("results/retrieval/MiniLM_40epoch_ts_heavy")   # existing top-500 JSONs
BPR_DIR   = pathlib.Path("results/ranked/BPR/time_based_heavy")       # output from RecBole BPR
SASREC_DIR= pathlib.Path("results/ranked/SASRec/ts_heavy")    # output from RecBole SASRec
TRANS_DIR = pathlib.Path("results/ranked/transformer/time_based_heavy")  

repeat_lookup = pickle.load(open("reranker/repeat_lookup.pkl", "rb"))
pos_val_lookup = pickle.load(open("reranker/gt_val_lookup.pkl", "rb"))

def is_repeat(uid: int, iid: int) -> int:
    return int(iid in repeat_lookup.get(uid, ())) # 1 or 0

def is_positive(uid: int, iid: int) -> int:
    """
    Check if <uid, iid> is a positive sample in the validation set.
    Returns 1 if positive, else 0.
    """
    return int(iid in pos_val_lookup.get(uid, ())) # 1 or 0

@lru_cache(maxsize=None)
def _load_user_scores(file_path: pathlib.Path):
    """
    Load the entire 'user_xxx.json' once, convert to {item_id: score}.
    Cached so multiple look-ups for the same user are O(1).
    """
    with open(file_path) as f:
        lst = json.load(f)                # a list of {"streamer_id": ..., "score": ...}
    return {d["streamer_id"]: d["score"] for d in lst}

def get_candidate_score(uid: int, iid: int, directory: pathlib.Path,
                        default: float = np.nan) -> float:
    """
    Return the score assigned by the model for <uid, iid>.
    If the user file or the specific item is absent, return `default`.
    """
    file_path = directory / f"user_{uid}.json"
    if not file_path.exists():
        return default

    mapping = _load_user_scores(file_path)
    return mapping.get(iid, default)

rows = []
for file in tqdm(SRC_DIR.glob("user_*.json"), desc="Processing user files"):
    uid = int(file.stem.split('_')[1])
    cand_df = pd.read_json(file)
    cand_iids = set()

    for row in cand_df.itertuples():
        iid   = row.streamer_id
        cand_iids.add(iid)

        rows.append({
            "user_id"     : uid,
            "streamer_id" : iid,
            "label"       : is_positive(uid, iid),  # 1 if positive, else 0
            "retr_score"  : row.score,                       # similarity from MiniLM
            "bpr_score"   : get_candidate_score(uid, iid, BPR_DIR,  default=np.nan),
            "sasrec_score": get_candidate_score(uid, iid, SASREC_DIR,default=np.nan),
            "trans_score" : get_candidate_score(uid, iid, TRANS_DIR, default=np.nan),
            "is_repeat"   : is_repeat(uid, iid)               # 1 if repeat, else 0
        })

    # Add missing positive items that were not retrieved
    missing_pos_iids = pos_val_lookup.get(uid, set()) - cand_iids
    for iid in missing_pos_iids:
        rows.append({
            "user_id"     : uid,
            "streamer_id" : iid,
            "label"       : 1,
            "retr_score"  : np.nan,  # no retrieval score for missing items
            "bpr_score"   : np.nan,
            "sasrec_score": np.nan,
            "is_repeat"   : is_repeat(uid, iid)  # 1 if repeat, else 0
        })

df = pd.DataFrame(rows)
df.to_parquet("reranker/train.parquet")
