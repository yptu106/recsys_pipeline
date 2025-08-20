# scripts/train_lgbm_ranker.py
import lightgbm as lgb, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from lightgbm import LGBMRanker, early_stopping
from joblib import dump

df = pd.read_parquet("reranker/train.parquet")

# -------------  split by user so no leakage -------------
train_u, val_u = train_test_split(df.user_id.unique(),
                                  test_size=0.1, random_state=42)
train_df = df[df.user_id.isin(train_u)]
val_df   = df[df.user_id.isin(val_u)]

FEATURES = ["retr_score", "bpr_score", "sasrec_score",
            "trans_score", "is_repeat"]

def build_lgb_dataset(sub):
    X = sub[FEATURES]
    y = sub["label"]
    group = sub.groupby("user_id").size().values
    return lgb.Dataset(X, y, group=group)

train_set = build_lgb_dataset(train_df)
val_set   = build_lgb_dataset(val_df)

model = LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    ndcg_eval_at=[10, 20, 50],
    num_leaves=63,
    n_estimators=100,
    learning_rate=0.05,
)

callbacks = [early_stopping(stopping_rounds=30, verbose=True)]

model.fit(
    train_set.data, train_set.label,
    group=train_set.group,
    eval_set=[(val_set.data, val_set.label)],
    eval_group=[val_set.group],
    eval_metric="ndcg",
    callbacks=callbacks
)

# Save model and scaler
output_dir = Path("reranker/models")
output_dir.mkdir(parents=True, exist_ok=True)

model_path = output_dir / f"lgbm.txt"
model.booster_.save_model(str(model_path))
