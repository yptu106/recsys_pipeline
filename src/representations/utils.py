import pathlib
import pandas as pd
import numpy as np
from typing import Union, List, Optional

def get_emb_paths(dir_path: Union[str, pathlib.Path]) -> tuple[str, str]:
    dir_path = pathlib.Path(dir_path)
    emb_path = next(dir_path.glob("*.npy"))
    lookup_path = next(dir_path.glob("*.parquet"))
    return str(emb_path), str(lookup_path)

def load_embeddings(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    else:
        raise ValueError(f"Unsupported file format for embeddings: {path}")

def load_lookup_table(path: str, id_col: str) -> dict:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format for lookup table: {path}")
    
    df = df.reset_index(drop=True)

    if id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' not found in the lookup table.")
    if df[id_col].isnull().any():
        raise ValueError(f"Column '{id_col}' contains null values.")
    if df[id_col].duplicated().any():
        raise ValueError(f"Column '{id_col}' contains duplicate values.")
    
    return dict(zip(df[id_col], df.index))

def load_embedding_and_lookup(emb_dir: str, id_col: str) -> tuple[np.ndarray, dict]:
    emb_path, lookup_path = get_emb_paths(emb_dir)
    emb = load_embeddings(emb_path)
    lookup = load_lookup_table(lookup_path, id_col)
    return emb, lookup