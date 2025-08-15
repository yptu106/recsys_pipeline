"""build_faiss.py
Create a FAISS index from the NumPy matrix exported by `build_item_vecs.py`.

Supports three index types:
    * flat        – exact cosine similarity (IndexFlatIP)
    * hnsw        – HNSW graph (IndexHNSWFlat)  [good default]
    * ivfpq       – IVF + PQ (quantised, memory‑efficient)

Example
-------
python -m vector_store.build_faiss \
        --vectors embeddings/streamer_embeddings.npy \
        --out     index/faiss/index_flat.idx \
        --index-type flat
"""
from __future__ import annotations

import argparse
import pathlib
import time

import faiss
import numpy as np


INDEX_TYPES = {"flat", "hnsw", "ivfpq"}


def build_index(x: np.ndarray, kind: str = "hnsw") -> faiss.Index:
    d = x.shape[1]
    kind = kind.lower()
    if kind == "flat":
        index = faiss.IndexFlatIP(d)                     # exact cosine (after L2‑norm)
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, 32)              # M=32 by default
        index.hnsw.efConstruction = 200
    elif kind == "ivfpq":
        nlist = int(np.sqrt(len(x)))                    # heuristic
        m = 8                                           # PQ bytes per vector
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)  # 8 bits per sub‑quantiser
        index.nprobe = int(max(1, 0.05 * nlist))
        print(f"[ivfpq] nlist={nlist} nprobe={index.nprobe} m={m}")
        print("→ training IVF‑PQ (this can take a minute)…")
        index.train(x)
    else:
        raise ValueError(f"index_type must be one of {INDEX_TYPES}")

    tic = time.time()
    index.add(x)
    print(f"✓ added {index.ntotal} vectors to FAISS index in {time.time()-tic:.1f}s")
    return index



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, help="NumPy .npy from `build_streamer_emb.py`")
    parser.add_argument("--out",     required=True, help="Output .idx path")
    parser.add_argument("--index-type", default="hnsw", choices=INDEX_TYPES)
    args = parser.parse_args()

    x = np.load(args.embeddings, mmap_mode="r")            # float32 [N, d] already L2‑normed
    print(f"loaded embeddings: {x.shape[0]}×{x.shape[1]}")

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    idx = build_index(x, args.index_type)
    faiss.write_index(idx, args.out)
    print(f"✓ wrote FAISS index → {args.out}")


if __name__ == "__main__":
    main()
