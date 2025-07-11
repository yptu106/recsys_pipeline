
#!/bin/bash
set -e


python -m src.vector_store.build_faiss \
    --embeddings embeddings/streamer_embeddings.npy \
    --out index/faiss/item_hnsw.idx \
    --index-type hnsw