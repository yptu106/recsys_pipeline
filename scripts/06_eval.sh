
#!/bin/bash
set -e

python -m src.eval.evaluate_retrieval \
    --test-path data/splits/test.parquet \
    --emb-dir embeddings \
    --index index/faiss/item_hnsw.idx \
    --user-log data/splits/interactions_train.parquet