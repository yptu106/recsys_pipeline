
#!/bin/bash
set -e

# Usage: ./01_run_retrieval.sh [interaction_type] [emb_type] [model_name] [k]

interaction_type="$1" # donate or enter
emb_type="$2" # item_sentence or format_sentence or item_sentence_num or format_sentence_num
model_name="$3" # MiniLM or bge
k="${4:-100}" # Number of top candidates to retrieve, default is 100

python -m src.eval.run_retrieval \
    --test-path data/splits/$interaction_type/test.parquet \
    --emb-dir  embeddings/streamer/$model_name/$emb_type \
    --index index/faiss/$model_name/$emb_type/index_flat.idx \
    --user-log data/splits/$interaction_type/interactions_train.parquet \
    --out-dir results/retrieval/$model_name/$emb_type/ \
    --k $k