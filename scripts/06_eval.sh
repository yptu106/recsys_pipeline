
#!/bin/bash
set -e
interaction_type="$1" # donate or enter
emb_type="$2" # item_sentence or format_sentence or item_sentence_num or format_sentence_num
model_name="$3" # MiniLM or bge

echo "Evaluating retrieval for $interaction_type with $emb_type embeddings using model $model_name ..."

python -m src.eval.evaluate_retrieval \
    --test-path data/splits/$interaction_type/test.parquet \
    --emb-dir embeddings/$model_name/$emb_type \
    --index index/faiss/$model_name/$emb_type/index_flat.idx \
    --user-log data/splits/$interaction_type/interactions_train.parquet