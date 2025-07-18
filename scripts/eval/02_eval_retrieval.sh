#!/bin/bash
set -e

# Usage: ./02_eval_retrieval.sh [interaction_type] [emb_type] [model_name]

interaction_type="$1" # donate or enter
emb_type="$2" # item_sentence or format_sentence or item_sentence_num or format_sentence_num
model_name="$3" # MiniLM or bge
# ks =("${@:4}") # List of k values, e.g., 10 20 50

echo "Evaluating retrieval for $interaction_type with $emb_type embeddings using model $model_name ..."

python -m src.eval.evaluate_retrieval \
    --test-path data/splits/$interaction_type/test.parquet \
    --retrieval-dir results/retrieval/$model_name/$emb_type/