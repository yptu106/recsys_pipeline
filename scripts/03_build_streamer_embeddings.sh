
#!/bin/bash
set -e

# Usage: ./03_build_streamer_embeddings.sh [features_parquet]
features_parquet="${1:-features/streamer/latest.parquet}"
format_type="$2" # item_sentence or format_sentence
include_numerical="$3" # true or false
model_name="$4" # MiniLM or bge

case "$model_name" in
  MiniLM)
    full_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ;;
  bge)
    full_model="bge-m3:567m"
    ;;
  *)
    echo "Unknown model: $model_name"
    exit 1
    ;;
esac

extra_args=""
if [ "$include_numerical" = "true" ]; then
  extra_args="--include-numerical-cols"
fi

echo "Building streamer embeddings using $features_parquet with format type $format_type and model $model_name ..."

python -m src.embeddings.build_streamer_emb \
    --features $features_parquet \
    --encode-col $format_type \
    --out-emb "embeddings/${model_name}/${format_type}"/streamer_embeddings.npy \
    --out-map "embeddings/${model_name}/${format_type}"/lookup.parquet \
    --model $full_model \
    $extra_args