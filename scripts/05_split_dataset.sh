
#!/bin/bash
set -e

interaction_type="${1:-donate}" # 'donate' or 'enter'
streamer_lookup="${2:-embeddings/MiniLM/item_sentence/lookup.parquet}" # choose whatever streamer lookup that contains all streamers

echo "Splitting dataset for interaction type: $interaction_type ..."

python -m src.preprocessing.split_dataset \
    --interactions data/processed/interactions/$interaction_type/latest.parquet \
    --streamer-lookup $streamer_lookup \
    --filter-missing-streamers \
    --out_dir data/splits/$interaction_type \