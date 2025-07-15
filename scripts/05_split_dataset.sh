
#!/bin/bash
set -e

python -m src.preprocessing.split_dataset \
    --interactions data/processed/interactions/latest.parquet \
    --streamer-lookup embeddings/MiniLM/item_sentence/lookup.parquet \
    --filter-missing-streamers \
    --out_dir data/splits \