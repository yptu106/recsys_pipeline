
#!/bin/bash
set -e

python -m src.preprocessing.split_dataset \
    --interactions data/processed/interactions/latest.parquet \
    --streamer-lookup embeddings/lookup.parquet \
    --out_dir data/splits \