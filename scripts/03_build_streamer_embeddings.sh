
#!/bin/bash
set -e

# Usage: ./03_build_streamer_embeddings.sh [features_parquet]

python -m src.embeddings.build_streamer_emb \
    --features features/streamer/latest.parquet
