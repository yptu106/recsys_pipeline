
#!/bin/bash
set -e

# Usage: ./01_build_streamer_features.sh [streamers_csv]

streamers_cvs="$1"
user_interactions_parquet="${2:-data/processed/interactions/latest.parquet}"

echo "Building streamer features for using $streamers_cvs ..."
python -m src.preprocessing.build_features \
    --streamers_csv $streamers_cvs \
    --user_interactions $user_interactions_parquet \
    --outdir features/streamer

