
#!/bin/bash
set -e

# Usage: ./02_build_streamer_features.sh [streamers_csv] [interaction_type]

streamers_cvs="$1"
interaction_type="${2:-donate}" # 'donate' or 'enter'

# interactions data currently might cause data leakage since we split data based on this interactions

echo "Building streamer features for using $streamers_cvs ..."
python -m src.preprocessing.build_features \
    --streamers_csv $streamers_cvs \
    --user_interactions data/processed/interactions/$interaction_type/latest.parquet \
    --outdir features/streamer

