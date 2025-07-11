
#!/bin/bash
set -e

# Usage: ./01_build_streamer_features.sh [streamers_csv]

# Accept input CSV as first argument, or use default
STREAMERS_CSV="${1:-data/raw/streamers.csv}"
# OUTDIR="features/streamer"
# DATE=$(date +%F)

# Optionally activate your environment here
# source /path/to/venv/bin/activate

echo "Building streamer features for using $STREAMERS_CSV ..."
python -m src.preprocessing.build_features \
    --streamers_csv "$STREAMERS_CSV" \


