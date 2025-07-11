
#!/bin/bash
set -e

# Usage: ./02_build_interactions.sh [interactions_csv]

INTERACTIONS_CSV="${1:-data/raw/streamers.csv}"

python -m src.preprocessing.build_interactions \
    --csv "${1:-data/raw/interactions.csv}" \

