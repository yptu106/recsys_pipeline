
#!/bin/bash
set -e

# Usage: ./01_build_interactions.sh [interactions_csv] [filter_condition]

interactions_csv="$1"
filter_condition="$2" # donate or enter

python -m src.preprocessing.build_interactions \
    --csv $interactions_csv \
    --filter-conditions $filter_condition \
    --out-dir data/processed/interactions/$filter_condition
