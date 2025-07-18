#!/bin/bash
set -e

interaction_type="$1" # donate or enter
ranker="${2:-mlp}" # Default to mlp if not provided

echo "Evaluating ranking for $interaction_type using ranker $ranker ..."

python -m src.eval.evaluate_rank \
    --test-path data/splits/$interaction_type/test.parquet \
    --ranked-dir results/ranked/$ranker/ 