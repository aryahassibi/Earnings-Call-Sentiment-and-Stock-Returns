#!/usr/bin/env bash
set -euo pipefail

# List of parquet files
FILES=(
  "data/additional_ec_data.parquet"
  "data/algoseek_nyse_nasdaq.parquet"
  "data/ec_mapping.parquet"
  "data/transcripts.parquet"
  "data/var_descriptions.parquet"
)
source .venv/bin/activate
# Loop through each file and run your script
for f in "${FILES[@]}"; do
  echo "Processing $f ..."
  python utils/analyze_parquet.py "$f"
done
