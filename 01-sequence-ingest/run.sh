#!/bin/bash

set -e
echo "Starting sequence ingestion..."
mkdir -p outputs

DATA_PATHS="${DATA_PATHS:-{\"input\":\"inputs/sequences.csv\", \"output\":\"outputs/ingested_sequences.csv\"}}"
echo "Config string received: $DATA_PATHS"

python run_ingest.py "$DATA_PATHS"

echo "Sequence ingestion completed successfully."