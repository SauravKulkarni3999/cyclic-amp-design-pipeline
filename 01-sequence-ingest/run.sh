#!/bin/bash

set -e
echo "Starting sequence ingestion..."
mkdir -p output_files

DATA_PATHS="${DATA_PATHS:-{\"input\":\"sequences.csv\", \"output\":\"output_files/ingested_sequences.csv\"}}"
echo "Config string received: $DATA_PATHS"

pip install --upgrade pip && pip install pandas peptides
python run_ingest.py "$DATA_PATHS"

echo "Sequence ingestion completed successfully."