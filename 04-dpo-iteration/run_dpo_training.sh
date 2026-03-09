#!/bin/bash
set -e

echo "Starting DPO Alignment..."

# Print workspace root so we can verify Silva's mount point
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
echo "WORKSPACE_ROOT: $WORKSPACE_ROOT"
echo "Contents of workspace:"
ls "$WORKSPACE_ROOT" 2>/dev/null || echo "  WARNING: $WORKSPACE_ROOT not found"

# Define input/output paths
NODE_01_OUT="$WORKSPACE_ROOT/01-sequence-ingest/output_files/ingested_sequences.csv"
NODE_03_OUT="$WORKSPACE_ROOT/03-membrane-scoring/preferences.jsonl"
OUTPUT_DIR="outputs"

# Verify input files exist before running
echo "Checking input files..."
if [ ! -f "$NODE_01_OUT" ]; then
    echo "ERROR: Node 01 output not found at: $NODE_01_OUT"
    echo "Searching for ingested_sequences.csv anywhere in workspace..."
    find "$WORKSPACE_ROOT" -name "ingested_sequences.csv" 2>/dev/null || echo "  Not found"
    exit 1
fi
echo "  ✓ $NODE_01_OUT"

if [ ! -f "$NODE_03_OUT" ]; then
    echo "ERROR: Node 03 output not found at: $NODE_03_OUT"
    echo "Searching for preferences.jsonl anywhere in workspace..."
    find "$WORKSPACE_ROOT" -name "preferences.jsonl" 2>/dev/null || echo "  Not found"
    exit 1
fi
echo "  ✓ $NODE_03_OUT"

echo "Running DPO training..."
python main.py \
    --seq_csv_path "$NODE_01_OUT" \
    --pref_path "$NODE_03_OUT" \
    --output_dir "$OUTPUT_DIR"