#!/bin/bash
# post_run.sh — Node 02 (ESMFold API version)
# No model weights to cache — API version has no local downloads.
# Just verifies outputs and reports memory.

set -e

echo "Node 02: Post-run starting..."

# Memory diagnostic
FREE=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
TOTAL=$(awk '/MemTotal/{printf "%.1f", $2/1048576}' /proc/meminfo)
echo "Memory after prediction: ${FREE}GB free / ${TOTAL}GB total"

# Verify PDB outputs were created
PDB_COUNT=$(find outputs/prediction_results/ -name "*.pdb" 2>/dev/null | wc -l)
echo "PDB files created: $PDB_COUNT"

if [ "$PDB_COUNT" -eq 0 ]; then
    echo "ERROR: No PDB files found. API predictions may have failed."
    exit 1
fi

echo "Node 02: Post-run complete."