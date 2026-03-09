#!/bin/bash
set -e

echo "Node 05: Structure Prediction + Scoring for Generated Sequences"
echo "ESMFold will load from HuggingFace cache (no re-download)"
echo "Estimated runtime: ~7 minutes for 50 sequences"

export HF_HUB_DISABLE_PROGRESS_BARS=1
export TQDM_DISABLE=1
export HF_TOKEN="${HF_TOKEN:-}"

python main.py

echo "Node 05 complete."
echo "Scored results: outputs/scored_generated.csv"
echo "PDB files:      outputs/generated_pdbs/*.pdb"