#!/bin/bash
set -e

echo "Node 06: Physics Validation (OpenMM)"
echo "Force field: AMBER14 + GBn2 implicit solvent"
echo "Validating top 10 candidates from Node 05..."

export TOP_N=${TOP_N:-10}
export MAX_ITERATIONS=${MAX_ITERATIONS:-500}

python main.py

echo "Node 06 complete."
echo "Results: outputs/physics_validation.csv"
echo "Summary: outputs/validation_summary.json"