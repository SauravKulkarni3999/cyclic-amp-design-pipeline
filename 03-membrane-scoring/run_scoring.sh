#!/bin/bash


# Install dependencies if not in container
pip install biopython pandas numpy

# Execute main script

if [ -z "$SCORING_CONFIG" ]; then
  SCORING_CONFIG='{"target_organism":"E_coli","anionic_fraction":0.25,"hydrophobic_moment_weight":5.0,"cyclicity_penalty_threshold":4.5}'
fi
echo "SCORING_CONFIG: $SCORING_CONFIG"
python main.py --config "$SCORING_CONFIG"