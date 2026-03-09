#!/bin/bash
set -e

echo "Extracting sequences for prediction..."
python prepare_fasta.py 

mkdir -p outputs/prediction_results/

echo "Running ESMFold predictions locally..."
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TQDM_DISABLE=1

python predict_esmfold.py 

echo "Prediction completed successfully."
echo "Results saved to outputs/prediction_results/"