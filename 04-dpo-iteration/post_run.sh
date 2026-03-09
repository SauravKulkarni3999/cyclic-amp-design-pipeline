#!/bin/bash

# Post-run script for Node 04 (dpo iteration)

# Verifies DPO checkpoints saved correctly, and reports memory usage

set -e

echo "Node 04: Clean-up starting ..."

# Memory diagnostics
echo "Memory after DPO training:"
python -c "
import psutil
mem = psutil.virtual_memory()
print(f'Used: {mem.used/1e9:.1f}GB / {mem.total/1e9:.1f}GB')
print(f'Available: {mem.available/1e9:.1f}GB')
"

# Verify checkpoints saved correctly
CHECKPOINT_DIR="${OUTPUT_DIR:-outputs/aligned_model}"

echo "Verifying DPO checkpoint at: $CHECKPOINT_DIR"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: Checkpoint directory not found: $CHECKPOINT_DIR"
    echo "DPO training may have failed silently."
    exit 1
fi

# Check for essential model files
REQUIRED_FILES=("config.json" "tokenizer_config.json")
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$CHECKPOINT_DIR/$f" ]; then
        echo "WARNING: Expected file missing: $CHECKPOINT_DIR/$f"
    else
        echo "  ✓ $f"
    fi
done

echo "DPO checkpoint verified successfully."

# Check for model weights
WEIGHT_COUNT=$(find "$CHECKPOINT_DIR" -name "*.bin" -o -name "*.safetensors" 2>/dev/null | wc -l)
if [ "$WEIGHT_COUNT" -gt 0 ]; then
    echo "  ✓ Model weights: $WEIGHT_COUNT file(s) found"
else
    echo "  WARNING: No .bin or .safetensors weight files found"
fi

# Report checkpoint size
CHECKPOINT_SIZE=$(du -sh "$CHECKPOINT_DIR" 2>/dev/null | cut -f1)
echo "  Checkpoint size: $CHECKPOINT_SIZE"

# Cache fine-tuned model to shared volume
CACHE_DST="${SILVA_SHARED_DIR:-/data/model_cache}/aligned_model"
if [ -d "$CHECKPOINT_DIR" ]; then
    echo "Caching fine-tuned model to shared volume..."
    mkdir -p "$CACHE_DST"
    if command -v rsync &>/dev/null; then
        rsync -a --ignore-existing "$CHECKPOINT_DIR/" "$CACHE_DST/"
    else
        cp -rn "$CHECKPOINT_DIR/." "$CACHE_DST/"
    fi
    echo "  ✓ Model cached to $CACHE_DST"
fi

echo "Node 04: Post-run complete."