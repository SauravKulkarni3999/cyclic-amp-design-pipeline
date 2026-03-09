#!/bin/bash

# Pre-run script for Node 03 (membrane scoring)

# Prepares workspace for scoring

set -e

pip install psutil -q 2>/dev/null || true

echo "Node 03: Pre-run starting ..."
echo "Memory before pre-run:"
python -c "
import psutil
mem = psutil.virtual_memory()
print(f'Used: {mem.used/1e9:.1g}GB / {mem.total/1e9:.1g}GB ({mem.percent:.0f}%)')
"

echo "[pre_run] Node 03 pre-run complete."
