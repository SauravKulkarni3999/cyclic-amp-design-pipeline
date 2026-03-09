#!/bin/bash

set -e


echo "Node 02: Pre-run starting ..."
echo "Memory before pre-run:"
python -c "
import psutil
mem = psutil.virtual_memory()
print(f'Used: {mem.used/1e9:.1g}GB / {mem.total/1e9:.1g}GB ({mem.percent:.0f}%)')
"

echo "[pre_run] Node 02 pre-run complete."