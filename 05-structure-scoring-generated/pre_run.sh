#!/bin/bash
# Pre-run script for Node 05 (structure scoring generated)
# Restores ESMFold weights from the shared cache written by
# Node 02's post_run.sh - avoids re-downloading 2.7GB from
# HuggingFace and ensures Node 05 starts with known memory state.

set -e

echo "Node 05: Pre-run starting ..."
echo "Memory before pre-run:"
python -c "
import psutil
mem = psutil.virtual_memory()
print(f'Used: {mem.used/1e9:.1g}GB / {mem.total/1e9:.1g}GB ({mem.percent:.0f}%)')
"

# Restore ESMFold weights from shared cache 
CACHE_SRC="${SILVA_SHARED_DIR:-/data/model_cache}/esmfold"
CACHE_DST="$HOME/.cache/huggingface/hub/models--facebook--esmfold_v1"

if [ -d "$CACHE_SRC" ]; then
    echo "[pre_run] Restoring ESMFold weights from shared cache..."
    mkdir -p "$(dirname $CACHE_DST)"
    if command -v rsync &>/dev/null; then
        rsync -a "$CACHE_SRC/" "$CACHE_DST/"
    else
        cp -r "$CACHE_SRC/." "$CACHE_DST/"
    fi
    echo "[pre_run] ESMFold weights restored — no HuggingFace download needed"
    
    # Verify weights are intact
    python3 - <<PYEOF
import os
cache_dst = os.path.expanduser("$CACHE_DST")
if os.path.exists(cache_dst):
    size_gb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, dn, filenames in os.walk(cache_dst)
        for f in filenames
    ) / 1e9
    print(f"[pre_run] Cache verified: {size_gb:.1f}GB at {cache_dst}")
else:
    print("[pre_run] WARNING: Cache restore may have failed")
PYEOF

else
    echo "[pre_run] Shared cache not found at $CACHE_SRC"
    echo "[pre_run] ESMFold will download from HuggingFace (~2.7GB)"
fi

# Final memory check 
echo "[pre_run] Final memory before ESMFold load:"
python3 -c "
import psutil
mem = psutil.virtual_memory()
print(f'  Available: {mem.available/1e9:.1f}GB')
if mem.available/1e9 < 4:
    print('  WARNING: Less than 4GB available — ESMFold load may fail')
else:
    print('  Status: OK')
"

echo "[pre_run] Node 05 pre-run complete."