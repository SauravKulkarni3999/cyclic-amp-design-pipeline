#!/bin/bash
set -e
echo "Node 03: Post-run. Waiting for memory reclaim..."

# Wait until host has at least 8GB free before Node 04 can start
for i in $(seq 1 20); do
    FREE=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
    echo "  Attempt $i: ${FREE}GB available"
    if [ "$FREE" -ge 8 ]; then
        echo "  Sufficient memory available. Proceeding."
        break
    fi
    sleep 5
done

echo "Node 03: Post-run complete."