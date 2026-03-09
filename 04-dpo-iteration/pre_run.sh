#!/bin/bash

# Pre-run script for Node 04 (dpo iteration)

# Prepares workspace for DPO training

set -e

FREE=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
TOTAL=$(awk '/MemTotal/{printf "%.1f", $2/1048576}' /proc/meminfo)
echo "Memory: ${FREE}GB free / ${TOTAL}GB total"
echo "[pre_run] Node 04 pre-run complete."