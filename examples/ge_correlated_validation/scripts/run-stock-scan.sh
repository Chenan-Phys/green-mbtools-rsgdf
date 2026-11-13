#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/environment.sh"
export STUDY_MODEL=compact
export STUDY_RUNS="$STUDY_RUNS/compact-gth-szv-2x2x2-v2"
"$FIX_PYTHON" "$STUDY_HOME/scripts/run_stock_scan.py"
