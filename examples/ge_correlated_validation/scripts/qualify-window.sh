#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/environment.sh"
export STUDY_MODEL=compact
export STUDY_RUNS="$STUDY_RUNS/compact-gth-szv-2x2x2-v2"
export REFERENCE_GF2_RANKS=4
"$FIX_PYTHON" "$STUDY_HOME/scripts/compare_mpi_reference.py"
"$FIX_PYTHON" "$STUDY_HOME/scripts/qualify_window.py"
