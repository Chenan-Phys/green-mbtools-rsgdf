#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/environment.sh"
export STUDY_MODEL=compact
export STUDY_RUNS="$STUDY_RUNS/compact-gth-szv-2x2x2-v2"
if [ ! -f "$STUDY_RUNS/canonical-mpi-check/a5.700000/gf2-cpu-ccgdf/run.json" ]; then
  "$FIX_PYTHON" "$STUDY_HOME/scripts/run_green.py" --a 5.7 --method GF2 --mode cpu --backend ccgdf \
  --stage canonical-mpi-check --density-tolerance 1e-12 --energy-threshold 1e-11 \
  --mixing-type SIGMA_MIXING --mixing-weight .15 --itermax 100 --timeout 600 --ranks 4 \
  --seed-results "$STUDY_RUNS/canonical-refined/a5.700000/gf2-cpu-ccgdf/results.h5"
fi
"$FIX_PYTHON" "$STUDY_HOME/scripts/compare_mpi_reference.py"
