#!/usr/bin/env bash
: "${FIX_SRC:?Set a separate producer checkout at the required revision}"
: "${FIX_PYTHON:?Set the Python executable with producer dependencies}"
: "${STUDY_HOME:?Set an isolated build and script directory}"
: "${STUDY_RUNS:?Set the large-data root on local scratch storage}"
: "${GREEN_GRID_FILE:?Set the IR 1e6 grid path}"
export STUDY_MODEL=compact
export PYTHONPATH="$FIX_SRC:${PYTHONPATH:-}"
export PYSCF_TMPDIR="${PYSCF_TMPDIR:-$STUDY_RUNS/pyscf_tmp}"
export TMPDIR="$PYSCF_TMPDIR" OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" CUDACXX="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$STUDY_HOME/install/lib:$STUDY_HOME/install/lib64:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
mkdir -p "$STUDY_RUNS" "$PYSCF_TMPDIR"
