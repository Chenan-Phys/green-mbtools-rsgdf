#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/environment.sh"
export OMP_NUM_THREADS=2
mkdir -p "$STUDY_HOME/evidence"
git -C "$FIX_SRC" diff --exit-code
test "$(git -C "$FIX_SRC" rev-parse HEAD)" = 0f101cb945b4d13cb71c9926fa023942ec14b60f
if [ ! -d "$STUDY_HOME/green-mbpt/.git" ]; then
  git clone https://github.com/Green-Phys/green-mbpt.git "$STUDY_HOME/green-mbpt"
  git -C "$STUDY_HOME/green-mbpt" checkout --detach 2346c87763e20cea5c55b95ab769ee55421f7386
fi
if [ ! -d "$STUDY_HOME/green-gpu/.git" ]; then
  git clone https://github.com/Green-Phys/green-gpu.git "$STUDY_HOME/green-gpu"
  git -C "$STUDY_HOME/green-gpu" checkout --detach 4dcb075a9084f801dfb04a513112a736f4423dbc
fi
test "$(git -C "$STUDY_HOME/green-mbpt" rev-parse HEAD)" = 2346c87763e20cea5c55b95ab769ee55421f7386
test "$(git -C "$STUDY_HOME/green-gpu" rev-parse HEAD)" = 4dcb075a9084f801dfb04a513112a736f4423dbc
cmake -S "$STUDY_HOME/green-mbpt" -B "$STUDY_HOME/build" \
  -DCMAKE_BUILD_TYPE=Release -DBuild_Tests=OFF \
  -DCMAKE_INSTALL_PREFIX="$STUDY_HOME/install" \
  -DCUSTOM_KERNELS=https://github.com/Green-Phys/green-gpu \
  -DFETCHCONTENT_SOURCE_DIR_GREEN-GPU="$STUDY_HOME/green-gpu" \
  -DCUDAToolkit_ROOT="$CUDA_HOME" -DGPU_ARCHS=89
cmake --build "$STUDY_HOME/build" --target mbpt.exe -j2
"$FIX_PYTHON" "$STUDY_HOME/scripts/record_build.py"
echo ISOLATED_MAIN_BUILD_COMPLETE
