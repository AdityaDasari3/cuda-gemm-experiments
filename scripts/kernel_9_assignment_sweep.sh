#!/usr/bin/env bash

set -u

cd "$(dirname "$0")"
cd "../build"

RUNNER="../src/runner.cu"
KERNEL="../src/kernels/9_kernel_autotuned.cuh"
OUTPUT="../benchmark_results/kernel_9_assignment_sweep.txt"

mkdir -p ../benchmark_results
echo "" > "$OUTPUT"

export DEVICE="0"

# Keep NT fixed so we can isolate tiling effects
NUM_THREADS=256

# Format:
# BK BM BN TM TN
CONFIGS=(
  "8   128 128 8 8"
  "16  128 128 8 8"
  "32  128 128 8 8"
  "64  128 128 8 8"

  "16  64  64  4 4"
  "16  128 128 4 4"
  "16  128 128 8 8"
  "16  256 256 16 16"

  "16  128 128 4 8"
  "16  128 128 8 4"
)

CONFIG_NUM=0

for CONFIG in "${CONFIGS[@]}"; do
  read BK BM BN TM TN <<< "$CONFIG"

  CONFIG_NUM=$((CONFIG_NUM + 1))

  echo ""
  echo "===== CONFIG $CONFIG_NUM/${#CONFIGS[@]} ====="
  echo "BK=$BK BM=$BM BN=$BN TM=$TM TN=$TN NT=$NUM_THREADS"

  # Modify parameters
  sed -i "s/const uint K9_BK = .*/const uint K9_BK = $BK;/" "$RUNNER"
  sed -i "s/const uint K9_TM = .*/const uint K9_TM = $TM;/" "$RUNNER"
  sed -i "s/const uint K9_TN = .*/const uint K9_TN = $TN;/" "$RUNNER"
  sed -i "s/const uint K9_BM = .*/const uint K9_BM = $BM;/" "$RUNNER"
  sed -i "s/const uint K9_BN = .*/const uint K9_BN = $BN;/" "$RUNNER"
  sed -i "s/const int K9_NUM_THREADS = .*/const int K9_NUM_THREADS = $NUM_THREADS;/" "$KERNEL"

  # Build only sgemm
  if ! cmake --build . --target sgemm -j > /tmp/k9_build.log 2>&1; then
    echo "BUILD_FAIL BK=$BK BM=$BM BN=$BN TM=$TM TN=$TN" | tee -a "$OUTPUT"
    cat /tmp/k9_build.log
    continue
  fi

  echo "CONFIG BK=$BK BM=$BM BN=$BN TM=$TM TN=$TN NT=$NUM_THREADS" \
    | tee -a "$OUTPUT"

  # One fixed representative matrix size
  ./sgemm 9 2048 2048 2048 | tee -a "$OUTPUT"

done