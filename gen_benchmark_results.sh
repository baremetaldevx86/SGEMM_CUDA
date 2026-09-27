#!/usr/bin/env bash

set -euo pipefail

# This script runs the ./sgemm binary for all existing kernels and logs
# both human-readable and CSV output in benchmark_results/.

mkdir -p benchmark_results

WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-50}"
SEED="${SEED:-1234}"
SGEMM_BIN="${SGEMM_BIN:-./build/sgemm}"

for kernel in {0..12}; do
    echo ""
    "$SGEMM_BIN" "$kernel" --warmup "$WARMUP" --iters "$ITERS" \
        --seed "$SEED" --csv "benchmark_results/${kernel}_output.csv" \
        | tee "benchmark_results/${kernel}_output.txt"
    sleep 2
done

python3 plot_benchmark_results.py
