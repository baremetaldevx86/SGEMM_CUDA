# Fast CUDA SGEMM from Scratch

Step-by-step optimization of matrix multiplication, implemented in CUDA.
For an explanation of each kernel, see [siboehm.com/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM).

## Overview

Historical results on a NVIDIA A6000 (Ampere), from the original accumulated-C
benchmark. The repaired benchmark resets C before each launch, so new timings
are **not directly comparable** with this table/image:

![](benchmark_results.png)

GFLOPs at matrix size 4096x4096:
<!-- benchmark_results -->
| Kernel                              |  GFLOPs/s | Performance relative to cuBLAS |
|:------------------------------------|----------:|:-------------------------------|
| 1: Naive                            |   `309.0` | 1.3%                           |
| 2: GMEM Coalescing                  |  `1986.5` | 8.5%                           |
| 3: SMEM Caching                     |  `2980.3` | 12.8%                          |
| 4: 1D Blocktiling                   |  `8474.7` | 36.5%                          |
| 5: 2D Blocktiling                   | `15971.7` | 68.7%                          |
| 7: Avoid Bank Conflicts (Linearize) | `16213.4` | 69.7%                          |
| 8: Avoid Bank Conflicts (Offset)    | `16459.2` | 70.8%                          |
| 11: Double Buffering                | `17278.3` | 74.3%                          |
| 6: Vectorized Mem Access            | `18237.3` | 78.4%                          |
| 9: Autotuning                       | `19721.0` | 84.8%                          |
| 10: Warptiling                      | `21779.3` | 93.7%                          |
| 0: cuBLAS                           | `23249.6` | 100.0%                         |
<!-- benchmark_results -->

## Setup

1. Install dependencies: CUDA toolkit 12, Python (+ Seaborn), CMake, Ninja. See [environment.yml](environment.yml).
1. Configure your GPU's [compute capability](https://developer.nvidia.com/cuda-gpus)
   through CMake rather than editing source. The default remains `86` (Ampere):
   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
   cmake --build build --parallel
   ```
   For multiple architectures use, for example, `-DCMAKE_CUDA_ARCHITECTURES='80;86'`.
   `CUDAARCHS` is honored on a fresh configuration. `native` is available with
   CMake 3.24+ when the GPU is visible and supported by your installed toolkit.
   Older toolkits cannot generate native code for newer architectures; upgrade
   the toolkit rather than assuming driver PTX compatibility gives native code.
   The Makefile equivalent is `make CUDA_ARCHITECTURES=86`.
1. Run one of the kernels: `DEVICE=<device_id> ./build/sgemm <kernel number>`
   Optional benchmark controls are available for warmup launches, timed
   iterations, reproducible inputs, and machine-readable output:
   ```bash
   ./build/sgemm 10 --warmup 5 --iters 50 --seed 1234 --csv benchmark.csv
   ./build/sgemm --kernel 10 --alpha 1.0 --beta 0.0 --csv beta-zero.csv
   ```
1. Profiling via [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) (ncu): `make profile KERNEL=<kernel number>`

## Benchmark correctness and supported shapes

- A, B, and the original C are generated once from `--seed`. Original C is
  immutable and distinct from both the cuBLAS reference and working output.
- Before **every** validation, warmup, and timed launch, the working C is reset.
  Each timed launch has its own CUDA event pair; reset copies are queued before
  the start event and excluded from reported time. This is a reset-per-launch
  benchmark, not a sustained back-to-back throughput measurement. The reset also
  affects cache state, so compare kernels under the same policy.
- `--alpha` / `--beta` accept finite floats, defaulting to `0.5` / `3.0`.
  Both are appended to CSV after the existing fields. The `verified` field is
  true only after a custom output passes the cuBLAS comparison; kernel 0 is the
  reference and reports false rather than claiming independent verification.
- Kernels 3–12 require full tiles; partial dimensions are **not** implemented.
  Unsupported presets are rejected on the host. The default sweep skips kernel
  11 at size 128 because its column tile is 256; skips print a reason and produce
  no timing row. Broken 64×64 variants of kernels 6–8 are rejected, not silently
  replaced. See [kernel launch contracts](src/kernels/README.md).
- Errors from launches, synchronization, events, memory operations, and cuBLAS
  are checked. General-shape support, BLAS-style beta=0 no-read semantics, and
  broader precision-aware validation remain follow-up work.

## Tests

CTest targets are built by default (`-DBUILD_TESTING=OFF` disables them):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
# Run only host-side tests; a CUDA toolkit is needed to build the full project.
ctest --test-dir build -L cpu --output-on-failure
# Run the independent CPU-reference GPU suite only.
ctest --test-dir build -L gpu --output-on-failure
```

The runner suite checks shape/alignment rejection and helper functions, then
compares all 13 kernel IDs against a CPU double-precision reference on supported
square/rectangular shapes. It also covers irregular shapes for IDs 0–2, small
kernel-5 tiles, pipeline boundaries for IDs 11–12, and five alpha/beta pairs.
GPU tests are reported as **skipped** (exit 77) only if no device or usable driver
is available; numerical and execution failures are errors. Host tests still run.
Python 3 enables benchmark lifecycle/CLI and bank-calculator regression tests.
The benchmark mock can also be [built without CUDA](tests/benchmark_mock/README.md).

For a supported GPU/toolchain, additionally run Compute Sanitizer:

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/sgemm_runner_tests --gpu
compute-sanitizer --tool racecheck --error-exitcode 1 ./build/sgemm_runner_tests --gpu
compute-sanitizer --tool synccheck --error-exitcode 1 ./build/sgemm_runner_tests --gpu
```

Numerical tests do not establish race-freedom. Kernel 11's branch-local barriers
and kernel 12's asynchronous pipeline still need sanitizer validation on supported
hardware. Kernel 12 also emits CUDA 12.4 shared-barrier initialization warnings;
its barrier objects are explicitly initialized before use.

Credit goes to [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE) for the benchmarking setup.
