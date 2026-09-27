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
1. List the kernels without initializing CUDA or requiring a GPU:
   ```bash
   ./build/sgemm --list-kernels
   ```
   Select by numeric ID, exact name, or `auto`. `DEVICE=<device_id>` selects the
   CUDA device. Optional controls set warmup launches, timed iterations,
   reproducible inputs, and machine-readable output:
   ```bash
   ./build/sgemm 10 --warmup 5 --iters 50 --seed 1234 --csv benchmark.csv
   ./build/sgemm --kernel warp-tiled --alpha 1.0 --beta 0.0 --csv beta-zero.csv
   ./build/sgemm --kernel auto --m 256 --n 512 --k 128 --csv rectangular.csv
   ```
1. Profiling via [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) (ncu): `make profile KERNEL=<kernel number>`

## Kernel selection and workloads

The host-only registry preserves numeric IDs 0–12. Names are exact and
case-sensitive; `--list-kernels` lists names and declared requirements and is a
standalone command. A kernel is otherwise required, either positionally
(`sgemm vectorized`) or with `--kernel vectorized`. Unknown names and IDs fail.

| ID | Canonical name |
|---:|---|
| 0 | `cublas-fp32` |
| 1 | `naive` |
| 2 | `coalesced` |
| 3 | `shared` |
| 4 | `block-1d` |
| 5 | `block-2d` |
| 6 | `vectorized` |
| 7 | `bank-linearized` |
| 8 | `bank-padded` |
| 9 | `autotuned` |
| 10 | `warp-tiled` |
| 11 | `double-buffered` |
| 12 | `async-double-buffered` |

`--kernel auto` uses a **deterministic heuristic policy**, choosing the first
eligible kernel in **10 → 6 → 5 → 0** order for each workload. Eligibility checks
shape and declared hardware requirements. This is not runtime autotuning or a
claim to select the fastest kernel: it does not benchmark candidates. Kernel 9's
`autotuned` name is historical and denotes a fixed preset. Kernels 11 and 12 remain
explicitly selectable but are excluded from auto pending synchronization and
pipeline audits. Explicit selection never substitutes a different kernel; an
automatic fallback happens only with `auto`, not after CUDA execution failures.

With no dimension flags, the original square sweep is unchanged:
`M=N=K` at 128, 256, 512, 1024, 2048, and 4096. To run exactly one workload,
supply **all three** of `--m M --n N --k K`, each exactly once. A is row-major
M×K, B is K×N, and C is M×N. Dimensions must be positive integers and each
matrix's element count (MK, KN, MN) must fit `INT_MAX`. Missing, repeated, or
invalid dimension flags fail; they do not silently restore the square sweep or
trigger auto fallback.

For an explicit single workload, an unsupported kernel/shape/device combination
fails during preflight. The legacy sweep instead reports and skips unsupported
sizes, without fake timing/CSV rows; if no sizes can run, the command fails. Auto
can fall back to cuBLAS for valid shapes not supported by its custom candidates.
This does not add partial-tile support to any custom kernel.

Preflight uses device metadata (compute capability, maximum threads per block,
per-block shared memory, and grid limits), queried once outside benchmark timing.
Kernel 12 requires compute capability 8.0 or newer under this supported policy.
These checks are conservative declared requirements, not an occupancy/performance
model or a guarantee of executable-code compatibility. They cannot prove pointer
accessibility, allocation sizes, alias safety, or synchronization correctness.
Compiled architecture/toolkit compatibility and runtime errors remain CUDA's
responsibility; vendor-internal cuBLAS resources are not checked as custom-kernel
resources. See [kernel launch contracts](src/kernels/README.md).

## Benchmark correctness and supported shapes

- A, B, and the original C are generated once from `--seed`. Original C is
  immutable and distinct from both the cuBLAS reference and working output.
  The legacy sweep retains maximum-sized input buffers; an explicit workload
  allocates the needed MK, KN, and MN elements.
- Before **every** validation, warmup, and timed launch, the working C is reset.
  Each timed launch has its own CUDA event pair; reset copies are queued before
  the start event and excluded from reported time. This is a reset-per-launch
  benchmark, not a sustained back-to-back throughput measurement. The reset also
  affects cache state, so compare kernels under the same policy.
- `--alpha` / `--beta` accept finite floats, defaulting to `0.5` / `3.0`.
  The CSV `verified` field is true only after a custom output passes the cuBLAS
  comparison; kernel 0 is the reference and reports false rather than claiming
  independent verification, including when auto selects it.
- Kernels 3–12 require full tiles; partial dimensions are **not** implemented.
  Unsupported presets are rejected on the host. The default sweep skips kernel
  11 at size 128 because its column tile is 256; skips print a reason and produce
  no timing row. Broken 64×64 variants of kernels 6–8 are rejected, not silently
  replaced. See [kernel launch contracts](src/kernels/README.md).
- Errors from launches, synchronization, events, memory operations, and cuBLAS
  are checked. General-shape support for tiled custom kernels, BLAS-style beta=0
  no-read semantics, and broader precision-aware validation remain follow-up work.

## Output and plotting

For each workload the console reports the requested selector, actual selected
kernel ID/name, and selection reason. The legacy square timing line retains its
`size: (...)` format. Non-cube timing lines instead identify M, N, and K rather
than labeling M as a square matrix size. Selection and preflight are outside the
timed region.

`--csv FILE` writes one row per completed workload. Its first ten columns retain
the existing order; the six shape/selection columns are appended:

```text
kernel,size,average_seconds,gflops,warmup,iters,seed,verified,alpha,beta,m,n,k,kernel_name,requested_kernel,selection_reason
```

- `kernel` is the **actual selected numeric ID**, not an auto sentinel;
  `kernel_name` is its canonical registry name.
- `requested_kernel` preserves the user's original token (ID, name, or `auto`);
  `selection_reason` explains the explicit choice or deterministic policy
  outcome. Text fields are CSV-escaped, including reasons containing commas.
- `m`, `n`, and `k` always describe the workload. `size` is populated only when
  **M=N=K**; it is blank for all other workloads, even when the output C is square
  but the inner dimension K differs.

`plot_benchmark_results.py` remains a **square-sweep** plotting tool. It accepts
older CSVs without M/N/K columns and legacy square text logs. For newer CSVs it
requires all three dimensions, validates that they match `size`, and explicitly
rejects any rectangular/non-cube row rather than silently dropping or mislabeling
it. Use a separate square-only dataset to plot; rectangular visualization is not
implemented. The script reads `benchmark_results/*_output.csv` preferentially
and otherwise falls back to text logs. Running it rewrites `benchmark_results.png`
and the README's marked historical table; it is not run automatically by tests.

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
The pure host registry suite checks names, shape/device eligibility, and the
explicit-versus-auto policy with synthetic device capabilities and no GPU.
Python 3 enables benchmark lifecycle/CLI, bank-calculator, plot-parser, and
real-executable CLI end-to-end regression tests. The plot-parser tests require
pandas, matplotlib, and seaborn and skip with exit 77 when these optional
dependencies are absent; they do not generate plots or modify benchmark
artifacts. The CLI end-to-end test runs the real `sgemm` binary, checks named and
auto selection, rectangular CSV metadata, cuBLAS fallback, explicit preflight
failure, and host-only listing. It returns 77 when CUDA hardware/driver is
unavailable. They can also be run directly:
`python3 tests/test_plot_results.py` or
`python3 tests/test_cli_e2e.py build/sgemm`.
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
