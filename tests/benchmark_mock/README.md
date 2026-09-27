# CPU-only benchmark regression tests

These tests compile the **actual `sgemm.cu`** as C++, substituting this directory's
`runner.cuh` for the CUDA runner. No CUDA headers, CUDA libraries, `nvcc`, or GPU
are needed to build or execute this mock target. Python uses only its standard
library and accepts the already compiled executable as its positional argument.
All run outputs and validation logs are isolated in temporary directories.

From the repository root:

```sh
build_dir=$(mktemp -d)
g++ -std=c++17 -O2 -DNDEBUG -Wall -Wextra -Werror \
    -Itests/benchmark_mock tests/benchmark_mock/benchmark_main.cpp \
    -o "$build_dir/sgemm_benchmark_mock"
python3 tests/benchmark_mock/test_benchmark.py "$build_dir/sgemm_benchmark_mock"
rm -rf "$build_dir"
```

`benchmark_main.cpp` only includes `../../sgemm.cu`; it does not duplicate any
benchmark logic. Equivalently, compile `-x c++ sgemm.cu` with the same mock include
directory. The wrapper lets CMake use its C++ compiler without changing the
language of `sgemm.cu` in the real CUDA target. Mock checks use an explicit
failure function, **not `assert`**, so `-DNDEBUG` does not disable them. Python's
`unittest` checks also remain active under `python3 -O`.

## Coverage and limits

- Immutable host/device initial C (and inputs); reset before every validation,
  warmup, and timed launch; correct reset source and size at all six transitions.
- Reset copies outside event intervals on the same default stream; exactly one
  GEMM per timed interval; synchronization, launch-error checks, and cleanup.
- Per-iteration durations vary (`1, 2, ...` milliseconds), checking time
  accumulation, averaging, GFLOPs arithmetic, and exclusion of warmups.
- Positional/flag kernel selection, default/nondefault alpha and beta (also
  checked at each launch), seed, zero warmups, all CSV columns, and optional CSV.
- cuBLAS rows explicitly say `verified=false`, while custom kernels are checked
  against the mock reference before reporting `verified=true`.
- Kernel 11 rejects size 128; skips have no launches and no CSV row. A separate
  all-skipped case must produce only the CSV header and clean up successfully.
- Invalid CLI/DEVICE values, device-less help, unavailable devices, CUDA/cuBLAS
  failures (including later repeated calls), invalid elapsed times, validation
  failure logs preserving initial C, and CSV/log open and write failures.

This is a lifecycle model, **not numerical GEMM or real kernel-contract
validation**. Fake device allocations store one representative float and their
logical byte size. `kernel_shape_error` models the benchmark's fixed sizes and
kernel-11 rejection, rather than duplicating the production shape validator.
The real benchmark still allocates its five maximum-size host vectors, so allow
roughly **320 MiB** of memory per subprocess. Cases run serially, with a 60-second
subprocess timeout. `/dev/full` write-error tests are skipped when that device
is unavailable; there are no fixed temporary-directory paths.

For manual fault injection, set `SGEMM_MOCK_FAIL=event_record` to fail that API,
or `SGEMM_MOCK_FAIL=event_record:2` to fail only its second call. The test driver
removes inherited `DEVICE` and `SGEMM_MOCK_*` settings before configuring each
case. Mock invariant violations exit 90, distinct from expected benchmark
failures (exit 1).

## Suggested parent CMake / CTest integration

Add this to the parent build configuration (these files do not change it):

```cmake
include(CTest)
if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_executable(sgemm_benchmark_mock
        tests/benchmark_mock/benchmark_main.cpp)
    target_compile_features(sgemm_benchmark_mock PRIVATE cxx_std_17)
    target_include_directories(sgemm_benchmark_mock PRIVATE
        ${PROJECT_SOURCE_DIR}/tests/benchmark_mock)
    add_test(NAME benchmark_mock
        COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/benchmark_mock/test_benchmark.py
            $<TARGET_FILE:sgemm_benchmark_mock>)
    set_tests_properties(benchmark_mock PROPERTIES TIMEOUT 180)
endif()
```

Do not link `src/runner.cu`, CUDA libraries, or add the production `src` include
directory to the mock target. Then run:

```sh
cmake --build build --target sgemm_benchmark_mock
ctest --test-dir build -R '^benchmark_mock$' --output-on-failure
```

The repository's top-level project still enables CUDA at configuration time;
use the standalone `g++` command above on a machine without a CUDA toolkit.
