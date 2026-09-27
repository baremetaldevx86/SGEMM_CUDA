# Kernel launch contracts

Kernels 3–12 are **full-tile implementations**, not general-shape GEMM kernels.
The public `run_kernel()` dispatcher checks the current presets on the host and
throws `std::invalid_argument` for unsupported shapes or null/misaligned pointers
before launching. It launches an actual numeric ID, not the auto selector.
These checks mirror the current fixed runner presets and must be updated if
those presets change.

## Host registry and selection

[`../kernel_registry.h`](../kernel_registry.h) provides a CUDA-independent
registry and preflight policy. `kernel_registry()` lists descriptors and
`find_kernel()` accepts an existing numeric ID or exact, case-sensitive name.
The names in ID order are:

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

`sgemm --list-kernels` is a standalone, host-only listing: it needs neither a GPU
nor CUDA initialization. `sgemm --kernel NAME` and `sgemm NAME` also accept numeric
IDs and `auto`; unknown selectors fail. `auto` is handled by `select_kernel()`
using `kAutoKernel`, not by `find_kernel()` or `run_kernel()`.

`problem_shape_error()` requires positive M/N/K and MK, KN, and MN element counts
within `INT_MAX`. `kernel_shape_error()` checks only shape compatibility, without
CUDA access. `kernel_support_error()` adds checks against supplied device
capabilities: declared minimum compute capability, launch threads per block,
static shared memory, and grid limits. Kernel 12 is gated to CC≥8.0 and has a
conservative allowance for its shared barrier objects. cuBLAS is eligible for
all valid problem shapes; its internal resource decisions are not treated as
custom-kernel launch requirements.

`select_kernel()` returns the actual selected ID and a reason. Explicit IDs must
pass preflight or fail; they are never silently substituted. Auto is a
**deterministic first-eligible heuristic, 10 → 6 → 5 → 0**, not runtime tuning or a
fastest-kernel guarantee. Kernel 9's `autotuned` label is historical: its launch
preset is fixed. Kernels 11 and 12 are excluded from auto pending further audits,
but remain available explicitly. Invalid problem dimensions cannot be repaired
by fallback. There is no retry/fallback after a CUDA launch or execution error.

The CLI queries device capabilities once and does selection outside timing.
These metadata checks cannot establish that the binary has executable device
code for the GPU, that pointers and allocations are valid, that the kernels are
race-free, or that a configuration is performant. CUDA/runtime checks still
apply; build architecture and toolkit choices still matter.

## Workloads and launch validation

Without dimension flags the benchmark retains its square sweep at 128, 256,
512, 1024, 2048, and 4096. `--m M --n N --k K` specifies exactly one row-major
A(M×K) × B(K×N) → C(M×N) workload. All three flags are required together and
must occur once each; invalid tuples fail. Unsupported explicit single workloads
fail preflight. The legacy sweep skips unsupported sizes (notably kernel 11 at
128×128), prints a reason, and emits no timing/CSV row for skipped cases. A sweep
with no eligible workloads fails. Only selecting `auto` enables policy fallback.

Console output identifies requested and actual selected ID/name and reason.
CSV retains its first ten columns and appends
`m,n,k,kernel_name,requested_kernel,selection_reason`: `kernel` is the actual ID,
`requested_kernel` preserves the original token, and text is CSV-escaped. `size`
is blank unless M=N=K. Non-cube console timings also show all three dimensions;
legacy square timing lines retain their format. The plotting tool rejects
non-cube CSV workloads; see [output details](../../README.md#output-and-plotting).

For callers launching kernel templates directly, their entry guards in `common.cuh` are active even with `NDEBUG`: invalid
shapes, block/grid dimensions, null pointers, or insufficient pointer alignment
execute a device trap before matrix access or synchronization. The guard also
attempts a precondition diagnostic, but a trap can prevent device printf output
from being flushed. The caller must check CUDA launch **and synchronization**
results. This is an asynchronous failure, not a successful no-op or fallback;
a device trap may leave the CUDA context unusable.

For the current runner specializations, the supported positive dimensions are:

| Kernel | M multiple | N multiple | K multiple | Notes |
|---|---:|---:|---:|---|
| 3 | 32 | 32 | 32 | Grid x indexes rows, grid y indexes columns |
| 4 | 64 | 64 | 8 | |
| 5 | 128 or 64 | 128 or 64 | 8 | Uses 128 when both M and N are at least 128; otherwise 64 |
| 6–8 | 128 | 128 | 8 | Legacy 64×64 specialization is explicitly rejected |
| 9–10 | 128 | 128 | 16 | |
| 11 | 128 | 256 | 16 | |
| 12 | 128 | 128 | 16 | |

The grid must exactly cover the output tiles, with z=1. Blocks are one-dimensional
and must have the specialization's expected thread count. Kernels 6–12 require
16-byte-aligned A, B, and C pointers; kernels 3–5 require float alignment.
Compile-time checks also enforce the tiled loaders' and warp mappings' structural
requirements. Shared arrays accessed through `float4` have explicit 16-byte
alignment. `WARPSIZE` and the compatibility `CEIL_DIV` macro are defined in the
common header, so no kernel depends on kernel 10 being included first.

## Deliberately unresolved

- No edge masking or padding was added to tiled kernels. Partial M/N/K tiles
  are still rejected by kernels 3–12; only the separate `auto` policy may select
  another eligible implementation. Explicit selection does not fall back.
  Kernels 1–2 already mask output edges. K=0 and zero-sized GEMMs are rejected
  for every registry entry, including cuBLAS.
- Kernels 6–8 perform one vector load per thread. Their existing 64×64
  specializations load only half of each shared tile; kernel 7 also hard-codes
  a layout for BN=128 and TN=8 and can index outside the 64-column shared tile.
  The guards prevent executing these configurations rather than repairing the
  loading algorithms. Kernel 7 rejects other BN/TN layouts as well.
- Guards cannot validate device allocations, allocation sizes, aliasing, or
  pointer accessibility. The public dispatcher rejects matrix element counts
  exceeding `INT_MAX`, but direct template callers must ensure their index
  arithmetic fits. This is not a large-index or arbitrary-grid repair.
- Kernels still unconditionally read C even when beta is zero, and do not
  implement a special alpha=0 path. Callers must supply initialized storage;
  BLAS-style no-read semantics and exceptional-value handling are not fixed.
- Double-buffered synchronization/pipeline behavior is unchanged. In particular,
  kernel 11 retains block barriers in distinct thread-group branches. Aligned
  GPU numerical smoke tests passed for all twelve kernels, including odd/even
  K-tile counts for kernels 11–12, but broader tests and Compute Sanitizer
  (including synccheck/racecheck) remain necessary. The installed sanitizer
  could not instrument the available GPU, so sanitizer coverage is still absent.
