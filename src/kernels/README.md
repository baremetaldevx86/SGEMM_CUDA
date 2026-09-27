# Kernel launch contracts

Kernels 3–12 are **full-tile implementations**, not general-shape GEMM kernels.
The public `run_kernel()` dispatcher checks the current presets on the host and
throws `std::invalid_argument` for unsupported shapes or null/misaligned pointers
before launching. `kernel_shape_error()` exposes the shape check without touching
CUDA. The benchmark skips unsupported sizes (notably kernel 11 at 128×128),
prints a reason, and emits no CSV row for the skipped case. It never substitutes
another kernel under the selected ID. These checks mirror the current fixed
runner presets and must be updated if those presets change.

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

- No edge masking, padding, or automatic fallback was added. Partial M/N/K
  tiles, K=0, and zero-sized GEMMs are rejected by kernels 3–12. Use a suitable
  general-purpose implementation instead; kernels 1–2 already mask output
  edges, but their contracts were not changed in this repair.
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
