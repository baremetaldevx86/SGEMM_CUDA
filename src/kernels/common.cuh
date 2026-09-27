#pragma once

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

// Kept for compatibility with the host launchers. Each kernel header can now
// be included on its own, without depending on kernel 10's include order.
#ifndef CEIL_DIV
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#endif
constexpr int WARPSIZE = 32; // CUDA's warpSize is not a constant expression.

namespace sgemm_detail {

// Unlike assert(), this guard is active in release builds too. These are
// block-uniform preconditions, checked before any shared-memory barrier or
// global-memory access. A failed launch is reported at CUDA synchronization;
// it must never silently return an incomplete result.
__device__ __forceinline__ void require(bool condition, const char *message) {
  if (!condition) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&
        blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
      printf("SGEMM kernel precondition failed: %s\n", message);
    }
    __trap();
  }
}

template <int BM, int BN, int BK, int NUM_THREADS, bool VECTORIZED = false,
          bool ROW_FIRST_GRID = false>
__device__ __forceinline__ void require_tiled_launch(
    int M, int N, int K, const float *A, const float *B, const float *C) {
  static_assert(BM > 0 && BN > 0 && BK > 0, "Tile sizes must be positive");
  static_assert(NUM_THREADS > 0 && NUM_THREADS <= 1024,
                "Invalid thread-block size");
  static_assert(!VECTORIZED || (BN % 4 == 0 && BK % 4 == 0),
                "Vectorized tiles require BN and BK divisible by four");
  require(M > 0 && N > 0 && K > 0 && M % BM == 0 && N % BN == 0 &&
              K % BK == 0,
          "Tiled kernels require positive, tile-aligned M, N, and K");
  require(blockDim.x == NUM_THREADS && blockDim.y == 1 && blockDim.z == 1,
          "Tiled kernel launched with the wrong block dimensions");
  require(gridDim.x == (ROW_FIRST_GRID ? M / BM : N / BN) &&
              gridDim.y == (ROW_FIRST_GRID ? N / BN : M / BM) && gridDim.z == 1,
          "Tiled kernel launched with the wrong grid dimensions");
  constexpr unsigned alignment = VECTORIZED ? sizeof(float4) : alignof(float);
  require(A && B && C && reinterpret_cast<std::uintptr_t>(A) % alignment == 0 &&
              reinterpret_cast<std::uintptr_t>(B) % alignment == 0 &&
              reinterpret_cast<std::uintptr_t>(C) % alignment == 0,
          "Tiled kernel requires non-null, suitably aligned matrix pointers");
}

// Kernels 6-8 issue exactly one float4 load per thread for each operand.
// Their legacy 64x64 launch specialization compiles, but loads only half a
// tile. Reject it explicitly rather than consuming uninitialized shared memory.
template <int BM, int BN, int BK, int TM, int TN>
__device__ __forceinline__ void require_single_vector_load() {
  static_assert(TM > 0 && TN > 0 && BM % TM == 0 && BN % TN == 0,
                "Per-thread tiles must divide the block tile");
  static_assert(TN % 4 == 0, "Vectorized output requires TN divisible by four");
  constexpr int threads = (BM * BN) / (TM * TN);
  require(BM * BK == 4 * threads && BN * BK == 4 * threads,
          "Kernels 6-8 require exactly one float4 load per thread per tile; "
          "the legacy 64x64 specialization is unsupported");
}

template <int BM, int BN, int BK, int LOAD_THREADS>
__device__ __forceinline__ void validate_vector_loads() {
  static_assert(BK > 0 && BN > 0 && BK % 4 == 0 && BN % 4 == 0,
                "Vector loads require positive multiples of four");
  static_assert(LOAD_THREADS > 0 && (LOAD_THREADS * 4) % BK == 0 &&
                    (LOAD_THREADS * 4) % BN == 0,
                "Vector load strides must cover complete rows");
  static_assert((BM * BK) % (4 * LOAD_THREADS) == 0 &&
                    (BN * BK) % (4 * LOAD_THREADS) == 0,
                "Vector load iterations must cover complete tiles");
}

template <int BM, int BN, int BK, int WM, int WN, int WNITER, int TM, int TN,
          int NUM_THREADS, int LOAD_THREADS = NUM_THREADS>
__device__ __forceinline__ void validate_warp_tiling() {
  static_assert(WM > 0 && WN > 0 && WNITER > 0 && TM > 0 && TN > 0,
                "Warp and thread tile sizes must be positive");
  static_assert(NUM_THREADS % WARPSIZE == 0 && BM % WM == 0 && BN % WN == 0 &&
                    (BM / WM) * (BN / WN) == NUM_THREADS / WARPSIZE,
                "Warps must exactly cover the block tile");
  static_assert((WM * WN) % (WARPSIZE * TM * TN * WNITER) == 0,
                "Threads must exactly cover the warp tile");
  constexpr int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  static_assert(WMITER > 0 && WM % WMITER == 0 && WN % WNITER == 0,
                "Warp subtiles must divide the warp tile");
  static_assert((WM / WMITER) % TM == 0 && (WN / WNITER) % TN == 0 &&
                    TN % 4 == 0,
                "Per-thread tiles must cover vector-aligned warp subtiles");
  validate_vector_loads<BM, BN, BK, LOAD_THREADS>();
}

} // namespace sgemm_detail
