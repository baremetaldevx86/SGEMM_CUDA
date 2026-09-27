#include "kernel_registry.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {
constexpr std::array<KernelDescriptor, 13> kKernels{{
    {0, "cublas-fp32", "cuBLAS row-major FP32 GEMM",
     "Any positive M/N/K within 32-bit element indexing", 0, alignof(float)},
    {1, "naive", "Naive one-output-per-thread GEMM",
     "Any positive M/N/K within 32-bit element indexing", 0, alignof(float)},
    {2, "coalesced", "Global-memory coalesced GEMM",
     "Any positive M/N/K within 32-bit element indexing", 0, alignof(float)},
    {3, "shared", "Shared-memory blocked GEMM",
     "M/N/K multiples of 32", 0, alignof(float)},
    {4, "block-1d", "One-dimensional register block tiling",
     "M/N multiples of 64; K a multiple of 8", 0, alignof(float)},
    {5, "block-2d", "Two-dimensional register block tiling",
     "M/N multiples of 128 when both >=128, otherwise multiples of 64; K a multiple of 8",
     0, alignof(float)},
    {6, "vectorized", "Vectorized memory loads and stores",
     "M/N multiples of 128; K a multiple of 8", 0, 16},
    {7, "bank-linearized", "Linearized shared-memory bank layout",
     "M/N multiples of 128; K a multiple of 8", 0, 16},
    {8, "bank-padded", "Padded shared-memory bank layout",
     "M/N multiples of 128; K a multiple of 8", 0, 16},
    {9, "autotuned", "Historical autotuned preset (not a runtime tuner)",
     "M/N multiples of 128; K a multiple of 16", 0, 16},
    {10, "warp-tiled", "Warp-level register tiling",
     "M/N multiples of 128; K a multiple of 16", 0, 16},
    {11, "double-buffered", "Double-buffered preset (explicit only; audit pending)",
     "M a multiple of 128; N a multiple of 256; K a multiple of 16", 0, 16},
    {12, "async-double-buffered",
     "Asynchronous double-buffered preset (explicit only; audit pending)",
     "M/N multiples of 128; K a multiple of 16", 80, 16},
}};

// Declared requirements for the fixed specializations in runner.cu. These are
// host metadata, not CUDA function-attribute queries. Kernel 5's small preset is
// adjusted below. Registers and compiled-code compatibility remain CUDA's job.
struct LaunchRequirements {
  int id;
  int tile_m;
  int tile_n;
  int threads;
  std::size_t shared_bytes;
  bool grid_x_is_m;
};
constexpr std::array<LaunchRequirements, 13> kLaunchRequirements{{
    {0, 0, 0, 0, 0, false}, // cuBLAS owns its launch/resource decisions
    {1, 32, 32, 1024, 0, true},
    {2, 32, 32, 1024, 0, true},
    {3, 32, 32, 1024, 2 * 32 * 32 * sizeof(float), true},
    {4, 64, 64, 512, (64 * 8 + 8 * 64) * sizeof(float), false},
    {5, 128, 128, 256, (128 * 8 + 8 * 128) * sizeof(float), false},
    {6, 128, 128, 256, (128 * 8 + 8 * 128) * sizeof(float), false},
    {7, 128, 128, 256, (128 * 8 + 8 * 128) * sizeof(float), false},
    {8, 128, 128, 256, (128 * 8 + 8 * (128 + 5)) * sizeof(float), false},
    {9, 128, 128, 256, (128 * 16 + 16 * 128) * sizeof(float), false},
    {10, 128, 128, 128, (128 * 16 + 16 * 128) * sizeof(float), false},
    {11, 128, 256, 256, 2 * (128 * 16 + 16 * 256) * sizeof(float), false},
    // Reserve 128 extra bytes conservatively for both CUDA barriers and their
    // alignment/padding, without importing CUDA types into this pure host file.
    {12, 128, 128, 128, 2 * (128 * 16 + 16 * 128) * sizeof(float) + 128, false},
}};

constexpr bool registry_ids_match() {
  for (std::size_t i = 0; i < kKernels.size(); ++i) {
    if (kKernels[i].id != static_cast<int>(i) ||
        kLaunchRequirements[i].id != kKernels[i].id) {
      return false;
    }
  }
  return true;
}
static_assert(registry_ids_match(), "Registry and launch metadata IDs must agree");

int ceil_tiles(int dimension, int tile) {
  // Validated positive dimensions; no dimension + tile - 1 signed overflow.
  return dimension / tile + (dimension % tile != 0);
}

std::string kernel_label(const KernelDescriptor &kernel) {
  return "kernel " + std::to_string(kernel.id) + " (" + kernel.name + ")";
}
} // namespace

const std::array<KernelDescriptor, 13> &kernel_registry() { return kKernels; }

const KernelDescriptor *find_kernel(int id) {
  return id >= 0 && id < static_cast<int>(kKernels.size()) ? &kKernels[id]
                                                        : nullptr;
}

const KernelDescriptor *find_kernel(std::string_view name_or_id) {
  for (const auto &kernel : kKernels) {
    if (name_or_id == kernel.name) {
      return &kernel;
    }
  }
  if (name_or_id.empty()) {
    return nullptr;
  }
  int id = 0;
  for (const char digit : name_or_id) {
    if (digit < '0' || digit > '9') {
      return nullptr;
    }
    id = id * 10 + (digit - '0');
    // Check each digit: even arbitrarily long strings cannot overflow id.
    if (id >= static_cast<int>(kKernels.size())) {
      return nullptr;
    }
  }
  return find_kernel(id);
}

const char *problem_shape_error(GemmShape shape) {
  if (shape.m <= 0 || shape.n <= 0 || shape.k <= 0) {
    return "M, N, and K must be positive";
  }
  const auto max_index = std::numeric_limits<int>::max();
  if (static_cast<std::int64_t>(shape.m) * shape.n > max_index ||
      static_cast<std::int64_t>(shape.m) * shape.k > max_index ||
      static_cast<std::int64_t>(shape.k) * shape.n > max_index) {
    return "Matrix element counts exceed the supported 32-bit indexing range";
  }
  return nullptr;
}

const char *kernel_shape_error(int kernel_id, int M, int N, int K) {
  if (!find_kernel(kernel_id)) {
    return "Unknown kernel number (expected 0-12)";
  }
  if (const char *error = problem_shape_error({M, N, K})) {
    return error;
  }
  // Mirrors the fixed runner specializations, not an automatic selector.
  switch (kernel_id) {
  case 0:
  case 1:
  case 2:
    return nullptr;
  case 3:
    return M % 32 || N % 32 || K % 32
               ? "Kernel 3 requires M/N/K multiples of 32" : nullptr;
  case 4:
    return M % 64 || N % 64 || K % 8
               ? "Kernel 4 requires M/N multiples of 64 and K a multiple of 8"
               : nullptr;
  case 5: {
    const int tile = M >= 128 && N >= 128 ? 128 : 64;
    return M % tile || N % tile || K % 8
               ? "Kernel 5 requires full 128x128 (or small 64x64) tiles and K a multiple of 8"
               : nullptr;
  }
  case 6:
  case 7:
  case 8:
    return M % 128 || N % 128 || K % 8
               ? "Kernels 6-8 require M/N multiples of 128 and K a multiple of 8; small tiles are unsupported"
               : nullptr;
  case 11:
    return M % 128 || N % 256 || K % 16
               ? "Kernel 11 requires M a multiple of 128, N of 256, and K of 16"
               : nullptr;
  default: // 9, 10, 12
    return M % 128 || N % 128 || K % 16
               ? "Kernels 9/10/12 require M/N multiples of 128 and K a multiple of 16"
               : nullptr;
  }
}

std::string kernel_support_error(int kernel_id, GemmShape shape,
                                 const DeviceCapabilities &device) {
  if (const char *error = kernel_shape_error(kernel_id, shape.m, shape.n, shape.k)) {
    return error;
  }
  // Do not apply our custom-kernel resource assumptions to vendor internals.
  if (kernel_id == 0) {
    return {};
  }
  const auto &kernel = *find_kernel(kernel_id);
  if (device.compute_capability < kernel.minimum_compute_capability) {
    return "Requires compute capability >= " +
           std::to_string(kernel.minimum_compute_capability) +
           "; device reports " + std::to_string(device.compute_capability);
  }
  auto launch = kLaunchRequirements[kernel_id];
  if (kernel_id == 5 && (shape.m < 128 || shape.n < 128)) {
    launch.tile_m = launch.tile_n = 64;
    launch.threads = 64;
    launch.shared_bytes = (64 * 8 + 8 * 64) * sizeof(float);
  }
  if (launch.threads > device.max_threads_per_block) {
    return "Requires " + std::to_string(launch.threads) +
           " threads per block; device limit is " +
           std::to_string(device.max_threads_per_block);
  }
  if (launch.shared_bytes > device.shared_memory_per_block) {
    return "Requires " + std::to_string(launch.shared_bytes) +
           " bytes of static shared memory per block (conservative estimate); "
           "device limit is " + std::to_string(device.shared_memory_per_block);
  }
  const int tiles_m = ceil_tiles(shape.m, launch.tile_m);
  const int tiles_n = ceil_tiles(shape.n, launch.tile_n);
  const int grid_x = launch.grid_x_is_m ? tiles_m : tiles_n;
  const int grid_y = launch.grid_x_is_m ? tiles_n : tiles_m;
  if (grid_x > device.max_grid_x || grid_y > device.max_grid_y) {
    return "Requires grid (" + std::to_string(grid_x) + ", " +
           std::to_string(grid_y) + "); device grid limits are (" +
           std::to_string(device.max_grid_x) + ", " +
           std::to_string(device.max_grid_y) + ")";
  }
  return {};
}

KernelSelection select_kernel(int requested_id, GemmShape shape,
                              const DeviceCapabilities &device) {
  // An invalid problem must never become a successful cuBLAS fallback.
  if (const char *error = problem_shape_error(shape)) {
    throw std::invalid_argument(error);
  }
  if (requested_id != kAutoKernel) {
    const auto *kernel = find_kernel(requested_id);
    if (!kernel) {
      throw std::invalid_argument("Unknown kernel number (expected 0-12 or auto)");
    }
    const std::string error = kernel_support_error(requested_id, shape, device);
    if (!error.empty()) {
      throw std::invalid_argument("Explicit " + kernel_label(*kernel) +
                                  " is unsupported: " + error);
    }
    return {requested_id, "Explicit request accepted: " + kernel_label(*kernel) +
                              "; shape and declared device requirements satisfied"};
  }

  constexpr std::array<int, 4> policy{{10, 6, 5, 0}};
  std::string skipped;
  for (const int id : policy) {
    const auto &kernel = *find_kernel(id);
    const std::string error = kernel_support_error(id, shape, device);
    if (error.empty()) {
      return {id, "Auto heuristic policy 10 -> 6 -> 5 -> 0 selected " +
                      kernel_label(kernel) +
                      (id == 0 ? " (cuBLAS fallback)" : " (first eligible candidate)") +
                      "; not measured/autotuned" + skipped};
    }
    skipped += "; skipped " + kernel_label(kernel) + ": " + error;
  }
  throw std::logic_error("Auto policy has no eligible candidate for a valid problem");
}
