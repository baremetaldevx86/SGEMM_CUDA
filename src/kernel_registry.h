#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <string_view>

// Registry and selection are host-only: listing kernels and testing the policy
// do not initialize CUDA or require a device. Numeric IDs remain stable.
constexpr int kAutoKernel = -1;

struct GemmShape {
  int m;
  int n;
  int k;
};

struct DeviceCapabilities {
  int compute_capability = 0; // e.g. 86 for sm_86
  int max_threads_per_block = 0;
  std::size_t shared_memory_per_block = 0;
  int max_grid_x = 0;
  int max_grid_y = 0;
};

struct KernelDescriptor {
  int id;
  const char *name;
  const char *description;
  const char *shape_requirements;
  int minimum_compute_capability; // 0: no extra feature gate beyond built CUDA code
  std::size_t pointer_alignment;
};

struct KernelSelection {
  int id;
  std::string reason;
};

const std::array<KernelDescriptor, 13> &kernel_registry();
const KernelDescriptor *find_kernel(int id);
// Accepts a canonical, case-sensitive name or a decimal numeric ID, not "auto".
const KernelDescriptor *find_kernel(std::string_view name_or_id);

// All dimensions must be positive and row-major element counts fit INT_MAX.
const char *problem_shape_error(GemmShape shape);
// Compatibility API: no CUDA/device query. nullptr indicates an eligible shape.
const char *kernel_shape_error(int kernel_id, int m, int n, int k);
// Empty string indicates supported shape and declared hardware requirements.
// Allocation sizes, pointer accessibility and compiled-code compatibility are
// still caller/build responsibilities; CUDA launch failures remain fatal.
std::string kernel_support_error(int kernel_id, GemmShape shape,
                                 const DeviceCapabilities &device);
// Explicit ID: never substitutes another implementation. Auto (-1): deterministic
// first-supported policy 10 -> 6 -> 5 -> cuBLAS, not measured/autotuned selection.
// Throws std::invalid_argument for an invalid problem or unsupported explicit ID.
KernelSelection select_kernel(int requested_id, GemmShape shape,
                              const DeviceCapabilities &device);
