// Pure C++ registry/selection tests: no CUDA headers, libraries, or device.
// Checks deliberately throw rather than use assert(), so Release tests count.
#include "../src/kernel_registry.h"

#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
class Checks {
public:
  void require(bool condition, const std::string &message) {
    if (!condition) {
      throw std::runtime_error(message);
    }
    ++count_;
  }
  std::size_t count() const { return count_; }

private:
  std::size_t count_ = 0;
};

template <typename Function>
void expect_invalid_argument(Checks &checks, const std::string &label,
                             Function function) {
  try {
    function();
  } catch (const std::invalid_argument &error) {
    checks.require(error.what()[0] != '\0', label + ": empty diagnostic");
    return;
  } catch (const std::exception &error) {
    throw std::runtime_error(label + ": wrong exception: " + error.what());
  }
  throw std::runtime_error(label + ": expected std::invalid_argument");
}

DeviceCapabilities ample_device() {
  return {86, 1024, 64 * 1024, std::numeric_limits<int>::max(),
          std::numeric_limits<int>::max()};
}

std::string label(int id, GemmShape shape) {
  return "kernel " + std::to_string(id) + " M=" + std::to_string(shape.m) +
         " N=" + std::to_string(shape.n) + " K=" + std::to_string(shape.k);
}

void check_support(Checks &checks, int id, GemmShape shape,
                   const DeviceCapabilities &device, bool supported) {
  const std::string error = kernel_support_error(id, shape, device);
  checks.require(error.empty() == supported,
                 label(id, shape) + (supported ? ": unexpectedly rejected: "
                                              : ": unexpectedly supported") + error);
  if (supported) {
    const auto selected = select_kernel(id, shape, device);
    checks.require(selected.id == id, label(id, shape) + ": explicit ID substituted");
    checks.require(!selected.reason.empty(), label(id, shape) + ": no selection reason");
  } else {
    expect_invalid_argument(checks, label(id, shape), [&] {
      (void)select_kernel(id, shape, device);
    });
  }
}

void check_shape(Checks &checks, int id, GemmShape shape, bool supported) {
  const char *error = kernel_shape_error(id, shape.m, shape.n, shape.k);
  checks.require((error == nullptr) == supported,
                 label(id, shape) + ": unexpected shape eligibility");
  if (error) {
    checks.require(error[0] != '\0', label(id, shape) + ": empty shape diagnostic");
  }
  check_support(checks, id, shape, ample_device(), supported);
}

void test_registry(Checks &checks) {
  constexpr std::array<const char *, 13> names = {
      "cublas-fp32", "naive", "coalesced", "shared", "block-1d", "block-2d",
      "vectorized", "bank-linearized", "bank-padded", "autotuned", "warp-tiled",
      "double-buffered", "async-double-buffered"};
  const auto &registry = kernel_registry();
  checks.require(registry.size() == names.size(), "stable registry size");
  checks.require(kAutoKernel == -1, "stable auto request sentinel");
  for (std::size_t i = 0; i != names.size(); ++i) {
    const auto &entry = registry[i];
    checks.require(entry.id == static_cast<int>(i), "stable registry ID/order");
    checks.require(entry.name && std::string_view(entry.name) == names[i],
                   "stable canonical name for ID " + std::to_string(i));
    checks.require(entry.description && entry.description[0] != '\0',
                   "nonempty description for ID " + std::to_string(i));
    checks.require(entry.shape_requirements && entry.shape_requirements[0] != '\0',
                   "nonempty shape requirements for ID " + std::to_string(i));
    checks.require(entry.minimum_compute_capability == (i == 12 ? 80 : 0),
                   "declared compute capability for ID " + std::to_string(i));
    checks.require(entry.pointer_alignment == (i >= 6 ? 16U : alignof(float)),
                   "declared pointer alignment for ID " + std::to_string(i));
    checks.require(find_kernel(static_cast<int>(i)) == &entry, "integer lookup");
    checks.require(find_kernel(names[i]) == &entry, "canonical name lookup");
    checks.require(find_kernel(std::to_string(i)) == &entry, "decimal ID lookup");
    checks.require(find_kernel("00" + std::to_string(i)) == &entry,
                   "leading-zero decimal ID lookup");
    for (std::size_t j = 0; j < i; ++j) {
      checks.require(std::string_view(entry.name) != registry[j].name,
                     "unique canonical names");
    }
  }
  for (int id : {kAutoKernel, -2, 13, std::numeric_limits<int>::min(),
                 std::numeric_limits<int>::max()}) {
    checks.require(find_kernel(id) == nullptr, "unknown integer ID rejected");
  }
  for (std::string_view token : {
           "", "auto", "AUTO", "Naive", "CUBLAS-FP32", "warp_tiled", "unknown",
           "-1", "-0", "-2", "+1", "13", "99", "2147483648", "4294967296",
           "999999999999999999999999999999999999999", " 1", "1 ", "\t1",
           "1\n", "0xA", "1.0", "1e1", "10junk", "1 0", "warp-tiled "}) {
    checks.require(find_kernel(token) == nullptr,
                   "invalid string rejected: '" + std::string(token) + "'");
  }
  const char embedded_nul[] = {'1', '0', '\0', 'x'};
  checks.require(find_kernel(std::string_view(embedded_nul, sizeof(embedded_nul))) == nullptr,
                 "lookup must consume embedded-NUL input fully");
  const char bounded[] = {'x', '1', '0', 'x'};
  checks.require(find_kernel(std::string_view(bounded + 1, 2)) == find_kernel(10),
                 "lookup honors a non-NUL-terminated string_view");
}

void test_problem_validation(Checks &checks) {
  const int max = std::numeric_limits<int>::max();
  for (GemmShape shape : {GemmShape{1, 1, 1}, {7, 13, 5}, {max, 1, 1},
                         {1, max, 1}, {1, 1, max}, {46340, 46340, 1},
                         {32768, 65280, 32}}) {
    checks.require(problem_shape_error(shape) == nullptr, "valid problem rejected");
    for (int id : {0, 1, 2}) {
      check_shape(checks, id, shape, true);
    }
  }
  auto reject = [&](GemmShape shape) {
    const char *error = problem_shape_error(shape);
    checks.require(error && error[0] != '\0', "invalid problem lacks diagnostic");
    for (int id = 0; id <= 12; ++id) {
      check_shape(checks, id, shape, false);
    }
    // Neither abundant hardware nor an all-zero capability record may turn a
    // malformed/overflowing problem into a successful cuBLAS fallback.
    for (const auto &device : {ample_device(), DeviceCapabilities{}}) {
      expect_invalid_argument(checks, "invalid problem must not auto-fallback", [&] {
        (void)select_kernel(kAutoKernel, shape, device);
      });
      check_support(checks, 0, shape, device, false);
    }
  };
  for (int bad : {0, -1, std::numeric_limits<int>::min()}) {
    reject({bad, 128, 32});
    reject({128, bad, 32});
    reject({128, 128, bad});
  }
  // Overflow each independent row-major allocation count, including values
  // that would wrap signed or unsigned 32-bit multiplication.
  for (GemmShape shape : {GemmShape{46341, 46341, 1}, {46341, 1, 46341},
                         {1, 46341, 46341}, {65536, 32768, 32},
                         {65536, 256, 32768}, {128, 65536, 32768},
                         {65536, 65536, 65536}, {max, max, max}}) {
    reject(shape);
  }
  for (int id : {-2, 13, std::numeric_limits<int>::min(), max}) {
    check_shape(checks, id, {128, 256, 32}, false);
  }
  checks.require(kernel_shape_error(kAutoKernel, 128, 256, 32) != nullptr,
                 "auto is a selector request, not a launchable kernel ID");
  checks.require(!kernel_support_error(kAutoKernel, {128, 256, 32}, ample_device()).empty(),
                 "auto is not a descriptor for support queries");
}

void test_preset_shapes(Checks &checks) {
  // Public supported presets, not a reimplementation of the divisibility
  // logic. Each fixture tests independent M/N/K boundary violations.
  struct Preset { int id; GemmShape tile; };
  constexpr Preset presets[] = {
      {3, {32, 32, 32}}, {4, {64, 64, 8}}, {5, {128, 128, 8}},
      {6, {128, 128, 8}}, {7, {128, 128, 8}}, {8, {128, 128, 8}},
      {9, {128, 128, 16}}, {10, {128, 128, 16}},
      {11, {128, 256, 16}}, {12, {128, 128, 16}}};
  for (const auto &preset : presets) {
    const GemmShape shape = preset.tile;
    check_shape(checks, preset.id, shape, true);
    check_shape(checks, preset.id, {shape.m * 2, shape.n * 3, shape.k * 5}, true);
    for (int delta : {-1, 1}) {
      check_shape(checks, preset.id, {shape.m + delta, shape.n, shape.k}, false);
      check_shape(checks, preset.id, {shape.m, shape.n + delta, shape.k}, false);
      check_shape(checks, preset.id, {shape.m, shape.n, shape.k + delta}, false);
    }
    check_shape(checks, preset.id, {1, 1, 1}, false);
    check_shape(checks, preset.id, {7, 13, 5}, false);
  }
  for (GemmShape shape : {GemmShape{64, 64, 8}, {64, 192, 24}, {192, 64, 40}}) {
    check_shape(checks, 5, shape, true);
    for (int id : {6, 7, 8}) {
      check_shape(checks, id, shape, false);
    }
  }
  // Both dimensions >=128 select kernel 5's large tile, even for multiples
  // of 64. Kernel 11 has a different N tile than the other vectorized kernels.
  for (GemmShape shape : {GemmShape{192, 128, 8}, {128, 192, 8}, {32, 64, 8}}) {
    check_shape(checks, 5, shape, false);
  }
  check_shape(checks, 11, {128, 128, 16}, false);
}

void test_device_limits(Checks &checks) {
  const GemmShape aligned{256, 512, 32};
  // Independently specified launch fixtures for the current public presets.
  // Grid axes intentionally differ: IDs 1-3 use M on x; IDs 4-12 use N on x.
  struct Resource { int id, threads, grid_x, grid_y; std::size_t shared; };
  constexpr Resource resources[] = {
      {1, 1024, 8, 16, 0}, {2, 1024, 8, 16, 0}, {3, 1024, 8, 16, 8192},
      {4, 512, 8, 4, 4096}, {5, 256, 4, 2, 8192}, {6, 256, 4, 2, 8192},
      {7, 256, 4, 2, 8192}, {8, 256, 4, 2, 8352}, {9, 256, 4, 2, 16384},
      {10, 128, 4, 2, 16384}, {11, 256, 2, 2, 49152}};
  for (const auto &resource : resources) {
    auto device = ample_device();
    device.max_threads_per_block = resource.threads;
    device.shared_memory_per_block = resource.shared;
    device.max_grid_x = resource.grid_x;
    device.max_grid_y = resource.grid_y;
    check_support(checks, resource.id, aligned, device, true);
    auto insufficient = device;
    insufficient.max_threads_per_block -= 1;
    check_support(checks, resource.id, aligned, insufficient, false);
    insufficient = device;
    insufficient.max_grid_x -= 1;
    check_support(checks, resource.id, aligned, insufficient, false);
    insufficient = device;
    insufficient.max_grid_y -= 1;
    check_support(checks, resource.id, aligned, insufficient, false);
    if (resource.shared != 0) {
      insufficient = device;
      insufficient.shared_memory_per_block -= 1;
      check_support(checks, resource.id, aligned, insufficient, false);
    }
  }
  auto device = ample_device();
  device.max_threads_per_block = 64;
  device.shared_memory_per_block = 4096;
  device.max_grid_x = 3;
  device.max_grid_y = 1;
  check_support(checks, 5, {64, 192, 24}, device, true);
  device.max_threads_per_block = 63;
  check_support(checks, 5, {64, 192, 24}, device, false);
  device.max_threads_per_block = 64;
  device.shared_memory_per_block = 4095;
  check_support(checks, 5, {64, 192, 24}, device, false);
  device.shared_memory_per_block = 4096;
  device.max_grid_x = 1;
  device.max_grid_y = 3;
  check_support(checks, 5, {192, 64, 24}, device, true);

  for (int id : {1, 2}) {
    device = ample_device();
    device.max_grid_x = 2;
    device.max_grid_y = 3;
    check_support(checks, id, {33, 65, 1}, device, true);
    device.max_grid_x = 1;
    check_support(checks, id, {33, 65, 1}, device, false);
    device.max_grid_x = 2;
    device.max_grid_y = 2;
    check_support(checks, id, {33, 65, 1}, device, false);
    device = ample_device();
    device.max_grid_x = 67108864;
    check_support(checks, id, {std::numeric_limits<int>::max(), 1, 1}, device, true);
    --device.max_grid_x;
    check_support(checks, id, {std::numeric_limits<int>::max(), 1, 1}, device, false);
  }

  for (int cc : {0, 70, 75, 79, 80, 86, 90}) {
    device = ample_device();
    device.compute_capability = cc;
    check_support(checks, 12, aligned, device, cc >= 80);
    // No feature gate is added to the older kernels by this supported policy.
    for (int id = 0; id < 12; ++id) {
      check_support(checks, id, aligned, device, true);
    }
  }
  device = ample_device();
  device.max_threads_per_block = 128;
  device.max_grid_x = 4;
  device.max_grid_y = 2;
  check_support(checks, 12, aligned, device, true);
  device.max_threads_per_block = 127;
  check_support(checks, 12, aligned, device, false);
  device = ample_device();
  // The two tile buffers alone occupy 32 KiB; barriers require extra space.
  // Do not hard-code CUDA's barrier ABI in this pure C++ test.
  device.shared_memory_per_block = 32768;
  check_support(checks, 12, aligned, device, false);
  device = ample_device();
  device.max_grid_x = 3;
  check_support(checks, 12, aligned, device, false);
  device = ample_device();
  device.max_grid_y = 1;
  check_support(checks, 12, aligned, device, false);
  for (int id = 1; id <= 12; ++id) {
    check_support(checks, id, aligned, DeviceCapabilities{}, false);
  }
  // Vendor internals are not constrained by our custom launch limits.
  check_support(checks, 0, aligned, DeviceCapabilities{}, true);
  check_support(checks, 0, {7, 13, 5}, DeviceCapabilities{}, true);
}

void check_auto(Checks &checks, GemmShape shape,
                const DeviceCapabilities &device, int expected) {
  const auto selected = select_kernel(kAutoKernel, shape, device);
  checks.require(selected.id == expected,
                 label(expected, shape) + ": wrong auto choice " + std::to_string(selected.id));
  checks.require(!selected.reason.empty(), "auto choice has no explanation");
  checks.require(selected.reason.find("policy") != std::string::npos ||
                     selected.reason.find("heuristic") != std::string::npos,
                 "auto explanation must identify a policy/heuristic");
  checks.require(selected.reason.find("fastest") == std::string::npos,
                 "auto explanation must not claim measured fastest selection");
  const auto repeated = select_kernel(kAutoKernel, shape, device);
  checks.require(repeated.id == selected.id && repeated.reason == selected.reason,
                 "auto selection and explanation must be deterministic");
}

void test_selection_policy(Checks &checks) {
  const auto ample = ample_device();
  check_auto(checks, {256, 256, 256}, ample, 10);
  check_auto(checks, {128, 512, 32}, ample, 10);
  check_auto(checks, {128, 256, 8}, ample, 6);
  check_auto(checks, {64, 64, 8}, ample, 5);
  check_auto(checks, {64, 192, 24}, ample, 5);
  check_auto(checks, {192, 64, 40}, ample, 5);
  check_auto(checks, {192, 128, 8}, ample, 0);
  check_auto(checks, {1, 1, 1}, ample, 0);
  check_auto(checks, {7, 13, 5}, ample, 0);
  check_auto(checks, {32, 32, 32}, ample, 0);

  auto device = ample;
  device.shared_memory_per_block = 8192;
  check_auto(checks, {128, 256, 32}, device, 6);
  device.shared_memory_per_block = 8191;
  check_auto(checks, {128, 256, 32}, device, 0);
  device = ample;
  device.max_threads_per_block = 128;
  check_auto(checks, {128, 256, 32}, device, 10);
  device.max_threads_per_block = 127;
  check_auto(checks, {128, 256, 32}, device, 0);
  device = ample;
  device.max_threads_per_block = 64;
  device.shared_memory_per_block = 4096;
  check_auto(checks, {64, 192, 24}, device, 5);
  device.max_threads_per_block = 63;
  check_auto(checks, {64, 192, 24}, device, 0);
  device.max_threads_per_block = 64;
  device.shared_memory_per_block = 4095;
  check_auto(checks, {64, 192, 24}, device, 0);
  device = ample;
  device.max_grid_x = 1;
  // Kernel 11 could fit this grid; it is deliberately excluded from auto.
  check_support(checks, 11, {128, 256, 32}, device, true);
  check_auto(checks, {128, 256, 32}, device, 0);
  device = ample;
  device.max_grid_y = 1;
  check_auto(checks, {256, 128, 32}, device, 0);
  check_auto(checks, {128, 256, 32}, DeviceCapabilities{}, 0);

  // Explicit choices survive even when auto would choose a different ID.
  for (int id = 0; id <= 12; ++id) {
    check_support(checks, id, {128, 256, 32}, ample, true);
  }
  check_support(checks, 3, {32, 32, 32}, ample, true);
  check_support(checks, 10, {128, 256, 8}, ample, false); // auto would use 6
  check_support(checks, 6, {64, 192, 24}, ample, false); // auto would use 5
  check_support(checks, 5, {7, 13, 5}, ample, false);    // auto would use cuBLAS
  device = ample;
  device.compute_capability = 79;
  check_support(checks, 12, {128, 256, 32}, device, false);
  check_auto(checks, {128, 256, 32}, device, 10);
}
} // namespace

int main() {
  try {
    Checks checks;
    test_registry(checks);
    test_problem_validation(checks);
    test_preset_shapes(checks);
    test_device_limits(checks);
    test_selection_policy(checks);
    std::cout << "PASS pure C++ kernel registry/selection (" << checks.count()
              << " checks; no CUDA device required)\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
