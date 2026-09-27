// Public-runner regression tests; link this file with src/runner.cu and cuBLAS.
// --host never queries or initializes a CUDA device. --gpu returns 77 only when
// device discovery reports no device/insufficient driver (CTest SKIP_RETURN_CODE).
// These numerical tests are not a replacement for separate sanitizer runs.
#include "../src/runner.cuh"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kSkip = 77;
// Inputs are bounded by 1, K <= 256, and accumulation is FP32. Retain an
// absolute term for cancellation near zero, without hiding indexing/tile bugs.
constexpr double kAbsoluteTolerance = 2.0e-4;
constexpr double kRelativeTolerance = 2.0e-5;

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

std::string shape_label(int id, int m, int n, int k) {
  std::ostringstream out;
  out << "kernel " << id << " M=" << m << " N=" << n << " K=" << k;
  return out.str();
}

void check_shape(Checks &checks, int id, int m, int n, int k,
                 bool supported) {
  const std::string label = shape_label(id, m, n, k);
  const char *error = kernel_shape_error(id, m, n, k);
  checks.require((error == nullptr) == supported,
                 label + (supported ? ": unexpectedly rejected: "
                                    : ": unexpectedly supported") +
                     (error ? error : ""));
  if (!supported) {
    checks.require(error[0] != '\0', label + ": empty shape diagnostic");
    // Deliberately ordinary host pointers: every rejected input must throw
    // before a CUDA API or kernel can access them, even without a driver.
    alignas(16) float storage[3][8]{};
    expect_invalid_argument(checks, label, [&] {
      run_kernel(id, m, n, k, 1.0f, storage[0], storage[1], 0.0f,
                 storage[2], nullptr);
    });
  }
}

void test_shapes(Checks &checks) {
  const int max_int = std::numeric_limits<int>::max();
  for (int id : {-1, 13, std::numeric_limits<int>::min(), max_int}) {
    check_shape(checks, id, 256, 256, 32, false);
  }
  for (int id = 0; id <= 12; ++id) {
    check_shape(checks, id, 256, 256, 256, true);
    check_shape(checks, id, 128, 512, 32, true);
    // Just below the indexing limit, while aligned for every tiled preset.
    check_shape(checks, id, 32768, 65280, 32, true);
    for (int dimension = 0; dimension != 3; ++dimension) {
      for (int bad : {0, -1, std::numeric_limits<int>::min()}) {
        int dims[] = {256, 256, 32};
        dims[dimension] = bad;
        check_shape(checks, id, dims[0], dims[1], dims[2], false);
      }
    }
    // Independently overflow C=M*N, A=M*K, and B=K*N. These products
    // overflow signed 32-bit arithmetic, but the dimensions themselves fit.
    check_shape(checks, id, 65536, 32768, 32, false);
    check_shape(checks, id, 65536, 256, 32768, false);
    check_shape(checks, id, 128, 65536, 32768, false);
    check_shape(checks, id, max_int, max_int, max_int, false);
  }
  for (int id = 0; id <= 2; ++id) {
    for (const auto &dims : std::vector<std::vector<int>>{
             {1, 1, 1}, {7, 13, 5}, {31, 33, 17}, {37, 19, 65},
             {max_int, 1, 1}, {1, max_int, 1}, {1, 1, max_int}}) {
      // Boundary-size successes are shape queries only; never allocate or
      // launch these enormous matrices.
      check_shape(checks, id, dims[0], dims[1], dims[2], true);
    }
    check_shape(checks, id, 46341, 46341, 1, false);
    check_shape(checks, id, 46341, 1, 46341, false);
    check_shape(checks, id, 1, 46341, 46341, false);
  }

  // Explicit current public presets, independent of runner implementation.
  struct Preset { int id, m, n, k; };
  const Preset presets[] = {
      {3, 32, 32, 32}, {4, 64, 64, 8}, {5, 128, 128, 8},
      {6, 128, 128, 8}, {7, 128, 128, 8}, {8, 128, 128, 8},
      {9, 128, 128, 16}, {10, 128, 128, 16},
      {11, 128, 256, 16}, {12, 128, 128, 16}};
  for (const Preset &p : presets) {
    check_shape(checks, p.id, p.m, p.n, p.k, true);
    check_shape(checks, p.id, 2 * p.m, 3 * p.n, 3 * p.k, true);
    check_shape(checks, p.id, p.m - 1, p.n, p.k, false);
    check_shape(checks, p.id, p.m, p.n - 1, p.k, false);
    check_shape(checks, p.id, p.m, p.n, p.k - 1, false);
    check_shape(checks, p.id, p.m + 1, p.n, p.k, false);
    check_shape(checks, p.id, p.m, p.n + 1, p.k, false);
    check_shape(checks, p.id, p.m, p.n, p.k + 1, false);
  }
  // Kernel 5 retains its working 64x64 fallback; 6-8 do not.
  for (const auto &dims : std::vector<std::vector<int>>{
           {64, 64, 8}, {64, 192, 24}, {192, 64, 24}}) {
    check_shape(checks, 5, dims[0], dims[1], dims[2], true);
    for (int id = 6; id <= 8; ++id) {
      check_shape(checks, id, dims[0], dims[1], dims[2], false);
    }
  }
  check_shape(checks, 5, 192, 128, 8, false);
  check_shape(checks, 5, 128, 192, 8, false);
  check_shape(checks, 5, 32, 64, 8, false);
  check_shape(checks, 11, 128, 128, 16, false);
}

void test_pointer_rejections(Checks &checks) {
  alignas(16) float storage[3][8]{};
  for (int id = 0; id <= 12; ++id) {
    for (int operand = 0; operand != 3; ++operand) {
      const std::string label = "kernel " + std::to_string(id) + " operand " +
                                std::to_string(operand);
      float *pointers[] = {storage[0], storage[1], storage[2]};
      pointers[operand] = nullptr;
      expect_invalid_argument(checks, label + " null", [&] {
        run_kernel(id, 256, 256, 32, 1.0f, pointers[0], pointers[1],
                   0.0f, pointers[2], nullptr);
      });
      pointers[operand] = reinterpret_cast<float *>(
          reinterpret_cast<unsigned char *>(storage[operand]) + 1);
      expect_invalid_argument(checks, label + " byte-misaligned", [&] {
        run_kernel(id, 256, 256, 32, 1.0f, pointers[0], pointers[1],
                   0.0f, pointers[2], nullptr);
      });
      if (id >= 6) {
        // Float alignment alone is insufficient for vectorized presets.
        pointers[operand] = storage[operand] + 1;
        expect_invalid_argument(checks, label + " float4-misaligned", [&] {
          run_kernel(id, 256, 256, 32, 1.0f, pointers[0], pointers[1],
                     0.0f, pointers[2], nullptr);
        });
      }
    }
  }
}

bool within_tolerance(double expected, float actual) {
  return std::isfinite(expected) && std::isfinite(actual) &&
         std::abs(expected - static_cast<double>(actual)) <=
             kAbsoluteTolerance + kRelativeTolerance * std::abs(expected);
}

void test_helpers(Checks &checks) {
  float guarded[] = {-99.0f, -1.0f, -1.0f, -1.0f, 99.0f};
  range_init_matrix(guarded + 1, 3);
  checks.require(guarded[0] == -99.0f && guarded[1] == 0.0f &&
                     guarded[2] == 1.0f && guarded[3] == 2.0f &&
                     guarded[4] == 99.0f,
                 "range_init_matrix contents/bounds");
  zero_init_matrix(guarded + 1, 3);
  checks.require(guarded[0] == -99.0f && guarded[1] == 0.0f &&
                     guarded[2] == 0.0f && guarded[3] == 0.0f &&
                     guarded[4] == 99.0f,
                 "zero_init_matrix contents/bounds");
  const float source[] = {1.25f, -2.5f, 3.75f};
  copy_matrix(source, guarded + 1, 3);
  checks.require(guarded[0] == -99.0f && guarded[1] == source[0] &&
                     guarded[2] == source[1] && guarded[3] == source[2] &&
                     guarded[4] == 99.0f,
                 "copy_matrix contents/bounds");
  copy_matrix(guarded, guarded, 5);
  checks.require(guarded[1] == source[0] && guarded[4] == 99.0f,
                 "copy_matrix identical source/destination");
  copy_matrix(nullptr, nullptr, 0);
  range_init_matrix(nullptr, 0);
  zero_init_matrix(nullptr, 0);
  randomize_matrix(nullptr, 0);
  expect_invalid_argument(checks, "copy negative count", [&] {
    copy_matrix(source, guarded, -1);
  });
  expect_invalid_argument(checks, "copy null source", [&] {
    copy_matrix(nullptr, guarded, 1);
  });
  expect_invalid_argument(checks, "copy null destination", [&] {
    copy_matrix(source, nullptr, 1);
  });

  float random_full[32], random_split[32];
  std::srand(12345);
  randomize_matrix(random_full, 32);
  std::srand(12345);
  randomize_matrix(random_split, 11);
  randomize_matrix(random_split + 11, 21);
  for (int i = 0; i != 32; ++i) {
    checks.require(std::isfinite(random_full[i]) &&
                       std::abs(random_full[i]) <= 4.041f &&
                       random_full[i] == random_split[i],
                   "randomize_matrix caller-owned seed and finite range");
  }

  float reference[] = {0.0f, 1.0f, -2.0f};
  float output[] = {0.0f, 1.0f, -2.0f};
  checks.require(verify_matrix(reference, output, 3), "verify identical values");
  output[2] += 0.005f;
  checks.require(verify_matrix(reference, output, 3), "verify accepted tolerance");
  output[2] = 0.0f;
  std::cout << "Expected verify_matrix rejection diagnostics follow:\n";
  checks.require(!verify_matrix(reference, output, 3), "verify rejects mismatch");
  for (float bad : {std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity()}) {
    float finite = 1.0f;
    checks.require(!verify_matrix(&finite, &bad, 1), "verify rejects nonfinite output");
    checks.require(!verify_matrix(&bad, &finite, 1), "verify rejects nonfinite reference");
    checks.require(!verify_matrix(&bad, &bad, 1), "verify rejects equal nonfinite values");
  }
  checks.require(verify_matrix(nullptr, nullptr, 0), "verify empty matrices");
  float begin = 1000000.0f, end = 3500000.0f;
  checks.require(cpu_elapsed_time(begin, end) == 2.5f,
                 "cpu_elapsed_time microseconds-to-seconds conversion");
  checks.require(std::isfinite(get_sec()) && get_sec() > 0.0f,
                 "get_sec finite positive timestamp");
  checks.require(std::isfinite(get_current_sec()) && get_current_sec() > 0.0f,
                 "get_current_sec compatibility alias");
  checks.require(within_tolerance(0.0, 1.0e-4f) &&
                     !within_tolerance(0.0, 1.0e-3f) &&
                     within_tolerance(100.0, 100.001f) &&
                     !within_tolerance(100.0, 100.01f) &&
                     !within_tolerance(0.0, std::numeric_limits<float>::quiet_NaN()) &&
                     !within_tolerance(0.0, std::numeric_limits<float>::infinity()),
                 "CPU-reference comparator absolute/relative/finite checks");
}

void run_host_tests(Checks &checks) {
  test_shapes(checks);
  test_pointer_rejections(checks);
  test_helpers(checks);
  std::cout << "PASS host contracts/helpers (" << checks.count() << " checks)\n";
}

void check_cuda(cudaError_t status, const std::string &operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(operation + ": " + cudaGetErrorString(status));
  }
}

void check_cublas(cublasStatus_t status, const char *operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cublasGetStatusName(status) + " (" +
                             std::to_string(static_cast<int>(status)) + ")");
  }
}

class DeviceBuffer {
public:
  explicit DeviceBuffer(std::size_t elements) {
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&data_), elements * sizeof(float)),
               "cudaMalloc");
  }
  ~DeviceBuffer() {
    // Destructors cannot throw. Cleanup errors still fail, never become skips.
    cudaCheck(cudaFree(data_), __FILE__, __LINE__);
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  float *data() const { return data_; }

private:
  float *data_ = nullptr;
};

class BlasHandle {
public:
  BlasHandle() { check_cublas(cublasCreate(&handle_), "cublasCreate"); }
  ~BlasHandle() {
    const cublasStatus_t status = cublasDestroy(handle_);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::fprintf(stderr, "FAIL cublasDestroy: %s (%d)\n",
                   cublasGetStatusName(status), static_cast<int>(status));
      std::exit(EXIT_FAILURE);
    }
  }
  BlasHandle(const BlasHandle &) = delete;
  BlasHandle &operator=(const BlasHandle &) = delete;
  cublasHandle_t get() const { return handle_; }

private:
  cublasHandle_t handle_ = nullptr;
};

void fill_input(std::vector<float> &values, std::uint32_t state) {
  // Reproducible across standard libraries, independent of rand()/host tests.
  for (float &value : values) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    int centered = static_cast<int>(state % 2000) - 1000;
    if (centered >= 0) {
      ++centered;
    }
    value = static_cast<float>(centered) / 1000.0f;
  }
}

struct Scalars { float alpha, beta; };
constexpr Scalars kScalars[] = {
    {1.0f, 0.0f}, {0.5f, 1.25f}, {-0.75f, -0.5f},
    {0.0f, -1.5f}, {0.0f, 0.0f}};

void test_gpu_shape(Checks &checks, cublasHandle_t handle, int m, int n, int k,
                    const std::vector<int> &ids, std::size_t &case_count) {
  std::vector<float> a(static_cast<std::size_t>(m) * k);
  std::vector<float> b(static_cast<std::size_t>(k) * n);
  std::vector<float> initial_c(static_cast<std::size_t>(m) * n);
  fill_input(a, 0x12345678U);
  fill_input(b, 0x9abcdef1U);
  fill_input(initial_c, 0x73489201U);
  std::vector<double> product(initial_c.size());
  // Independent row-major reference: convert the actual float inputs to
  // double before multiplication, and accumulate entirely on the CPU.
  for (int row = 0; row != m; ++row) {
    for (int col = 0; col != n; ++col) {
      double sum = 0.0;
      for (int inner = 0; inner != k; ++inner) {
        sum += static_cast<double>(a[static_cast<std::size_t>(row) * k + inner]) *
               static_cast<double>(b[static_cast<std::size_t>(inner) * n + col]);
      }
      product[static_cast<std::size_t>(row) * n + col] = sum;
    }
  }

  DeviceBuffer device_a(a.size()), device_b(b.size()), device_c(initial_c.size());
  check_cuda(cudaMemcpy(device_a.data(), a.data(), a.size() * sizeof(float),
                         cudaMemcpyHostToDevice), "upload A");
  check_cuda(cudaMemcpy(device_b.data(), b.data(), b.size() * sizeof(float),
                         cudaMemcpyHostToDevice), "upload B");
  std::vector<float> actual(initial_c.size());
  for (int id : ids) {
    checks.require(kernel_shape_error(id, m, n, k) == nullptr,
                   shape_label(id, m, n, k) + ": numerical fixture unsupported");
    for (const Scalars &scalars : kScalars) {
      const std::string label = shape_label(id, m, n, k) + " alpha=" +
                                std::to_string(scalars.alpha) + " beta=" +
                                std::to_string(scalars.beta);
      // Always initialize and restore C, including beta=0 and repeated IDs.
      check_cuda(cudaMemcpy(device_c.data(), initial_c.data(),
                             initial_c.size() * sizeof(float), cudaMemcpyHostToDevice),
                 label + ": restore initial C");
      run_kernel(id, m, n, k, scalars.alpha, device_a.data(), device_b.data(),
                 scalars.beta, device_c.data(), handle);
      check_cuda(cudaGetLastError(), label + ": launch");
      check_cuda(cudaDeviceSynchronize(), label + ": synchronize");
      check_cuda(cudaMemcpy(actual.data(), device_c.data(), actual.size() * sizeof(float),
                             cudaMemcpyDeviceToHost), label + ": download C");
      for (std::size_t index = 0; index != actual.size(); ++index) {
        const double expected = static_cast<double>(scalars.alpha) * product[index] +
                                static_cast<double>(scalars.beta) * initial_c[index];
        if (!within_tolerance(expected, actual[index])) {
          std::ostringstream error;
          error << std::setprecision(17) << label << ": mismatch at ("
                << index / n << ',' << index % n << "), expected " << expected
                << ", got " << actual[index] << ", tolerance "
                << kAbsoluteTolerance + kRelativeTolerance * std::abs(expected);
          throw std::runtime_error(error.str());
        }
      }
      ++case_count;
    }
  }
  // Inputs are public non-const pointers, but GEMM must not mutate them.
  std::vector<float> copied_a(a.size()), copied_b(b.size());
  check_cuda(cudaMemcpy(copied_a.data(), device_a.data(), a.size() * sizeof(float),
                         cudaMemcpyDeviceToHost), "download unchanged A");
  check_cuda(cudaMemcpy(copied_b.data(), device_b.data(), b.size() * sizeof(float),
                         cudaMemcpyDeviceToHost), "download unchanged B");
  checks.require(copied_a == a && copied_b == b, "GEMM modified A or B");
  std::cout << "PASS GPU M=" << m << " N=" << n << " K=" << k
            << " (" << ids.size() << " kernel IDs, "
            << sizeof(kScalars) / sizeof(kScalars[0]) << " alpha/beta pairs)\n";
}

bool unavailable(cudaError_t status) {
  return status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver;
}

int run_gpu_tests(Checks &checks) {
  int count = 0;
  const cudaError_t discovery = cudaGetDeviceCount(&count);
  if (unavailable(discovery) || (discovery == cudaSuccess && count == 0)) {
    std::cout << "SKIP GPU: " << (discovery == cudaSuccess ? "no CUDA device"
                                                            : cudaGetErrorString(discovery))
              << '\n';
    return kSkip;
  }
  check_cuda(discovery, "cudaGetDeviceCount");
  const cudaError_t selection = cudaSetDevice(0);
  if (unavailable(selection)) {
    std::cout << "SKIP GPU: " << cudaGetErrorString(selection) << '\n';
    return kSkip;
  }
  check_cuda(selection, "cudaSetDevice(0)");
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
  std::cout << "GPU: " << properties.name << '\n';
  BlasHandle handle;
  check_cublas(cublasSetPointerMode(handle.get(), CUBLAS_POINTER_MODE_HOST),
               "cublasSetPointerMode");
  check_cublas(cublasSetMathMode(handle.get(), CUBLAS_DEFAULT_MATH),
               "cublasSetMathMode");
  check_cuda(cudaDeviceSynchronize(), "initialize context before host rejections");
  check_cuda(cudaGetLastError(), "context status before host rejections");
  // Repeat host rejection tests with a live context. A device trap/sticky CUDA
  // error is not acceptable: synchronization and every valid GEMM below must
  // still work afterwards. No reset or error suppression between these steps.
  run_host_tests(checks);
  check_cuda(cudaGetLastError(), "context status after host rejections");
  check_cuda(cudaDeviceSynchronize(), "context usable after host rejections");

  std::size_t case_count = 0;
  std::vector<int> all_ids;
  for (int id = 0; id <= 12; ++id) {
    all_ids.push_back(id);
  }
  test_gpu_shape(checks, handle.get(), 256, 256, 256, all_ids, case_count);
  test_gpu_shape(checks, handle.get(), 128, 512, 32, all_ids, case_count);
  for (const auto &dims : std::vector<std::vector<int>>{
           {1, 1, 1}, {7, 13, 5}, {31, 33, 17}, {37, 19, 65}}) {
    test_gpu_shape(checks, handle.get(), dims[0], dims[1], dims[2],
                   {0, 1, 2}, case_count);
  }
  // Exercise both orientations of kernel 5's small-tile branch as well as
  // the 64x64 case, with nontrivial/odd K-tile counts.
  test_gpu_shape(checks, handle.get(), 64, 64, 24, {5}, case_count);
  test_gpu_shape(checks, handle.get(), 64, 192, 8, {5}, case_count);
  test_gpu_shape(checks, handle.get(), 192, 64, 40, {5}, case_count);
  // Double buffering: one tile, two tiles, and odd three/five-tile tails.
  for (int k : {16, 32, 48, 80}) {
    test_gpu_shape(checks, handle.get(), 128, 256, k, {11, 12}, case_count);
  }
  check_cuda(cudaDeviceSynchronize(), "final GPU synchronization");
  std::cout << "PASS " << case_count << " CPU-double-reference GPU GEMMs; abs_tol="
            << kAbsoluteTolerance << ", rel_tol=" << kRelativeTolerance << '\n';
  return 0;
}
} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
    std::cout << "Usage: " << argv[0] << " --host | --gpu\n"
              << "  --host: CPU-only public runner contracts/helpers\n"
              << "  --gpu:  CPU-double-reference GEMMs and post-rejection context checks\n"
              << "GPU device absence/insufficient driver returns 77; all other failures return 1.\n";
    return 0;
  }
  if (argc != 2 || (std::strcmp(argv[1], "--host") != 0 &&
                    std::strcmp(argv[1], "--gpu") != 0)) {
    std::cerr << "Usage: " << argv[0] << " --host | --gpu\n";
    return 2;
  }
  try {
    Checks checks;
    if (std::strcmp(argv[1], "--host") == 0) {
      run_host_tests(checks);
      return 0;
    }
    return run_gpu_tests(checks);
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
