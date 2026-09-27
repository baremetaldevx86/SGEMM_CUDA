#pragma once

// CPU-only lifecycle model, NOT a numerical GEMM/CUDA implementation. Each fake
// device allocation stores one sentinel, but retains its real allocation size.
// These checks intentionally remain active in Release builds with -DNDEBUG.
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <vector>

using cudaError_t = int;
using cublasStatus_t = int;
constexpr int CUBLAS_STATUS_SUCCESS = 0;
using cublasHandle_t = void *;
using cudaStream_t = void *;
using cudaEvent_t = int *;
enum cudaMemcpyKind {
  cudaMemcpyHostToDevice,
  cudaMemcpyDeviceToHost,
  cudaMemcpyDeviceToDevice
};

namespace benchmark_mock {
inline void check(bool condition, const char *expression, int line) {
  if (!condition) {
    std::fprintf(stderr, "MOCK invariant failed at runner.cuh:%d: %s\n", line,
                 expression);
    std::exit(90); // Distinct from the benchmark's expected error exit code.
  }
}
#define MOCK_CHECK(expression) \
  ::benchmark_mock::check(static_cast<bool>(expression), #expression, __LINE__)

constexpr int sizes[] = {128, 256, 512, 1024, 2048, 4096};
constexpr std::size_t max_elements = 4096u * 4096u;
constexpr std::size_t max_bytes = sizeof(float) * max_elements;
struct Shape {
  int size;
  bool supported;
  int launches = 0;
  int resets = 0;
  int timed = 0;
  int verifications = 0;
};
enum class Timing { idle, started, stopped, synchronized };
inline Timing timing = Timing::idle;
inline int interval_launches = 0;
inline int event_creations = 0;
inline int live_events = 0;
inline int uploads = 0;
inline int random_calls = 0;
inline int selected_kernel = -1;
inline bool launch_checked = true;
inline bool device_synchronized = true;
inline bool handle_alive = false;
inline bool device_selected = false;
inline float *host_inputs[3] = {};
inline float *device_inputs[3] = {};
inline std::map<void *, std::size_t> allocations;
inline std::map<float *, std::size_t> ready_outputs;
inline std::vector<Shape> shapes;
inline std::string failed_operation;
inline std::map<std::string, int> failure_calls;

inline bool fail(const char *operation) {
  const int call = ++failure_calls[operation];
  const char *requested = std::getenv("SGEMM_MOCK_FAIL");
  // A bare operation fails every call; operation:N fails only its Nth call.
  if (requested && (std::string(requested) == operation ||
                    requested == std::string(operation) + ":" + std::to_string(call))) {
    failed_operation = requested;
    return true;
  }
  return false;
}
inline bool active_interval() { return timing == Timing::started; }
inline Shape &shape() {
  MOCK_CHECK(!shapes.empty() && shapes.back().supported);
  return shapes.back();
}
inline void check_baseline() {
  MOCK_CHECK(uploads == 3);
  for (int i = 0; i < 3; ++i) {
    MOCK_CHECK(host_inputs[i][0] == static_cast<float>(i + 1));
    MOCK_CHECK(device_inputs[i][0] == static_cast<float>(i + 1));
  }
}
inline void check_scalar(float actual, const char *variable, float fallback) {
  const char *expected = std::getenv(variable);
  MOCK_CHECK(actual == (expected ? std::stof(expected) : fallback));
}
inline cudaError_t reset(void *destination, const void *source,
                         std::size_t bytes) {
  MOCK_CHECK(timing == Timing::idle && launch_checked);
  check_baseline();
  MOCK_CHECK(source == device_inputs[2] && destination != source);
  MOCK_CHECK(destination != device_inputs[0] && destination != device_inputs[1]);
  MOCK_CHECK(allocations.count(destination) == 1);
  MOCK_CHECK(bytes == sizeof(float) * shape().size * shape().size);
  MOCK_CHECK(ready_outputs.emplace(static_cast<float *>(destination), bytes).second);
  *static_cast<float *>(destination) = *static_cast<const float *>(source);
  ++shape().resets;
  return fail("reset");
}
} // namespace benchmark_mock

inline void cudaCheck(cudaError_t status, const char *file, int line) {
  if (status) {
    std::fprintf(stderr, "mock CUDA failure: %s at %s:%d\n",
                 benchmark_mock::failed_operation.c_str(), file, line);
    std::exit(1);
  }
}
inline cudaError_t cudaGetDeviceCount(int *count) {
  *count = benchmark_mock::fail("no_devices") ? 0 : 1;
  return benchmark_mock::fail("device_count");
}
inline cudaError_t cudaSetDevice(int device) {
  MOCK_CHECK(device == 0);
  benchmark_mock::device_selected = true;
  return benchmark_mock::fail("set_device");
}
inline cublasStatus_t cublasCreate(cublasHandle_t *handle) {
  using namespace benchmark_mock;
  MOCK_CHECK(device_selected && !handle_alive);
  *handle = &handle_alive;
  handle_alive = true;
  return fail("cublas_create");
}
inline cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t stream) {
  MOCK_CHECK(handle == &benchmark_mock::handle_alive && !stream);
  return benchmark_mock::fail("cublas_stream");
}
inline cublasStatus_t cublasDestroy(cublasHandle_t handle) {
  using namespace benchmark_mock;
  MOCK_CHECK(handle == &handle_alive && handle_alive);
  MOCK_CHECK(allocations.empty() && live_events == 0);
  MOCK_CHECK(timing == Timing::idle && launch_checked && ready_outputs.empty());
  MOCK_CHECK(shapes.size() == 6 && random_calls == 3 && uploads == 3);
  int launches = 0, resets = 0, timed = 0;
  for (const Shape &entry : shapes) {
    MOCK_CHECK(entry.launches == entry.resets);
    if (!entry.supported) {
      MOCK_CHECK(entry.launches == 0 && entry.verifications == 0);
    }
    std::printf("MOCK_SIZE: size=%d supported=%d launches=%d resets=%d timed=%d "
                "verifications=%d\n", entry.size, entry.supported, entry.launches,
                entry.resets, entry.timed, entry.verifications);
    launches += entry.launches;
    resets += entry.resets;
    timed += entry.timed;
  }
  std::printf("MOCK_CHECK: launches=%d resets=%d timed=%d transitions=%zu\n",
              launches, resets, timed, shapes.size());
  handle_alive = false;
  return fail("cublas_destroy");
}
inline cudaError_t cudaEventCreate(cudaEvent_t *event) {
  using namespace benchmark_mock;
  MOCK_CHECK(event_creations < 2);
  *event = new int(event_creations++);
  ++live_events;
  return fail("event_create");
}
inline cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  using namespace benchmark_mock;
  MOCK_CHECK(!stream && launch_checked);
  if (*event == 0) {
    MOCK_CHECK(timing == Timing::idle && ready_outputs.size() == 1);
    timing = Timing::started;
    interval_launches = 0;
  } else {
    MOCK_CHECK(*event == 1 && active_interval() && interval_launches == 1);
    timing = Timing::stopped;
  }
  return fail("event_record");
}
inline cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  using namespace benchmark_mock;
  MOCK_CHECK(*event == 1 && timing == Timing::stopped);
  timing = Timing::synchronized;
  device_synchronized = true;
  return fail("event_sync");
}
inline cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t begin,
                                       cudaEvent_t end) {
  using namespace benchmark_mock;
  MOCK_CHECK(*begin == 0 && *end == 1 && timing == Timing::synchronized);
  // Different durations expose incorrect accumulation/averaging across iters.
  *ms = static_cast<float>(shape().timed);
  if (fail("zero_time")) *ms = 0.0f;
  if (fail("negative_time")) *ms = -1.0f;
  if (fail("nan_time")) *ms = std::numeric_limits<float>::quiet_NaN();
  if (fail("infinite_time")) *ms = std::numeric_limits<float>::infinity();
  timing = Timing::idle;
  return fail("event_elapsed");
}
inline cudaError_t cudaEventDestroy(cudaEvent_t event) {
  MOCK_CHECK(benchmark_mock::timing == benchmark_mock::Timing::idle);
  delete event;
  --benchmark_mock::live_events;
  return benchmark_mock::fail("event_destroy");
}
inline cudaError_t cudaMalloc(void **pointer, std::size_t bytes) {
  using namespace benchmark_mock;
  MOCK_CHECK(bytes == max_bytes && allocations.size() < 5);
  *pointer = new float(std::numeric_limits<float>::quiet_NaN());
  MOCK_CHECK(allocations.emplace(*pointer, bytes).second);
  return fail("malloc");
}
inline cudaError_t cudaFree(void *pointer) {
  using namespace benchmark_mock;
  MOCK_CHECK(timing == Timing::idle && ready_outputs.empty());
  MOCK_CHECK(allocations.erase(pointer) == 1);
  for (int i = 0; i < 3; ++i) {
    if (pointer == device_inputs[i]) {
      MOCK_CHECK(*static_cast<float *>(pointer) == static_cast<float>(i + 1));
      MOCK_CHECK(host_inputs[i][0] == static_cast<float>(i + 1));
    }
  }
  delete static_cast<float *>(pointer);
  return fail("free");
}
inline cudaError_t cudaMemcpy(void *destination, const void *source,
                              std::size_t bytes, cudaMemcpyKind kind) {
  using namespace benchmark_mock;
  MOCK_CHECK(timing == Timing::idle);
  if (kind == cudaMemcpyDeviceToDevice) return reset(destination, source, bytes);
  if (kind == cudaMemcpyHostToDevice) {
    MOCK_CHECK(uploads < 3 && source == host_inputs[uploads]);
    MOCK_CHECK(bytes == max_bytes && allocations.count(destination) == 1);
    device_inputs[uploads++] = static_cast<float *>(destination);
  } else {
    MOCK_CHECK(kind == cudaMemcpyDeviceToHost && device_synchronized);
    MOCK_CHECK(allocations.count(const_cast<void *>(source)) == 1);
    MOCK_CHECK(bytes == sizeof(float) * shape().size * shape().size);
    for (int i = 0; i < 3; ++i) MOCK_CHECK(destination != host_inputs[i]);
    check_baseline();
  }
  *static_cast<float *>(destination) = *static_cast<const float *>(source);
  return fail("copy");
}
inline cudaError_t cudaMemcpyAsync(void *destination, const void *source,
                                   std::size_t bytes, cudaMemcpyKind kind,
                                   cudaStream_t stream) {
  MOCK_CHECK(!stream && kind == cudaMemcpyDeviceToDevice);
  return benchmark_mock::reset(destination, source, bytes);
}
inline cudaError_t cudaDeviceSynchronize() {
  using namespace benchmark_mock;
  MOCK_CHECK(timing == Timing::idle && launch_checked);
  device_synchronized = true;
  return fail("device_sync");
}
inline cudaError_t cudaGetLastError() {
  MOCK_CHECK(!benchmark_mock::launch_checked);
  benchmark_mock::launch_checked = true;
  return benchmark_mock::fail("launch");
}
inline void randomize_matrix(float *matrix, int elements) {
  using namespace benchmark_mock;
  MOCK_CHECK(random_calls < 3 && elements == static_cast<int>(max_elements));
  host_inputs[random_calls] = matrix;
  matrix[0] = static_cast<float>(++random_calls);
}
inline bool verify_matrix(float *reference, float *output, int elements) {
  using namespace benchmark_mock;
  MOCK_CHECK(device_synchronized && timing == Timing::idle);
  MOCK_CHECK(reference != output && elements == shape().size * shape().size);
  check_baseline();
  ++shape().verifications;
  return reference[0] == output[0] && !fail("verify");
}
inline void print_matrix(const float *matrix, int, int, std::ofstream &stream) {
  stream << matrix[0] << '\n';
}
inline const char *kernel_shape_error(int kernel, int m, int n, int k) {
  using namespace benchmark_mock;
  MOCK_CHECK(kernel >= 0 && kernel <= 12 && m == n && n == k);
  MOCK_CHECK(shapes.size() < 6 && m == sizes[shapes.size()]);
  MOCK_CHECK(timing == Timing::idle && ready_outputs.empty());
  MOCK_CHECK(selected_kernel == -1 || selected_kernel == kernel);
  selected_kernel = kernel;
  check_baseline();
  const bool supported = !(kernel == 11 && m == 128) && !fail("skip_all");
  shapes.push_back({m, supported});
  return supported ? nullptr : "mock preset requires a larger aligned tile";
}
inline void run_kernel(int kernel, int m, int n, int k, float alpha, float *a,
                        float *b, float beta, float *output,
                        cublasHandle_t handle) {
  using namespace benchmark_mock;
  MOCK_CHECK(handle_alive && handle == &handle_alive && launch_checked);
  MOCK_CHECK(m == shape().size && m == n && n == k);
  MOCK_CHECK(a == device_inputs[0] && b == device_inputs[1]);
  MOCK_CHECK(output != device_inputs[2] && allocations.count(output) == 1);
  check_baseline();
  check_scalar(alpha, "SGEMM_MOCK_EXPECT_ALPHA", 0.5f);
  check_scalar(beta, "SGEMM_MOCK_EXPECT_BETA", 3.0f);
  const bool reference_launch = selected_kernel != 0 && shape().launches == 0;
  MOCK_CHECK(kernel == (reference_launch ? 0 : selected_kernel));
  MOCK_CHECK(ready_outputs.erase(output) == 1 && output[0] == host_inputs[2][0]);
  // Deliberately not GEMM: consume the initial C and leave a distinguishable
  // output so a missing reset cannot silently pass even with a nonzero beta.
  output[0] = alpha * static_cast<float>(m + n + k) + beta * output[0];
  ++shape().launches;
  launch_checked = false;
  device_synchronized = false;
  if (active_interval()) {
    ++interval_launches;
    ++shape().timed;
  } else {
    MOCK_CHECK(timing == Timing::idle);
  }
}

#undef MOCK_CHECK
