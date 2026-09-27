#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <kernel_registry.h>
#include <runner.cuh>
#include <stdexcept>
#include <string>
#include <vector>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

static void check_cublas(cublasStatus_t status, const char *operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) +
                             " failed with cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
  }
}

const std::string errLogFile = "matrixValidationFailure.txt";

struct Options {
  int kernel = kAutoKernel;
  std::string requested_kernel;
  bool list_kernels = false;
  bool custom_shape = false;
  GemmShape shape{0, 0, 0};
  int warmup = 5;
  int iterations = 50;
  unsigned int seed = 1234;
  float alpha = 0.5f;
  float beta = 3.0f;
  std::string csv_path;
};

void print_usage(const char *program) {
  std::cerr
      << "Usage: " << program
      << " <ID|name|auto> [--m M --n N --k K] [--warmup N] [--iters N]"
         " [--seed N] [--csv FILE] [--alpha X] [--beta X]\n"
      << "       " << program << " --kernel <ID|name|auto> [same options]\n"
      << "       " << program << " --list-kernels\n"
      << "  kernel: 0-12 or canonical name (0 selects NVIDIA cuBLAS)\n"
      << "  auto:   deterministic heuristic policy, not measured tuning\n"
      << "  --list-kernels: standalone host-only listing; no GPU required\n"
      << "  --m/--n/--k: all three positive dimensions, each specified once;\n"
      << "              otherwise use the default square sweep\n"
      << "  warmup: untimed launches before benchmarking (default: 5)\n"
      << "  iters:  timed launches per matrix size (default: 50)\n"
      << "  seed:   random input seed (default: 1234)\n"
      << "  csv:    write one result row per matrix size to FILE\n"
      << "  alpha:  finite GEMM multiplier (default: 0.5)\n"
      << "  beta:   finite initial-C multiplier (default: 3.0)\n"
      << "  C is reset before every launch; reset copies are not timed.\n";
}

int parse_integer(const std::string &value, const char *option) {
  std::size_t parsed = 0;
  int result = 0;
  try {
    result = std::stoi(value, &parsed);
  } catch (const std::exception &) {
    throw std::invalid_argument(std::string("Invalid value for ") + option +
                                ": " + value);
  }
  if (parsed != value.size() ||
      value.find_first_of(" \t\r\n\f\v") != std::string::npos) {
    throw std::invalid_argument(std::string("Invalid value for ") + option +
                                ": " + value);
  }
  return result;
}

float parse_scalar(const std::string &value, const char *option) {
  std::size_t parsed = 0;
  float result = 0.0f;
  try {
    result = std::stof(value, &parsed);
  } catch (const std::exception &) {
    throw std::invalid_argument(std::string("Invalid value for ") + option +
                                ": " + value);
  }
  if (parsed != value.size() || !std::isfinite(result) ||
      value.find_first_of(" \t\r\n\f\v") != std::string::npos) {
    throw std::invalid_argument(std::string("Invalid finite value for ") +
                                option + ": " + value);
  }
  return result;
}

unsigned int parse_seed(const std::string &value) {
  std::size_t parsed = 0;
  unsigned long result = 0;
  try {
    result = std::stoul(value, &parsed);
  } catch (const std::exception &) {
    throw std::invalid_argument("Invalid value for --seed: " + value);
  }
  if (parsed != value.size() ||
      value.find_first_of("- \t\r\n\f\v") != std::string::npos ||
      result > std::numeric_limits<unsigned int>::max()) {
    throw std::invalid_argument("Invalid value for --seed: " + value);
  }
  return static_cast<unsigned int>(result);
}

Options parse_options(int argc, char **argv) {
  Options options;
  bool kernel_specified = false;
  bool m_specified = false, n_specified = false, k_specified = false;
  auto set_kernel = [&](const std::string &token) {
    if (kernel_specified) {
      throw std::invalid_argument("Specify the kernel only once");
    }
    options.requested_kernel = token;
    if (token == "auto") {
      options.kernel = kAutoKernel;
    } else if (const KernelDescriptor *descriptor = find_kernel(token)) {
      options.kernel = descriptor->id;
    } else {
      // Keep the legacy integer spelling accepted by std::stoi (notably
      // leading-zero IDs and +N in --kernel), while names remain exact.
      try {
        const int numeric_id = parse_integer(token, "kernel");
        if (const KernelDescriptor *descriptor = find_kernel(numeric_id)) {
          options.kernel = descriptor->id;
          kernel_specified = true;
          return;
        }
      } catch (const std::exception &) {
        // Report a consistent unknown-kernel diagnostic below.
      }
      throw std::invalid_argument("Unknown kernel: " + token +
                                  "; use --list-kernels for valid IDs/names");
    }
    kernel_specified = true;
  };
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--help" || argument == "-h") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }

    auto require_value = [&](const char *option) -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument(std::string("Missing value for ") + option);
      }
      return argv[++i];
    };

    if (argument == "--list-kernels") {
      if (argc != 2) {
        throw std::invalid_argument("--list-kernels must be used standalone");
      }
      options.list_kernels = true;
      return options;
    } else if (argument == "--kernel") {
      set_kernel(require_value("--kernel"));
    } else if (argument == "--m" || argument == "--n" || argument == "--k") {
      bool &specified = argument == "--m" ? m_specified
                        : argument == "--n" ? n_specified : k_specified;
      int &dimension = argument == "--m" ? options.shape.m
                       : argument == "--n" ? options.shape.n : options.shape.k;
      if (specified) {
        throw std::invalid_argument("Specify " + argument + " only once");
      }
      dimension = parse_integer(require_value(argument.c_str()), argument.c_str());
      if (dimension <= 0) {
        throw std::invalid_argument(argument + " must be positive");
      }
      specified = true;
    } else if (argument == "--warmup") {
      options.warmup = parse_integer(require_value("--warmup"), "--warmup");
    } else if (argument == "--iters") {
      options.iterations = parse_integer(require_value("--iters"), "--iters");
    } else if (argument == "--seed") {
      options.seed = parse_seed(require_value("--seed"));
    } else if (argument == "--alpha") {
      options.alpha = parse_scalar(require_value("--alpha"), "--alpha");
    } else if (argument == "--beta") {
      options.beta = parse_scalar(require_value("--beta"), "--beta");
    } else if (argument == "--csv") {
      options.csv_path = require_value("--csv");
      if (options.csv_path.empty()) {
        throw std::invalid_argument("--csv requires a non-empty path");
      }
    } else if (!argument.empty() && argument.front() != '-' && !kernel_specified) {
      set_kernel(argument);
    } else {
      throw std::invalid_argument("Unknown argument: " + argument);
    }
  }

  if (!kernel_specified) {
    throw std::invalid_argument("Please select a kernel ID, name, or auto");
  }
  options.custom_shape = m_specified || n_specified || k_specified;
  if (options.custom_shape && !(m_specified && n_specified && k_specified)) {
    throw std::invalid_argument("--m, --n, and --k must all be specified together");
  }
  if (options.custom_shape) {
    if (const char *error = problem_shape_error(options.shape)) {
      throw std::invalid_argument(error);
    }
  }
  if (options.warmup < 0) {
    throw std::invalid_argument("--warmup must be non-negative");
  }
  if (options.iterations <= 0) {
    throw std::invalid_argument("--iters must be greater than zero");
  }
  return options;
}

void list_kernels() {
  std::cout << "Available kernels (IDs and canonical names):\n";
  for (const KernelDescriptor &kernel : kernel_registry()) {
    std::cout << "  " << kernel.id << "  " << kernel.name << " - "
              << kernel.description << "\n    shape: " << kernel.shape_requirements;
    if (kernel.minimum_compute_capability) {
      std::cout << "; minimum compute capability: "
                << kernel.minimum_compute_capability;
    }
    std::cout << "; pointer alignment: " << kernel.pointer_alignment << " bytes\n";
  }
  std::cout << "  auto - deterministic heuristic policy: warp-tiled (10), "
               "vectorized (6), block-2d (5), then cublas-fp32 (0).\n"
               "         First supported candidate, not measured tuning.\n";
}

std::string csv_quote(const std::string &value) {
  std::string quoted = "\"";
  for (char character : value) {
    if (character == '"') quoted += '"';
    quoted += character;
  }
  return quoted + '"';
}

struct BenchmarkJob {
  GemmShape shape;
  KernelSelection selection;
};

int main(int argc, char **argv) try {
  Options options;
  try {
    options = parse_options(argc, argv);
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n\n";
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }
  if (options.list_kernels) {
    list_kernels();
    return EXIT_SUCCESS;
  }

  int deviceIdx = 0;
  if (const char *device = std::getenv("DEVICE")) {
    deviceIdx = parse_integer(device, "DEVICE");
  }
  if (deviceIdx < 0) {
    throw std::invalid_argument("DEVICE must be non-negative");
  }
  int deviceCount = 0;
  cudaCheck(cudaGetDeviceCount(&deviceCount));
  if (deviceIdx >= deviceCount) {
    throw std::invalid_argument(
        "DEVICE is outside the available CUDA device range");
  }
  cudaCheck(cudaSetDevice(deviceIdx));
  cudaDeviceProp properties{};
  cudaCheck(cudaGetDeviceProperties(&properties, deviceIdx));
  const DeviceCapabilities device{
      properties.major * 10 + properties.minor, properties.maxThreadsPerBlock,
      properties.sharedMemPerBlock, properties.maxGridSize[0],
      properties.maxGridSize[1]};

  // Preflight every requested workload before handles, events, or buffers.
  // Explicit shapes fail rather than silently skip or substitute a kernel.
  std::vector<GemmShape> shapes;
  if (options.custom_shape) {
    shapes.push_back(options.shape);
  } else {
    for (int size : {128, 256, 512, 1024, 2048, 4096}) {
      shapes.push_back({size, size, size});
    }
  }
  std::vector<BenchmarkJob> jobs;
  for (GemmShape shape : shapes) {
    try {
      jobs.push_back({shape, select_kernel(options.kernel, shape, device)});
    } catch (const std::exception &error) {
      if (options.custom_shape) throw;
      std::cout << "Skipping kernel " << options.kernel << " ("
                << options.requested_kernel << ") at size " << shape.m << ": "
                << error.what() << '\n';
    }
  }
  if (jobs.empty()) {
    throw std::runtime_error("No runnable benchmark cases for requested kernel " +
                             options.requested_kernel);
  }

  if (options.kernel != kAutoKernel) {
    // Preserve the legacy banner for numeric explicit requests.
    printf("Running kernel %d on device %d.\n", options.kernel, deviceIdx);
  } else {
    std::cout << "Running requested kernel auto on device " << deviceIdx
              << ".\n";
  }
  printf("Configuration: warmup=%d, iterations=%d, seed=%u\n", options.warmup,
         options.iterations, options.seed);

  std::cout << "Timing policy: reset C before every launch; reset copies are "
               "excluded from GEMM timing.\n";

  cublasHandle_t handle = nullptr;
  check_cublas(cublasCreate(&handle), "cublasCreate");
  // All kernels, reset copies, and timing events use the default stream.
  check_cublas(cublasSetStream(handle, nullptr), "cublasSetStream");

  cudaEvent_t beg = nullptr, end = nullptr;
  cudaCheck(cudaEventCreate(&beg));
  cudaCheck(cudaEventCreate(&end));

  // Preserve one maximum-sized, single-seed baseline for the default sweep.
  // A custom workload instead allocates its exact MK, KN, and MN counts.
  const GemmShape allocation_shape = shapes.back();
  const std::size_t a_elements =
      static_cast<std::size_t>(allocation_shape.m) * allocation_shape.k;
  const std::size_t b_elements =
      static_cast<std::size_t>(allocation_shape.k) * allocation_shape.n;
  const std::size_t c_elements =
      static_cast<std::size_t>(allocation_shape.m) * allocation_shape.n;
  const std::size_t a_bytes = sizeof(float) * a_elements;
  const std::size_t b_bytes = sizeof(float) * b_elements;
  const std::size_t c_bytes = sizeof(float) * c_elements;
  if (!options.custom_shape) {
    std::cout << "Max size: " << allocation_shape.m << std::endl;
  }

  const float alpha = options.alpha, beta = options.beta;
  float *dA = nullptr, *dB = nullptr, *dC_initial = nullptr, *dC = nullptr,
        *dC_ref = nullptr;

  std::ofstream csv;
  if (!options.csv_path.empty()) {
    csv.open(options.csv_path);
    if (!csv) {
      throw std::runtime_error("Unable to open CSV output file: " +
                               options.csv_path);
    }
    csv << "kernel,size,average_seconds,gflops,warmup,iters,seed,verified,"
           "alpha,beta,m,n,k,kernel_name,requested_kernel,selection_reason\n";
  }

  std::srand(options.seed);
  auto random_matrix = [&](std::size_t elements) {
    std::vector<float> matrix(elements);
    randomize_matrix(matrix.data(), static_cast<int>(elements));
    return matrix;
  };
  const std::vector<float> A = random_matrix(a_elements);
  const std::vector<float> B = random_matrix(b_elements);
  // Keep the initial C immutable and separate from both validation outputs.
  const std::vector<float> C_initial = random_matrix(c_elements);
  std::vector<float> C(c_elements), C_ref(c_elements);

  cudaCheck(cudaMalloc((void **)&dA, a_bytes));
  cudaCheck(cudaMalloc((void **)&dB, b_bytes));
  cudaCheck(cudaMalloc((void **)&dC_initial, c_bytes));
  cudaCheck(cudaMalloc((void **)&dC, c_bytes));
  cudaCheck(cudaMalloc((void **)&dC_ref, c_bytes));

  cudaCheck(cudaMemcpy(dA, A.data(), a_bytes, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B.data(), b_bytes, cudaMemcpyHostToDevice));
  // dC_initial is uploaded once and is never passed to a GEMM as its output.
  cudaCheck(cudaMemcpy(dC_initial, C_initial.data(), c_bytes,
                       cudaMemcpyHostToDevice));

  for (const BenchmarkJob &job : jobs) {
    const int m = job.shape.m, n = job.shape.n, k = job.shape.k;
    const int kernel_num = job.selection.id;
    const KernelDescriptor &selected = *find_kernel(kernel_num);
    std::cout << "Requested kernel " << options.requested_kernel
              << "; selected kernel " << kernel_num << " (" << selected.name
              << ") for M=" << m << ", N=" << n << ", K=" << k << ": "
              << job.selection.reason << '\n';
    const std::size_t bytes = sizeof(float) * static_cast<std::size_t>(m) * n;
    auto reset_output = [&](float *output) {
      cudaCheck(cudaMemcpyAsync(output, dC_initial, bytes,
                                cudaMemcpyDeviceToDevice, nullptr));
    };
    auto launch = [&](int kernel, float *output) {
      run_kernel(kernel, m, n, k, alpha, dA, dB, beta, output, handle);
      cudaCheck(cudaGetLastError());
    };

    if (m == n && n == k) {
      std::cout << "dimensions(m=n=k) " << m;
    } else {
      std::cout << "dimensions(m,n,k) " << m << ',' << n << ',' << k;
    }
    std::cout << ", alpha: " << alpha << ", beta: " << beta << std::endl;
    // A larger size must not inherit either output from the preceding size.
    // Validation starts both implementations from the original C as well.
    // cuBLAS is the reference here, not an independently verified result.
    bool verified = false;
    if (kernel_num != 0) {
      reset_output(dC_ref);
      reset_output(dC);
      launch(0, dC_ref);
      launch(kernel_num, dC);
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(C_ref.data(), dC_ref, bytes, cudaMemcpyDeviceToHost));

      if (!verify_matrix(C_ref.data(), C.data(), m * n)) {
        std::cout
            << "Failed to pass the correctness verification against NVIDIA "
               "cuBLAS."
            << std::endl;
        if (m <= 128 && n <= 128 && k <= 128) {
          std::cout << " Logging faulty output into " << errLogFile << "\n";
          std::ofstream fs(errLogFile);
          if (!fs) {
            throw std::runtime_error("Unable to open validation log: " +
                                     errLogFile);
          }
          fs << "A:\n";
          print_matrix(A.data(), m, k, fs);
          fs << "B:\n";
          print_matrix(B.data(), k, n, fs);
          fs << "Initial C:\n";
          print_matrix(C_initial.data(), m, n, fs);
          fs << "C:\n";
          print_matrix(C.data(), m, n, fs);
          fs << "Should:\n";
          print_matrix(C_ref.data(), m, n, fs);
          fs.flush();
          if (!fs) {
            throw std::runtime_error("Unable to write validation log: " +
                                     errLogFile);
          }
        }
        throw std::runtime_error("Kernel correctness verification failed");
      }
      verified = true;
    }

    for (int j = 0; j < options.warmup; ++j) {
      reset_output(dC);
      launch(kernel_num, dC);
    }
    cudaCheck(cudaDeviceSynchronize());

    double elapsed_seconds = 0.0;
    for (int j = 0; j < options.iterations; ++j) {
      // Queue the reset BEFORE the start event on the same stream. Only the
      // GEMM is between the events, and beta always multiplies the original C.
      reset_output(dC);
      cudaCheck(cudaEventRecord(beg, nullptr));
      launch(kernel_num, dC);
      cudaCheck(cudaEventRecord(end, nullptr));
      cudaCheck(cudaEventSynchronize(end));
      float elapsed_ms = 0.0f;
      cudaCheck(cudaEventElapsedTime(&elapsed_ms, beg, end));
      elapsed_seconds += static_cast<double>(elapsed_ms) / 1000.0;
    }
    if (!std::isfinite(elapsed_seconds) || elapsed_seconds <= 0.0) {
      throw std::runtime_error("CUDA events reported an invalid elapsed time");
    }

    const double flops = 2.0 * m * n * k;
    const double average_seconds = elapsed_seconds / options.iterations;
    const double gflops = flops * 1e-9 / average_seconds;
    if (m == n && n == k) {
      // Keep the legacy square timing line for existing parsers.
      printf("Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. "
             "size: (%d).\n", average_seconds, gflops, m);
    } else {
      printf("Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. "
             "dimensions: (m=%d, n=%d, k=%d).\n", average_seconds, gflops, m, n, k);
    }
    if (csv.is_open()) {
      csv << kernel_num << ',';
      if (m == n && n == k) csv << m;
      csv << ',' << std::fixed << std::setprecision(9)
          << average_seconds << ',' << std::setprecision(3) << gflops << ','
          << options.warmup << ',' << options.iterations << ',' << options.seed
          << ',' << (verified ? "true" : "false") << ',' << std::defaultfloat
          << std::setprecision(std::numeric_limits<float>::max_digits10) << alpha
          << ',' << beta << ',' << m << ',' << n << ',' << k << ','
          << csv_quote(selected.name) << ',' << csv_quote(options.requested_kernel)
          << ',' << csv_quote(job.selection.reason) << '\n';
      csv.flush();
      if (!csv) {
        throw std::runtime_error("Unable to write CSV output file: " +
                                 options.csv_path);
      }
    }
    fflush(stdout);
  }

  cudaCheck(cudaFree(dA));
  cudaCheck(cudaFree(dB));
  cudaCheck(cudaFree(dC_initial));
  cudaCheck(cudaFree(dC));
  cudaCheck(cudaFree(dC_ref));
  cudaCheck(cudaEventDestroy(beg));
  cudaCheck(cudaEventDestroy(end));
  check_cublas(cublasDestroy(handle), "cublasDestroy");
  if (csv.is_open()) {
    csv.close();
    if (!csv) {
      throw std::runtime_error("Unable to close CSV output file: " +
                               options.csv_path);
    }
  }

  return EXIT_SUCCESS;
} catch (const std::exception &error) {
  std::cerr << "Benchmark failed: " << error.what() << '\n';
  return EXIT_FAILURE;
}
