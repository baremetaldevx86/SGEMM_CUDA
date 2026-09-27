#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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
  int kernel = -1;
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
      << " <kernel> [--warmup N] [--iters N] [--seed N] [--csv FILE]"
         " [--alpha X] [--beta X]\n"
      << "       " << program
      << " --kernel N [--warmup N] [--iters N] [--seed N] [--csv FILE]"
         " [--alpha X] [--beta X]\n"
      << "  kernel: 0-12 (0 selects NVIDIA cuBLAS)\n"
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

    if (argument == "--kernel") {
      if (kernel_specified) {
        throw std::invalid_argument("Specify the kernel only once");
      }
      options.kernel = parse_integer(require_value("--kernel"), "--kernel");
      kernel_specified = true;
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
      options.kernel = parse_integer(argument, "kernel");
      kernel_specified = true;
    } else {
      throw std::invalid_argument("Unknown argument: " + argument);
    }
  }

  if (options.kernel < 0 || options.kernel > 12) {
    throw std::invalid_argument("Please select a kernel in the range 0-12");
  }
  if (options.warmup < 0) {
    throw std::invalid_argument("--warmup must be non-negative");
  }
  if (options.iterations <= 0) {
    throw std::invalid_argument("--iters must be greater than zero");
  }
  return options;
}

int main(int argc, char **argv) try {
  Options options;
  try {
    options = parse_options(argc, argv);
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n\n";
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }
  const int kernel_num = options.kernel;

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

  printf("Running kernel %d on device %d.\n", kernel_num, deviceIdx);
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

  // cuBLAS FLOPs ceiling is reached at 8192
  const std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};
  const int max_size = sizes.back();
  const std::size_t max_elements = static_cast<std::size_t>(max_size) * max_size;
  const std::size_t max_bytes = sizeof(float) * max_elements;
  std::cout << "Max size: " << max_size << std::endl;

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
           "alpha,beta\n";
  }

  std::srand(options.seed);
  auto random_matrix = [&]() {
    std::vector<float> matrix(max_elements);
    randomize_matrix(matrix.data(), static_cast<int>(max_elements));
    return matrix;
  };
  const std::vector<float> A = random_matrix();
  const std::vector<float> B = random_matrix();
  // Keep the initial C immutable and separate from both validation outputs.
  const std::vector<float> C_initial = random_matrix();
  std::vector<float> C(max_elements), C_ref(max_elements);

  cudaCheck(cudaMalloc((void **)&dA, max_bytes));
  cudaCheck(cudaMalloc((void **)&dB, max_bytes));
  cudaCheck(cudaMalloc((void **)&dC_initial, max_bytes));
  cudaCheck(cudaMalloc((void **)&dC, max_bytes));
  cudaCheck(cudaMalloc((void **)&dC_ref, max_bytes));

  cudaCheck(cudaMemcpy(dA, A.data(), max_bytes, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B.data(), max_bytes, cudaMemcpyHostToDevice));
  // dC_initial is uploaded once and is never passed to a GEMM as its output.
  cudaCheck(cudaMemcpy(dC_initial, C_initial.data(), max_bytes,
                       cudaMemcpyHostToDevice));

  for (int size : sizes) {
    const int m = size, n = size, k = size;
    if (const char *error = kernel_shape_error(kernel_num, m, n, k)) {
      std::cout << "Skipping kernel " << kernel_num << " at size " << size
                << ": " << error << '\n';
      continue; // Do not emit a timing row or silently substitute another kernel.
    }
    const std::size_t bytes = sizeof(float) * static_cast<std::size_t>(m) * n;
    auto reset_output = [&](float *output) {
      cudaCheck(cudaMemcpyAsync(output, dC_initial, bytes,
                                cudaMemcpyDeviceToDevice, nullptr));
    };
    auto launch = [&](int kernel, float *output) {
      run_kernel(kernel, m, n, k, alpha, dA, dB, beta, output, handle);
      cudaCheck(cudaGetLastError());
    };

    std::cout << "dimensions(m=n=k) " << m << ", alpha: " << alpha
              << ", beta: " << beta << std::endl;
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
        if (m <= 128) {
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
    printf(
        "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
        "(%d).\n",
        average_seconds, gflops, m);
    if (csv.is_open()) {
      csv << kernel_num << ',' << m << ',' << std::fixed << std::setprecision(9)
          << average_seconds << ',' << std::setprecision(3) << gflops << ','
          << options.warmup << ',' << options.iterations << ',' << options.seed
          << ',' << (verified ? "true" : "false") << ',' << std::defaultfloat
          << std::setprecision(std::numeric_limits<float>::max_digits10) << alpha
          << ',' << beta << '\n';
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
