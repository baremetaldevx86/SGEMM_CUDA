#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <runner.cuh>
#include <stdexcept>
#include <string>
#include <vector>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

const std::string errLogFile = "matrixValidationFailure.txt";

struct Options {
  int kernel = -1;
  int warmup = 5;
  int iterations = 50;
  unsigned int seed = 1234;
  std::string csv_path;
};

void print_usage(const char *program) {
  std::cerr
      << "Usage: " << program
      << " <kernel> [--warmup N] [--iters N] [--seed N] [--csv FILE]\n"
      << "       " << program
      << " --kernel N [--warmup N] [--iters N] [--seed N] [--csv FILE]\n"
      << "  kernel: 0-12 (0 selects NVIDIA cuBLAS)\n"
      << "  warmup: untimed launches before benchmarking (default: 5)\n"
      << "  iters:  timed launches per matrix size (default: 50)\n"
      << "  seed:   random input seed (default: 1234)\n"
      << "  csv:    write one result row per matrix size to FILE\n";
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
  if (parsed != value.size()) {
    throw std::invalid_argument(std::string("Invalid value for ") + option +
                                ": " + value);
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
  if (parsed != value.size() || result > std::numeric_limits<unsigned int>::max()) {
    throw std::invalid_argument("Invalid value for --seed: " + value);
  }
  return static_cast<unsigned int>(result);
}

Options parse_options(int argc, char **argv) {
  Options options;
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
      options.kernel = parse_integer(require_value("--kernel"), "--kernel");
    } else if (argument == "--warmup") {
      options.warmup = parse_integer(require_value("--warmup"), "--warmup");
    } else if (argument == "--iters") {
      options.iterations = parse_integer(require_value("--iters"), "--iters");
    } else if (argument == "--seed") {
      options.seed = parse_seed(require_value("--seed"));
    } else if (argument == "--csv") {
      options.csv_path = require_value("--csv");
    } else if (!argument.empty() && argument.front() != '-' && options.kernel < 0) {
      options.kernel = parse_integer(argument, "kernel");
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

int main(int argc, char **argv) {
  Options options;
  try {
    options = parse_options(argc, argv);
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n\n";
    print_usage(argv[0]);
    exit(EXIT_FAILURE);
  }
  const int kernel_num = options.kernel;

  // get environment variable for device
  int deviceIdx = 0;
  if (getenv("DEVICE") != NULL) {
    deviceIdx = atoi(getenv("DEVICE"));
  }
  cudaCheck(cudaSetDevice(deviceIdx));

  printf("Running kernel %d on device %d.\n", kernel_num, deviceIdx);
  printf("Configuration: warmup=%d, iterations=%d, seed=%u\n", options.warmup,
         options.iterations, options.seed);

  // print some device info
  // CudaDeviceInfo();

  // Declare the handle, create the handle, cublasCreate will return a value of
  // type cublasStatus_t to determine whether the handle was created
  // successfully (the value is 0)
  cublasHandle_t handle;
  if (cublasCreate(&handle)) {
    std::cerr << "Create cublas handle error." << std::endl;
    exit(EXIT_FAILURE);
  };

  // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
  // publishing event tasks in the target stream
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  // cuBLAS FLOPs ceiling is reached at 8192
  std::vector<int> SIZE = {128, 256, 512, 1024, 2048, 4096};

  long m, n, k, max_size;
  max_size = SIZE[SIZE.size() - 1];
  std::cout << "Max size: " << max_size << std::endl;

  float alpha = 0.5, beta = 3.0; // GEMM input parameters, C=α*AB+β*C

  float *A = nullptr, *B = nullptr, *C = nullptr,
        *C_ref = nullptr; // host matrices
  float *dA = nullptr, *dB = nullptr, *dC = nullptr,
        *dC_ref = nullptr; // device matrices

  std::ofstream csv;
  if (!options.csv_path.empty()) {
    csv.open(options.csv_path);
    if (!csv) {
      std::cerr << "Unable to open CSV output file: " << options.csv_path
                << std::endl;
      exit(EXIT_FAILURE);
    }
    csv << "kernel,size,average_seconds,gflops,warmup,iters,seed,verified\n";
  }

  std::srand(options.seed);
  A = (float *)malloc(sizeof(float) * max_size * max_size);
  B = (float *)malloc(sizeof(float) * max_size * max_size);
  C = (float *)malloc(sizeof(float) * max_size * max_size);
  C_ref = (float *)malloc(sizeof(float) * max_size * max_size);

  randomize_matrix(A, max_size * max_size);
  randomize_matrix(B, max_size * max_size);
  randomize_matrix(C, max_size * max_size);

  cudaCheck(cudaMalloc((void **)&dA, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dB, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(float) * max_size * max_size));

  cudaCheck(cudaMemcpy(dA, A, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC_ref, C, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));

  for (int size : SIZE) {
    m = n = k = size;

    std::cout << "dimensions(m=n=k) " << m << ", alpha: " << alpha
              << ", beta: " << beta << std::endl;
    // Verify the correctness of the calculation, and execute it once before the
    // kernel function timing to avoid cold start errors
    bool verified = kernel_num == 0;
    if (kernel_num != 0) {
      run_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref,
                 handle); // cuBLAS
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC,
                 handle); // Executes the kernel, modifies the result matrix
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
      cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
      cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

      if (!verify_matrix(C_ref, C, m * n)) {
        std::cout
            << "Failed to pass the correctness verification against NVIDIA "
               "cuBLAS."
            << std::endl;
        if (m <= 128) {
          std::cout << " Logging faulty output into " << errLogFile << "\n";
          std::ofstream fs;
          fs.open(errLogFile);
          fs << "A:\n";
          print_matrix(A, m, n, fs);
          fs << "B:\n";
          print_matrix(B, m, n, fs);
          fs << "C:\n";
          print_matrix(C, m, n, fs);
          fs << "Should:\n";
          print_matrix(C_ref, m, n, fs);
        }
        exit(EXIT_FAILURE);
      }
      verified = true;
    }

    for (int j = 0; j < options.warmup; ++j) {
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
    }
    cudaCheck(cudaDeviceSynchronize());
    // Start timed launches from the same C matrix every time. This also
    // prevents warmup launches from changing the benchmark's beta term.
    cudaCheck(cudaMemcpy(dC, dC_ref, sizeof(float) * m * n,
                         cudaMemcpyDeviceToDevice));

    cudaEventRecord(beg);
    for (int j = 0; j < options.iterations; j++) {
      // We don't reset dC between runs to save time
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.; // Convert to seconds

    long flops = 2 * m * n * k;
    printf(
        "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
        "(%ld).\n",
        elapsed_time / options.iterations,
        (options.iterations * flops * 1e-9) / elapsed_time, m);
    if (csv) {
      csv << kernel_num << ',' << m << ',' << std::fixed << std::setprecision(9)
          << (elapsed_time / options.iterations) << ',' << std::setprecision(3)
          << ((options.iterations * flops * 1e-9) / elapsed_time) << ','
          << options.warmup << ',' << options.iterations << ',' << options.seed
          << ',' << (verified ? "true" : "false") << '\n';
    }
    fflush(stdout);
    // make dC and dC_ref equal again (we modified dC while calling our kernel
    // for benchmarking)
    cudaCheck(cudaMemcpy(dC, dC_ref, sizeof(float) * m * n,
                         cudaMemcpyDeviceToDevice));
  }

  // Free up CPU and GPU space
  free(A);
  free(B);
  free(C);
  free(C_ref);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dC_ref);
  cublasDestroy(handle);

  return 0;
};
