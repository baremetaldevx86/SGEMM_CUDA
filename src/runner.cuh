#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iosfwd>

void cudaCheck(cudaError_t error, const char *file, int line);
void CudaDeviceInfo(); // Print information for the current CUDA device.

void range_init_matrix(float *mat, int N);
void randomize_matrix(float *mat, int N);
void zero_init_matrix(float *mat, int N);
// Throws std::invalid_argument for a negative count or null non-empty buffers.
void copy_matrix(const float *src, float *dest, int N);
void print_matrix(const float *A, int M, int N, std::ofstream &fs);
bool verify_matrix(float *mat1, float *mat2, int N);

// Legacy wall-clock timestamps are in microseconds, despite these names.
// Prefer CUDA events for GPU timing; these float timestamps lose precision.
float get_sec();
float get_current_sec(); // Backward-compatible alias of get_sec().
// Return the difference between two legacy timestamps in seconds.
float cpu_elapsed_time(float &beg, float &end);

// Row-major GEMM wrappers. The handle must use host pointer mode for alpha/beta
// (the cuBLAS default). These asynchronous calls fail fast on cuBLAS API errors;
// callers remain responsible for checking execution errors at synchronization.
void runCublasFP32(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C);
void runCublasBF16(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C);
void runCublasTF32(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C);

// Returns nullptr for a supported shape, otherwise a diagnostic. This checks
// the current fixed runner presets without accessing the GPU. All dimensions
// must be positive and row-major element counts must fit 32-bit indexing.
const char *kernel_shape_error(int kernel_num, int m, int n, int k);

// Kernel IDs are stable: 0 selects cuBLAS FP32, and 1-12 select custom kernels.
// Throws std::invalid_argument before launch for unsupported shapes or null /
// misaligned pointers. Allocation sizes and non-aliasing remain caller contracts.
void run_kernel(int kernel_num, int m, int n, int k, float alpha, float *A,
                float *B, float beta, float *C, cublasHandle_t handle);
