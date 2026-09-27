#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t error = (call);                                           \
    if (error != cudaSuccess) {                                                \
      std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #call,  \
                   cudaGetErrorString(error));                                 \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    const cublasStatus_t status = (call);                                       \
    if (status != CUBLAS_STATUS_SUCCESS) {                                      \
      std::fprintf(stderr, "%s:%d: %s failed: cuBLAS status %d\n", __FILE__,     \
                   __LINE__, #call, static_cast<int>(status));                 \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

/*
 * A stand-alone script to invoke & benchmark standard cuBLAS SGEMM performance
 */

int main() {
  int m = 2;
  int k = 3;
  int n = 4;
  int print = 1;
  cublasHandle_t handle; // cuBLAS context

  int i, j;

  float *a, *b, *c;

  // malloc for a,b,c...
  a = (float *)malloc(m * k * sizeof(float));
  b = (float *)malloc(k * n * sizeof(float));
  c = (float *)malloc(m * n * sizeof(float));
  if (a == nullptr || b == nullptr || c == nullptr) {
    std::fprintf(stderr, "Failed to allocate host matrices\n");
    free(a);
    free(b);
    free(c);
    return EXIT_FAILURE;
  }

  int ind = 11;
  for (j = 0; j < m * k; j++) {
    a[j] = (float)ind++;
  }

  ind = 11;
  for (j = 0; j < k * n; j++) {
    b[j] = (float)ind++;
  }

  ind = 11;
  for (j = 0; j < m * n; j++) {
    c[j] = (float)ind++;
  }

  // DEVICE
  float *d_a, *d_b, *d_c;

  // cudaMalloc for d_a, d_b, d_c...
  CUDA_CHECK(cudaMalloc((void **)&d_a, m * k * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&d_b, k * n * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&d_c, m * n * sizeof(float)));

  CUBLAS_CHECK(cublasCreate(&handle)); // initialize CUBLAS context

  CUDA_CHECK(cudaMemcpy(d_a, a, m * k * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b, k * n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_c, c, m * n * sizeof(float), cudaMemcpyHostToDevice));

  float alpha = 1.0f;
  float beta = 0.5f;

  if (print == 1) {
    printf("alpha = %4.1f, beta = %4.1f\n", alpha, beta);
    printf("A = (mxk: %d x %d)\n", m, k);
    for (i = 0; i < m; i++) {
      for (j = 0; j < k; j++) {
        printf("%4.1f ", a[i * k + j]);
      }
      printf("\n");
    }
    printf("B = (kxn: %d x %d)\n", k, n);
    for (i = 0; i < k; i++) {
      for (j = 0; j < n; j++) {
        printf("%4.1f ", b[i * n + j]);
      }
      printf("\n");
    }
    printf("C = (mxn: %d x %d)\n", m, n);
    for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++) {
        printf("%4.1f ", c[i * n + j]);
      }
      printf("\n");
    }
  }

  // cuBLAS is column-major: C^T = B^T * A^T implements row-major C = A * B.
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                          d_b, n, d_a, k, &beta, d_c, n));
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(c, d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost));

  if (print == 1) {
    printf("\nC after SGEMM = \n");
    for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++) {
        printf("%4.1f ", c[i * n + j]);
      }
      printf("\n");
    }
  }

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  CUBLAS_CHECK(cublasDestroy(handle)); // destroy CUBLAS context
  free(a);
  free(b);
  free(c);

  return EXIT_SUCCESS;
}