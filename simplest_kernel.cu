#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t error = (call);                                           \
    if (error != cudaSuccess) {                                                \
      std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #call,  \
                   cudaGetErrorString(error));                                 \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

__global__ void kernel(unsigned int *A, unsigned int *B, unsigned int size) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size * size) {
    A[index] = index / size;
    B[index] = index % size;
  }
}

int main() {
  unsigned int *Xs, *Ys;
  unsigned int *Xs_d, *Ys_d;

  constexpr unsigned int SIZE = 4;
  constexpr size_t bytes = SIZE * SIZE * sizeof(unsigned int);

  Xs = (unsigned int *)malloc(bytes);
  Ys = (unsigned int *)malloc(bytes);
  if (Xs == nullptr || Ys == nullptr) {
    std::fprintf(stderr, "Failed to allocate host matrices\n");
    free(Xs);
    free(Ys);
    return EXIT_FAILURE;
  }

  CUDA_CHECK(cudaMalloc((void **)&Xs_d, bytes));
  CUDA_CHECK(cudaMalloc((void **)&Ys_d, bytes));

  dim3 grid_size(1, 1, 1);
  dim3 block_size(SIZE * SIZE);

  // Every element is written by the kernel; no initial device contents are read.
  kernel<<<grid_size, block_size>>>(Xs_d, Ys_d, SIZE);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(Xs, Xs_d, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(Ys, Ys_d, bytes, cudaMemcpyDeviceToHost));

  for (unsigned int row = 0; row < SIZE; ++row) {
    for (unsigned int col = 0; col < SIZE; ++col) {
      std::cout << "[" << Xs[row * SIZE + col] << "|" << Ys[row * SIZE + col]
                << "] ";
    }
    std::cout << "\n";
  }

  CUDA_CHECK(cudaFree(Xs_d));
  CUDA_CHECK(cudaFree(Ys_d));
  free(Xs);
  free(Ys);
  return EXIT_SUCCESS;
}
