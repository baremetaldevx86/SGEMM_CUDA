// Compile the actual benchmark as ordinary C++, resolving <runner.cuh> to the
// mock in this directory. Do not link src/runner.cu or any CUDA libraries.
#include "../../sgemm.cu"
