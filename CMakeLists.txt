cmake_minimum_required(VERSION 3.19)

# Keep the original Ampere default, but honor -DCMAKE_CUDA_ARCHITECTURES and
# CUDAARCHS. CMake initializes every CUDA target from this variable.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures to build")
endif()

project(NVIDIA_SGEMM_PRACTICE LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA is already enabled as a language; use the maintained toolkit package
# rather than the deprecated FindCUDA module.
find_package(CUDAToolkit 12 REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(sgemm sgemm.cu src/runner.cu src/kernel_registry.cpp)
target_include_directories(sgemm PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(sgemm PRIVATE CUDA::cudart CUDA::cublas)

add_executable(cuBLAS_sgemm cuBLAS_sgemm.cu)
target_link_libraries(cuBLAS_sgemm PRIVATE CUDA::cudart CUDA::cublas)

add_executable(simplest_kernel simplest_kernel.cu)
target_link_libraries(simplest_kernel PRIVATE CUDA::cudart)

include(CTest)
set(cuda_targets sgemm cuBLAS_sgemm simplest_kernel)
if(BUILD_TESTING)
    add_executable(sgemm_runner_tests tests/runner_tests.cu src/runner.cu
        src/kernel_registry.cpp)
    target_include_directories(sgemm_runner_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(sgemm_runner_tests PRIVATE CUDA::cudart CUDA::cublas)
    list(APPEND cuda_targets sgemm_runner_tests)
    add_test(NAME runner_host COMMAND sgemm_runner_tests --host)
    set_tests_properties(runner_host PROPERTIES LABELS "cpu")
    add_test(NAME runner_gpu COMMAND sgemm_runner_tests --gpu)
    set_tests_properties(runner_gpu PROPERTIES
        SKIP_RETURN_CODE 77 LABELS "gpu" RESOURCE_LOCK cuda_device_0 TIMEOUT 180)

    # Compile the actual CLI against a CPU-only runner model. CUDA headers and
    # libraries must not be added to this target's includes or link interface.
    add_executable(sgemm_kernel_registry_tests tests/kernel_registry_tests.cpp
        src/kernel_registry.cpp)
    target_include_directories(sgemm_kernel_registry_tests PRIVATE
        ${PROJECT_SOURCE_DIR}/src)
    add_test(NAME kernel_registry COMMAND sgemm_kernel_registry_tests)
    set_tests_properties(kernel_registry PROPERTIES LABELS "cpu" TIMEOUT 30)

    add_executable(sgemm_benchmark_mock tests/benchmark_mock/benchmark_main.cpp
        src/kernel_registry.cpp)
    target_include_directories(sgemm_benchmark_mock PRIVATE
        ${PROJECT_SOURCE_DIR}/tests/benchmark_mock
        ${PROJECT_SOURCE_DIR}/src)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME benchmark_mock COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/benchmark_mock/test_benchmark.py
            $<TARGET_FILE:sgemm_benchmark_mock>)
        set_tests_properties(benchmark_mock PROPERTIES LABELS "cpu" TIMEOUT 180)
        add_test(NAME bank_calc COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/test_bank_calc.py)
        set_tests_properties(bank_calc PROPERTIES LABELS "cpu" TIMEOUT 30)
        add_test(NAME plot_results COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/test_plot_results.py)
        set_tests_properties(plot_results PROPERTIES
            SKIP_RETURN_CODE 77 LABELS "cpu" TIMEOUT 30)
        add_test(NAME cli_e2e COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/test_cli_e2e.py
            $<TARGET_FILE:sgemm>)
        set_tests_properties(cli_e2e PROPERTIES
            SKIP_RETURN_CODE 77 LABELS "gpu" RESOURCE_LOCK cuda_device_0 TIMEOUT 300)
    else()
        message(WARNING "Python3 not found: benchmark lifecycle and bank calculator tests are unavailable")
    endif()
endif()

foreach(target IN LISTS cuda_targets)
    # Device debugging disables most optimizations and reduces performance.
    # Avoid -src-in-ptx: CUDA 12.4 can produce invalid PTX with source interleaving.
    target_compile_options(${target} PRIVATE
        "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
endforeach()
