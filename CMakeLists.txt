cmake_minimum_required(VERSION 3.20)
project(cache_evict LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Require sm_80+ (Ampere and newer)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)

find_package(CUDAToolkit REQUIRED)

# Common CUDA flags
set(CUDA_NVCC_FLAGS
    -O3
    --use_fast_math
    -Xcompiler -Wall
    -Xcompiler -O3
    --expt-relaxed-constexpr
)

# Static library of core kernels
add_library(cache_evict_lib STATIC
    src/reference_attn.cu
    src/fused_attn_evict.cu
)
target_include_directories(cache_evict_lib PUBLIC src)
target_compile_options(cache_evict_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

# Correctness tests
add_executable(test_correctness tests/test_correctness.cu)
target_link_libraries(test_correctness PRIVATE cache_evict_lib CUDA::cudart)
target_include_directories(test_correctness PRIVATE src)
target_compile_options(test_correctness PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

# Latency benchmark
add_executable(bench_latency benchmarks/bench_latency.cu)
target_link_libraries(bench_latency PRIVATE cache_evict_lib CUDA::cudart)
target_include_directories(bench_latency PRIVATE src)
target_compile_options(bench_latency PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

# Memory-quality tradeoff benchmark
add_executable(bench_tradeoff benchmarks/bench_tradeoff.cu)
target_link_libraries(bench_tradeoff PRIVATE cache_evict_lib CUDA::cudart)
target_include_directories(bench_tradeoff PRIVATE src)
target_compile_options(bench_tradeoff PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

enable_testing()
add_test(NAME correctness COMMAND test_correctness)
