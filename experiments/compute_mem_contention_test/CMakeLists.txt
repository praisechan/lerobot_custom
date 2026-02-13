cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(compute_mem_contention_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Executable
add_executable(compute_mem_contention_test src/main.cu)

# Set CUDA architectures (adjust for your GPU)
# For Hopper (H100): 90, for Ampere (A100): 80
set_target_properties(compute_mem_contention_test PROPERTIES
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
)

# Include directories
target_include_directories(compute_mem_contention_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link CUDA driver library (required for Green Contexts)
target_link_libraries(compute_mem_contention_test PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
)

# Compiler flags
target_compile_options(compute_mem_contention_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -O3
        --use_fast_math
        -lineinfo
    >
)

# Install
install(TARGETS compute_mem_contention_test DESTINATION bin)
