cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(sm_bw_sweep LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Executable
add_executable(sm_bw_sweep src/main.cu)

# Set CUDA architectures (adjust for your GPU)
# For Hopper (H100): 90, for Ampere (A100): 80
set_target_properties(sm_bw_sweep PROPERTIES
    CUDA_ARCHITECTURES "80;90"
    CUDA_SEPARABLE_COMPILATION ON
)

# Include directories
target_include_directories(sm_bw_sweep PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link CUDA driver library (required for Green Contexts)
target_link_libraries(sm_bw_sweep PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
)

# Compiler flags
target_compile_options(sm_bw_sweep PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -O3
        --use_fast_math
        -lineinfo
    >
)

# Install
install(TARGETS sm_bw_sweep DESTINATION bin)
