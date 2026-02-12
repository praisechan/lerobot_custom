cmake_minimum_required(VERSION 3.18)
project(sm_bw_sweep CUDA CXX C)

# Require C++14 for compatibility
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Path to libsmctrl source
set(LIBSMCTRL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/BulletServe/csrc")

# Check if libsmctrl exists
if(NOT EXISTS "${LIBSMCTRL_DIR}/src/libsmctrl.h")
    message(FATAL_ERROR "libsmctrl not found at ${LIBSMCTRL_DIR}")
endif()

# Build libsmctrl as a static library
add_library(smctrl STATIC
    ${LIBSMCTRL_DIR}/src/libsmctrl_core.c
    ${LIBSMCTRL_DIR}/src/libsmctrl_validator.cu
)

target_include_directories(smctrl PUBLIC
    ${LIBSMCTRL_DIR}/src
)

# Link libsmctrl against CUDA driver
target_link_libraries(smctrl PUBLIC CUDA::cuda_driver)

# Set compilation flags for libsmctrl
target_compile_options(smctrl PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>
)

# Main executable
add_executable(sm_bw_sweep
    src/main.cu
)

target_include_directories(sm_bw_sweep PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${LIBSMCTRL_DIR}/src
)

target_link_libraries(sm_bw_sweep PRIVATE
    smctrl
    CUDA::cudart
    CUDA::cuda_driver
)

# Set CUDA architecture (adjust based on your GPU)
# Auto-detect or set manually
set_property(TARGET sm_bw_sweep PROPERTY CUDA_ARCHITECTURES native)
set_property(TARGET smctrl PROPERTY CUDA_ARCHITECTURES native)

# Optimization flags
target_compile_options(sm_bw_sweep PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-dlcm=cg>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

# Enable verbose output for debugging
set(CMAKE_VERBOSE_MAKEFILE ON)

# Print configuration
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "libsmctrl directory: ${LIBSMCTRL_DIR}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
