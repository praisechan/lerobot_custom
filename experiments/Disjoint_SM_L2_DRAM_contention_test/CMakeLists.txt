cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Disjoint_SM_L2_DRAM_contention_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(Disjoint_SM_L2_DRAM_contention_test src/main.cu)

set_target_properties(Disjoint_SM_L2_DRAM_contention_test PROPERTIES
    CUDA_ARCHITECTURES "80;90;100;120;121"
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(Disjoint_SM_L2_DRAM_contention_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(Disjoint_SM_L2_DRAM_contention_test PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
)

target_compile_options(Disjoint_SM_L2_DRAM_contention_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -O3
        --use_fast_math
        -lineinfo
    >
)

install(TARGETS Disjoint_SM_L2_DRAM_contention_test DESTINATION bin)
