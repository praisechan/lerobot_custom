cmake_minimum_required(VERSION 3.18)
project(sm_bw_sweep LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(sm_bw_sweep
  src/main.cu
  src/green_context_utils.cu
)

target_include_directories(sm_bw_sweep PRIVATE src)

target_link_libraries(sm_bw_sweep PRIVATE CUDA::cuda_driver)

target_compile_options(sm_bw_sweep PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xptxas -dlcm=cg>
)

set_target_properties(sm_bw_sweep PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
