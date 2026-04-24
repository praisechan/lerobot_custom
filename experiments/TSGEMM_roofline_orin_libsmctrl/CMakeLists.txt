cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(TSGEMM_roofline_orin_libsmctrl LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(LIBSMCTRL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/BulletServe/csrc")

if(NOT EXISTS "${LIBSMCTRL_DIR}/src/libsmctrl.h")
    message(FATAL_ERROR "libsmctrl not found at ${LIBSMCTRL_DIR}")
endif()

add_library(smctrl STATIC
    ${LIBSMCTRL_DIR}/src/libsmctrl_core.c
    ${LIBSMCTRL_DIR}/src/libsmctrl_validator.cu
)

target_include_directories(smctrl PUBLIC
    ${LIBSMCTRL_DIR}/src
)

target_link_libraries(smctrl PUBLIC CUDA::cuda_driver)

target_compile_options(smctrl PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>
)

add_executable(TSGEMM_roofline_orin_libsmctrl
    src/main.cu
)

target_include_directories(TSGEMM_roofline_orin_libsmctrl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${LIBSMCTRL_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(TSGEMM_roofline_orin_libsmctrl PRIVATE
    smctrl
    CUDA::cudart
    CUDA::cuda_driver
)

set_property(TARGET smctrl PROPERTY CUDA_ARCHITECTURES native)
set_property(TARGET TSGEMM_roofline_orin_libsmctrl PROPERTY CUDA_ARCHITECTURES native)

target_compile_options(TSGEMM_roofline_orin_libsmctrl PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

install(TARGETS TSGEMM_roofline_orin_libsmctrl DESTINATION bin)
