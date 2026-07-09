cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_COMPILER)
    find_program(CMAKE_CUDA_COMPILER nvcc
        PATHS /usr/local/cuda/bin /usr/local/cuda-13.2/bin /usr/local/cuda-13/bin
        REQUIRED
    )
endif()

set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures")
project(sph_tsunami LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)
find_path(CCCL_INCLUDE_DIR thrust/version.h
    PATHS
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}/cccl
)

function(configure_sph_target target)
    target_include_directories(${target} PRIVATE include ${CCCL_INCLUDE_DIR})
    set_property(TARGET ${target} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(${target} PRIVATE CUDA::cudart)
    if(MSVC)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
        )
    endif()
endfunction()

add_executable(sph_tsunami src/main.cu src/kernels.cu)
configure_sph_target(sph_tsunami)
target_compile_options(sph_tsunami PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
        --generate-line-info
        --maxrregcount=64
    >
)

add_executable(sph_debug src/main.cu src/kernels.cu)
configure_sph_target(sph_debug)
target_compile_options(sph_debug PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-G -g>
)
target_compile_definitions(sph_debug PRIVATE DEBUG_BUILD)

install(TARGETS sph_tsunami DESTINATION bin)
