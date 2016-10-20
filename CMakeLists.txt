cmake_minimum_required(VERSION 2.8)

find_package(CUDA QUIET REQUIRED)
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_35,code=sm_35
    )

# Specify include directories
include_directories(
    kernels
    inc
    )
file(GLOB SOURCES src/*.cpp src/*.cc)
file(GLOB CUDA_SOURCES kernels/*.cu)

cuda_add_executable(cuda_bvh ${SOURCES} ${CUDA_SOURCES})
add_executable(cpu_bvh ${SOURCES})
