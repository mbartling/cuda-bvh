cmake_minimum_required(VERSION 2.8)

find_package(CUDA QUIET REQUIRED)
#set(
#    CUDA_NVCC_FLAGS
#    ${CUDA_NVCC_FLAGS};
#    -O3 -std=c++11 -gencode arch=compute_35,code=sm_35
#    )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;-std=c++11;-O3;-DVERBOSE")

# Specify include directories
include_directories(
    kernels
    inc
    )
file(GLOB SOURCES src/*.cpp src/*.cc)
file(GLOB CUDA_SOURCES kernels/*.cu)

cuda_add_executable(cuda_bvh ${SOURCES} ${CUDA_SOURCES})
#leave this out for now
#add_executable(cpu_bvh ${SOURCES})
