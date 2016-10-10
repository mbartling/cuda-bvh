cmake_minimum_required(VERSION 2.8)

find_package(CUDA QUIET REQUIRED)

file(GLOB SOURCES src/*.cpp src/*.cc)
file(GLOB CUDA_SOURCES kernels/*.cu)

cuda_add_executable(cuda_bvh ${SOURCES} ${CUDA_SOURCES})
