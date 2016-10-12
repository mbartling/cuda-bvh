cmake_minimum_required(VERSION 2.8)
project(bvh)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Debug)

include_directories(../inc)
file(GLOB SOURCES "../src/*.cpp" "../src/*.cc")

add_executable(bvh ${SOURCES})
