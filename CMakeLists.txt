cmake_minimum_required(VERSION 3.10)

project(Parallel-Summation)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")

add_executable(main main.cpp)
