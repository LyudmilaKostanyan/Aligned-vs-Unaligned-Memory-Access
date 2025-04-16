cmake_minimum_required(VERSION 3.10)

project(Aligned-vs-Unaligned-Memory-Access)

set(CMAKE_CXX_STANDARD 23)

if (MSVC)
    add_compile_options(/arch:AVX)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-mavx)
endif()

add_executable(main main.cpp)
