cmake_minimum_required(VERSION 3.15)
project(SIMD_Multiply_Lab VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -mmmx -msse4.2 -mavx2")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

add_executable(simd_lab
    src/main.cpp
    src/scalar.cpp
    src/mmx.cpp
    src/sse.cpp
    src/avx2.cpp
)

target_include_directories(simd_lab PRIVATE include)