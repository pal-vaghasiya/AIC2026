# CMakeLists.txt - Build script for C++ Native ML Engine of ControlPlane.ai
#
# RESPONSIBILITY:
# Configures native compilation for ONNX Runtime C++ execution provider bindings
# and llama.cpp GGUF SLM validation engine. Generates native static library `libcontrolplane_cpp.a`
# linked into Rust binary via FFI.
#
# LATENCY & HARDWARE OPTIMIZATIONS:
# Enables compiler SIMD flags (-mavx512f, -mavx512vl, -mavx2, -mfma) for x86_64,
# OpenMP thread pooling for ONNX execution, and optional CUDA backend target generation.

cmake_minimum_required(VERSION 3.18)
project(controlplane_cpp LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization Flags for Sub-5ms Pre-Flight Latency
if (MSVC)
    add_compile_options(/O2 /arch:AVX2 /std:c++17)
else()
    add_compile_options(-O3 -march=native -mavx2 -mfma -ffast-math -pthread)
endif()

# Find ONNX Runtime C++ Headers and Libraries
find_package(ONNXRuntime QUIET)
if (NOT ONNXRuntime_FOUND)
    message(STATUS "ONNX Runtime package not found via CMake module. Using standard include/link paths.")
    set(ONNXRUNTIME_INCLUDE_DIRS "/usr/local/include/onnxruntime" CACHE PATH "ONNX Runtime includes")
    set(ONNXRUNTIME_LIBRARIES "/usr/local/lib/libonnxruntime.so" CACHE FILEPATH "ONNX Runtime library")
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

# C++ Source Files for Classifier, Validator, and FFI Boundary
set(CPP_SOURCES
    cpp/src/onnx_classifier.cpp
    cpp/src/llama_validator.cpp
    cpp/src/ffi_bridge.cpp
)

# Build Native Static Library for Rust FFI Linking
add_library(controlplane_cpp STATIC ${CPP_SOURCES})

target_include_directories(controlplane_cpp PUBLIC
    cpp/include
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

target_link_libraries(controlplane_cpp PRIVATE
    ${ONNXRUNTIME_LIBRARIES}
    Threads::Threads
)
