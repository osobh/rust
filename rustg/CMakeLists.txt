cmake_minimum_required(VERSION 3.18)
project(rustg LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architecture (adjust based on your GPU)
# 70 = V100, 75 = RTX 2080, 80 = A100, 86 = RTX 3090, 89 = RTX 4090
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# Debug flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Source files
file(GLOB_RECURSE CUDA_SOURCES 
    src/core/*.cu
    src/lexer/*.cu
    src/parser/*.cu
)

file(GLOB_RECURSE CPP_SOURCES 
    src/core/*.cpp
    src/lexer/*.cpp
    src/parser/*.cpp
)

# Explicitly add new kernels
set(KERNEL_SOURCES
    src/lexer/kernels/tokenizer_optimized.cu
    src/lexer/kernels/string_comment_handler.cu
    src/lexer/kernels/rust_syntax_advanced.cu
    src/lexer/kernels/fused_tokenizer_ast.cu
    src/parser/kernels/ast_construction.cu
    src/macro_expansion/kernels/pattern_matcher.cu
    src/macro_expansion/kernels/macro_expander.cu
    src/macro_expansion/kernels/hygiene_tracker.cu
    src/macro_expansion/kernels/macro_pipeline.cu
    src/macro_expansion/kernels/macro_rules_matcher.cu
    src/macro_expansion/kernels/repetition_expander.cu
)

# Create the main library
add_library(rustg_core STATIC ${CUDA_SOURCES} ${CPP_SOURCES} ${KERNEL_SOURCES})

# Link CUDA libraries
target_link_libraries(rustg_core 
    CUDA::cudart
    CUDA::cuda_driver
)

# Enable separable compilation for CUDA
set_target_properties(rustg_core PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Testing executable
enable_testing()

# Find GTest (optional, for GPU kernel tests)
find_package(GTest QUIET)

if(GTest_FOUND)
    # GPU kernel tests
    add_executable(gpu_kernel_tests 
        tests/gpu_kernel_test.cu
        ${CUDA_SOURCES}
    )
    
    target_link_libraries(gpu_kernel_tests
        rustg_core
        GTest::GTest
        GTest::Main
        CUDA::cudart
    )
    
    set_target_properties(gpu_kernel_tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    
    add_test(NAME gpu_kernel_tests COMMAND gpu_kernel_tests)
else()
    message(WARNING "GTest not found, GPU kernel tests will not be built")
endif()

# Performance validation executable
add_executable(perf_validation
    tests/performance_validation.cu
    ${KERNEL_SOURCES}
)

target_link_libraries(perf_validation
    rustg_core
    CUDA::cudart
)

set_target_properties(perf_validation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_test(NAME performance_validation COMMAND perf_validation)

# Phase 2 integration tests
add_executable(phase2_tests
    tests/phase2_integration_test.cu
    tests/macro_expansion_test.cu
    ${KERNEL_SOURCES}
)

target_link_libraries(phase2_tests
    rustg_core
    CUDA::cudart
)

set_target_properties(phase2_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_test(NAME phase2_integration COMMAND phase2_tests)

# Installation rules
install(TARGETS rustg_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)