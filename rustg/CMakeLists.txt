cmake_minimum_required(VERSION 3.18)
project(rustg LANGUAGES CXX CUDA)

# Set C++ and CUDA standards (C++17 required by CUDA 13.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Check for modern compiler support (CUDA 13.0 supports GCC 15, Clang 20)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11.0")
        message(WARNING "GCC version ${CMAKE_CXX_COMPILER_VERSION} is old. Consider upgrading to GCC 13+ for CUDA 13.0")
    elseif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "15.0")
        message(STATUS "Using GCC ${CMAKE_CXX_COMPILER_VERSION} - fully supported by CUDA 13.0")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "20.0")
        message(STATUS "Using Clang ${CMAKE_CXX_COMPILER_VERSION} - fully supported by CUDA 13.0")
    endif()
endif()

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architecture (adjust based on your GPU)
# 70 = V100, 75 = RTX 2080, 80 = A100, 86 = RTX 3090, 89 = RTX 4090, 110 = RTX 5090 (Blackwell)
# Note: RTX 5090 uses sm_110 (Blackwell), not sm_89 (Ada Lovelace)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90 110)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
# Enable relocatable device code for Thrust/CUB and proper linking
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --extended-lambda -rdc=true")
# Generate both SASS and PTX for forward compatibility
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_110,code=sm_110")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_110,code=compute_110")

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
    src/lexer/kernels/char_classifier.cu
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

# Link CUDA libraries (including cudadevrt for device linking with -rdc=true)
target_link_libraries(rustg_core 
    CUDA::cudart
    CUDA::cuda_driver
)

# Link cudadevrt for RDC (if available)
if(EXISTS "${CUDAToolkit_LIBRARY_DIR}/libcudadevrt.a")
    target_link_libraries(rustg_core ${CUDAToolkit_LIBRARY_DIR}/libcudadevrt.a)
endif()

# Optional CUDA 13.0 libraries
find_library(NVRTC_LIB nvrtc PATHS ${CUDAToolkit_LIBRARY_DIR})
if(NVRTC_LIB)
    target_link_libraries(rustg_core ${NVRTC_LIB})
endif()

find_library(NVJITLINK_LIB nvJitLink PATHS ${CUDAToolkit_LIBRARY_DIR})
if(NVJITLINK_LIB)
    target_link_libraries(rustg_core ${NVJITLINK_LIB})
    message(STATUS "Found nvJitLink for CUDA 13.0 JIT compilation")
endif()

# Enable separable compilation and device symbol resolution for CUDA
set_target_properties(rustg_core PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON  # Resolves device symbols at link time
    POSITION_INDEPENDENT_CODE ON
)

# Testing executable
# Option to build tests (default OFF to avoid CUDA compatibility issues)
option(BUILD_TESTS "Build test executables" OFF)

if(BUILD_TESTS)
    enable_testing()
    message(STATUS "Building with tests enabled")
else()
    message(STATUS "Building without tests (use -DBUILD_TESTS=ON to enable)")
endif()

# Find GTest (optional, for GPU kernel tests) - only if building tests
if(BUILD_TESTS)
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

    # Phase 2 integration tests (separated to avoid multiple main() conflicts)
    add_executable(phase2_integration_test
        tests/phase2_integration_test.cu
        ${KERNEL_SOURCES}
    )
    target_link_libraries(phase2_integration_test
        rustg_core
        CUDA::cudart
    )
    set_target_properties(phase2_integration_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME phase2_integration COMMAND phase2_integration_test)

    add_executable(macro_expansion_test
        tests/macro_expansion_test.cu
        ${KERNEL_SOURCES}
    )
    target_link_libraries(macro_expansion_test
        rustg_core
        CUDA::cudart
    )
    set_target_properties(macro_expansion_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME macro_expansion COMMAND macro_expansion_test)

    # RTX 5090 validation tests (CUDA 13.0)
    add_executable(rtx5090_validation tests/rtx5090_validation.cu)
    target_link_libraries(rtx5090_validation
        rustg_core
        CUDA::cudart
        ${CUDAToolkit_LIBRARY_DIR}/libcudadevrt.a
    )
    set_target_properties(rtx5090_validation PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    add_test(NAME rtx5090_validation COMMAND rtx5090_validation)

    # Blackwell architecture feature tests
    add_executable(blackwell_features tests/blackwell_features.cu)
    target_link_libraries(blackwell_features
        rustg_core
        CUDA::cudart
        ${CUDAToolkit_LIBRARY_DIR}/libcudadevrt.a
    )
    set_target_properties(blackwell_features PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    add_test(NAME blackwell_features COMMAND blackwell_features)

endif() # BUILD_TESTS

# Installation rules
install(TARGETS rustg_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)