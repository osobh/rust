cmake_minimum_required(VERSION 3.18)
project(gpu-debug-profiler LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA package
find_package(CUDA REQUIRED)

# Set CUDA architecture (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# Include directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(include)

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --ptxas-options=-v")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")

# Test executables
add_executable(source_mapping_test tests/cuda/source_mapping_test.cu)
add_executable(timeline_tracing_test tests/cuda/timeline_tracing_test.cu)  
add_executable(profiling_test tests/cuda/profiling_test.cu)
add_executable(warp_debug_test tests/cuda/warp_debug_test.cu)

# Link CUDA libraries
target_link_libraries(source_mapping_test ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})
target_link_libraries(timeline_tracing_test ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})
target_link_libraries(profiling_test ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})
target_link_libraries(warp_debug_test ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})

# Custom target to run all tests
add_custom_target(run_tests
    COMMAND source_mapping_test
    COMMAND timeline_tracing_test
    COMMAND profiling_test
    COMMAND warp_debug_test
    DEPENDS source_mapping_test timeline_tracing_test profiling_test warp_debug_test
    COMMENT "Running all GPU debug profiler tests"
)

# Install targets (optional)
install(TARGETS source_mapping_test timeline_tracing_test profiling_test warp_debug_test
        RUNTIME DESTINATION bin)