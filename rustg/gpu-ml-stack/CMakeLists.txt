cmake_minimum_required(VERSION 3.18)
project(gpu_ml_stack_tests CUDA CXX)

# Find CUDA
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Set CUDA architecture (adjust for your GPU)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")

# Include directories
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Test executables
add_executable(tensor_core_test tests/cuda/tensor_core_test.cu)
add_executable(autodiff_test tests/cuda/autodiff_test.cu)
add_executable(kernel_fusion_test tests/cuda/kernel_fusion_test.cu)
add_executable(nn_layers_test tests/cuda/nn_layers_test.cu)
add_executable(training_test tests/cuda/training_test.cu)

# Link libraries for each test
foreach(test_target tensor_core_test autodiff_test kernel_fusion_test nn_layers_test training_test)
    target_link_libraries(${test_target}
        CUDA::cudart
        CUDA::cublas
        CUDA::cudnn
        CUDA::curand
        CUDA::nvrtc
        CUDA::cuda_driver
    )
    
    # Set properties
    set_target_properties(${test_target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endforeach()

# Custom target to run all tests
add_custom_target(run_tests
    COMMAND tensor_core_test
    COMMAND autodiff_test
    COMMAND kernel_fusion_test
    COMMAND nn_layers_test
    COMMAND training_test
    DEPENDS tensor_core_test autodiff_test kernel_fusion_test nn_layers_test training_test
    COMMENT "Running all GPU ML Stack tests..."
)