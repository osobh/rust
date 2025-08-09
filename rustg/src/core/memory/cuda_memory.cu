#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

extern "C" {

// Initialize CUDA runtime
int cuda_initialize() {
  // Set device 0 as default
  cudaError_t err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  
  // Reset device to clear any previous state
  err = cudaDeviceReset();
  return static_cast<int>(err);
}

// Clean up CUDA runtime
int cuda_cleanup() {
  cudaError_t err = cudaDeviceReset();
  return static_cast<int>(err);
}

// Get number of CUDA devices
int cuda_get_device_count() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    return -1;
  }
  return count;
}

// Get device memory size
size_t cuda_get_device_memory_size(int device) {
  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess) {
    return 0;
  }
  
  // Reserve some memory for system use (90% of total)
  return (size_t)(prop.totalGlobalMem * 0.9);
}

// Allocate device memory
void* cuda_malloc_device(size_t size) {
  void* ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, size);
  if (err != cudaSuccess) {
    return nullptr;
  }
  
  // Initialize memory to zero
  err = cudaMemset(ptr, 0, size);
  if (err != cudaSuccess) {
    cudaFree(ptr);
    return nullptr;
  }
  
  return ptr;
}

// Free device memory
int cuda_free_device(void* ptr) {
  if (ptr == nullptr) {
    return 0;
  }
  cudaError_t err = cudaFree(ptr);
  return static_cast<int>(err);
}

// Copy memory from host to device
int cuda_memcpy_host_to_device(void* dst, const void* src, size_t size) {
  cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
  return static_cast<int>(err);
}

// Copy memory from device to host
int cuda_memcpy_device_to_host(void* dst, const void* src, size_t size) {
  cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
  return static_cast<int>(err);
}

// Synchronize device
int cuda_synchronize() {
  cudaError_t err = cudaDeviceSynchronize();
  return static_cast<int>(err);
}

// Get last error
int cuda_get_last_error() {
  cudaError_t err = cudaGetLastError();
  return static_cast<int>(err);
}

// Get error string
const char* cuda_get_error_string(int error) {
  return cudaGetErrorString(static_cast<cudaError_t>(error));
}

} // extern "C"