#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <climits>

// Global memory pool handle for async allocations (CUDA 13.0)
static cudaMemPool_t global_mem_pool = nullptr;
static cudaStream_t default_stream = nullptr;

extern "C" {

// Initialize CUDA runtime with memory pools (CUDA 13.0)
int cuda_initialize() {
  // Set device 0 as default
  cudaError_t err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  
  // Create default stream for async operations
  err = cudaStreamCreate(&default_stream);
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  
  // Get the default memory pool for the current device (CUDA 13.0)
  int device;
  cudaGetDevice(&device);
  err = cudaDeviceGetDefaultMemPool(&global_mem_pool, device);
  if (err != cudaSuccess) {
    // Fallback: create a custom memory pool if default not available
    cudaMemPoolProps pool_props = {};
    pool_props.allocType = cudaMemAllocationTypePinned;
    pool_props.handleTypes = cudaMemHandleTypeNone;
    pool_props.location.type = cudaMemLocationTypeDevice;
    pool_props.location.id = device;
    
    err = cudaMemPoolCreate(&global_mem_pool, &pool_props);
    if (err != cudaSuccess) {
      return static_cast<int>(err);
    }
  }
  
  // Set memory pool attributes for better performance
  uint64_t threshold = UINT64_MAX;  // No release threshold
  cudaMemPoolSetAttribute(global_mem_pool, cudaMemPoolAttrReleaseThreshold, &threshold);
  
  return 0;
}

// Clean up CUDA runtime and memory pools
int cuda_cleanup() {
  cudaError_t err = cudaSuccess;
  
  // Destroy custom memory pool if created
  if (global_mem_pool != nullptr) {
    err = cudaMemPoolDestroy(global_mem_pool);
    global_mem_pool = nullptr;
  }
  
  // Destroy stream
  if (default_stream != nullptr) {
    err = cudaStreamDestroy(default_stream);
    default_stream = nullptr;
  }
  
  err = cudaDeviceReset();
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

// Allocate device memory (legacy for compatibility)
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

// Allocate device memory asynchronously (CUDA 13.0 - preferred)
void* cuda_malloc_async(size_t size, cudaStream_t stream) {
  void* ptr = nullptr;
  cudaStream_t str = (stream != nullptr) ? stream : default_stream;
  
  // Use async allocation from memory pool
  cudaError_t err = cudaMallocAsync(&ptr, size, str);
  if (err != cudaSuccess) {
    return nullptr;
  }
  
  // Initialize memory to zero asynchronously
  err = cudaMemsetAsync(ptr, 0, size, str);
  if (err != cudaSuccess) {
    cudaFreeAsync(ptr, str);
    return nullptr;
  }
  
  return ptr;
}

// Allocate host-pinned memory with async API (CUDA 13.0)
void* cuda_malloc_host_async(size_t size) {
  void* ptr = nullptr;
  
  // Use CU_MEM_LOCATION_TYPE_HOST for host-pinned allocation
  cudaError_t err = cudaMallocHost(&ptr, size);
  if (err != cudaSuccess) {
    return nullptr;
  }
  
  // Initialize to zero
  memset(ptr, 0, size);
  return ptr;
}

// Free device memory (legacy for compatibility)
int cuda_free_device(void* ptr) {
  if (ptr == nullptr) {
    return 0;
  }
  cudaError_t err = cudaFree(ptr);
  return static_cast<int>(err);
}

// Free device memory asynchronously (CUDA 13.0 - preferred)
int cuda_free_async(void* ptr, cudaStream_t stream) {
  if (ptr == nullptr) {
    return 0;
  }
  cudaStream_t str = (stream != nullptr) ? stream : default_stream;
  cudaError_t err = cudaFreeAsync(ptr, str);
  return static_cast<int>(err);
}

// Free host-pinned memory
int cuda_free_host(void* ptr) {
  if (ptr == nullptr) {
    return 0;
  }
  cudaError_t err = cudaFreeHost(ptr);
  return static_cast<int>(err);
}

// Copy memory from host to device (synchronous)
int cuda_memcpy_host_to_device(void* dst, const void* src, size_t size) {
  cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
  return static_cast<int>(err);
}

// Copy memory from host to device asynchronously (CUDA 13.0 - preferred)
int cuda_memcpy_host_to_device_async(void* dst, const void* src, size_t size, cudaStream_t stream) {
  cudaStream_t str = (stream != nullptr) ? stream : default_stream;
  cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, str);
  return static_cast<int>(err);
}

// Copy memory from device to host (synchronous)
int cuda_memcpy_device_to_host(void* dst, const void* src, size_t size) {
  cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
  return static_cast<int>(err);
}

// Copy memory from device to host asynchronously (CUDA 13.0 - preferred)
int cuda_memcpy_device_to_host_async(void* dst, const void* src, size_t size, cudaStream_t stream) {
  cudaStream_t str = (stream != nullptr) ? stream : default_stream;
  cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, str);
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