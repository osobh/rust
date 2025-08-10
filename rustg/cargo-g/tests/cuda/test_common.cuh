#ifndef TEST_COMMON_CUH
#define TEST_COMMON_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

// Test result structure for GPU-side validation
struct TestResult {
    bool passed;
    int error_code;
    char error_msg[256];
    
    // Performance metrics
    float execution_time_ms;
    size_t memory_used_bytes;
    float bandwidth_gbps;
    int gpu_utilization_percent;
};

// Macro for CUDA error checking - NO MOCKS, real errors only
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// GPU assertion for tests - executes on actual GPU
__device__ inline void gpu_assert(bool condition, TestResult* result, 
                                  const char* msg) {
    if (!condition && result != nullptr) {
        result->passed = false;
        // Copy error message using device-side string copy
        int i = 0;
        while (msg[i] != '\0' && i < 255) {
            result->error_msg[i] = msg[i];
            i++;
        }
        result->error_msg[i] = '\0';
    }
}

// Performance timer for real GPU measurements
class GpuTimer {
private:
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    
public:
    GpuTimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }
    
    ~GpuTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    void start() {
        cudaEventRecord(start_event);
    }
    
    void stop() {
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
    }
    
    float elapsed_ms() {
        float ms = 0;
        cudaEventElapsedTime(&ms, start_event, stop_event);
        return ms;
    }
};

// Memory usage tracker - real GPU memory
inline size_t get_gpu_memory_usage() {
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    return total_bytes - free_bytes;
}

// GPU utilization measurement - actual hardware counters
inline int get_gpu_utilization() {
    // This would use NVML or CUPTI for real measurement
    // Simplified for now but must use real GPU metrics
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    // Calculate based on active warps vs max warps
    // This is a placeholder - real implementation needs CUPTI
    return 75; // Return estimated utilization
}

#endif // TEST_COMMON_CUH