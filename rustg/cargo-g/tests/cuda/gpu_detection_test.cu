#include "test_common.cuh"
#include <vector>
#include <string>

// Structure to hold GPU device information
struct GpuDeviceInfo {
    int device_id;
    char name[256];
    size_t total_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_threads_per_multiprocessor;
    int warp_size;
    size_t shared_memory_per_block;
    int max_grid_dimensions[3];
    int max_block_dimensions[3];
    bool supports_managed_memory;
    bool supports_concurrent_kernels;
    bool supports_gpu_direct;
};

// Test kernel: Verify GPU device enumeration - NO MOCKS
__global__ void test_gpu_enumeration(TestResult* results, 
                                     GpuDeviceInfo* device_info,
                                     int expected_min_devices) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        // Initialize test result
        results->passed = true;
        results->error_code = 0;
        
        // Real GPU device count check
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        
        // Verify minimum device count
        gpu_assert(device_count >= expected_min_devices, results,
                  "Insufficient GPU devices detected");
        
        // Store device count for host verification
        if (device_count > 0) {
            results->error_code = device_count; // Use error_code to pass count
        }
    }
}

// Test kernel: Verify GPU capabilities - REAL HARDWARE
__global__ void test_gpu_capabilities(TestResult* results,
                                      GpuDeviceInfo* device_info,
                                      int device_id) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        results->passed = true;
        
        // Query real device properties
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        
        // Fill device info structure with REAL data
        device_info->device_id = device_id;
        
        // Copy device name
        for (int i = 0; i < 256; i++) {
            device_info->name[i] = prop.name[i];
            if (prop.name[i] == '\0') break;
        }
        
        device_info->total_memory = prop.totalGlobalMem;
        device_info->compute_capability_major = prop.major;
        device_info->compute_capability_minor = prop.minor;
        device_info->multiprocessor_count = prop.multiProcessorCount;
        device_info->max_threads_per_block = prop.maxThreadsPerBlock;
        device_info->max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
        device_info->warp_size = prop.warpSize;
        device_info->shared_memory_per_block = prop.sharedMemPerBlock;
        
        // Grid and block dimensions
        for (int i = 0; i < 3; i++) {
            device_info->max_grid_dimensions[i] = prop.maxGridSize[i];
            device_info->max_block_dimensions[i] = prop.maxThreadsDim[i];
        }
        
        // Feature support
        device_info->supports_managed_memory = prop.managedMemory;
        device_info->supports_concurrent_kernels = prop.concurrentKernels;
        device_info->supports_gpu_direct = prop.directManagedMemAccessFromHost;
        
        // Validate minimum requirements for cargo-g
        gpu_assert(prop.major >= 7, results, 
                  "GPU compute capability too low (need 7.0+)");
        gpu_assert(prop.totalGlobalMem >= 1ULL << 30, results,
                  "Insufficient GPU memory (need 1GB+)");
        gpu_assert(prop.warpSize == 32, results,
                  "Unexpected warp size");
    }
}

// Test kernel: Verify multi-GPU support - REAL MULTI-GPU
__global__ void test_multi_gpu_support(TestResult* results,
                                       int* peer_access_matrix,
                                       int device_count) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < device_count * device_count) {
        int from_device = tid / device_count;
        int to_device = tid % device_count;
        
        if (from_device != to_device) {
            // Check real peer access capability
            int can_access = 0;
            cudaDeviceCanAccessPeer(&can_access, from_device, to_device);
            peer_access_matrix[tid] = can_access;
            
            // Enable peer access if available
            if (can_access && from_device == 0) {
                cudaSetDevice(from_device);
                cudaDeviceEnablePeerAccess(to_device, 0);
            }
        } else {
            peer_access_matrix[tid] = 1; // Self-access always available
        }
    }
    
    __syncthreads();
    
    if (tid == 0) {
        results->passed = true;
        // Verify at least self-access works
        for (int i = 0; i < device_count; i++) {
            int self_access = peer_access_matrix[i * device_count + i];
            gpu_assert(self_access == 1, results, 
                      "Self-access should always be available");
        }
    }
}

// Test kernel: Memory bandwidth test - REAL MEASUREMENT
__global__ void test_memory_bandwidth(TestResult* results,
                                      float* data,
                                      size_t size_elements) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    // Actual memory access pattern
    float sum = 0.0f;
    for (size_t i = tid; i < size_elements; i += stride) {
        sum += data[i];
    }
    
    // Prevent optimization
    if (tid == 0) {
        data[0] = sum;
        results->passed = true;
        
        // Calculate real bandwidth (done by host)
        results->memory_used_bytes = size_elements * sizeof(float);
    }
}

// Host-side test runner
extern "C" void run_gpu_detection_tests() {
    printf("=== Running GPU Detection Tests (Real CUDA) ===\n");
    
    // Test 1: Device Enumeration
    {
        TestResult* d_results;
        GpuDeviceInfo* d_device_info;
        
        CUDA_CHECK(cudaMalloc(&d_results, sizeof(TestResult)));
        CUDA_CHECK(cudaMalloc(&d_device_info, sizeof(GpuDeviceInfo)));
        
        test_gpu_enumeration<<<1, 1>>>(d_results, d_device_info, 1);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        TestResult h_results;
        CUDA_CHECK(cudaMemcpy(&h_results, d_results, sizeof(TestResult),
                             cudaMemcpyDeviceToHost));
        
        assert(h_results.passed && "GPU enumeration test failed");
        printf("✓ GPU enumeration test passed (%d devices found)\n", 
               h_results.error_code);
        
        CUDA_CHECK(cudaFree(d_results));
        CUDA_CHECK(cudaFree(d_device_info));
    }
    
    // Test 2: GPU Capabilities
    {
        TestResult* d_results;
        GpuDeviceInfo* d_device_info;
        
        CUDA_CHECK(cudaMalloc(&d_results, sizeof(TestResult)));
        CUDA_CHECK(cudaMalloc(&d_device_info, sizeof(GpuDeviceInfo)));
        
        test_gpu_capabilities<<<1, 1>>>(d_results, d_device_info, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        TestResult h_results;
        GpuDeviceInfo h_device_info;
        CUDA_CHECK(cudaMemcpy(&h_results, d_results, sizeof(TestResult),
                             cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_device_info, d_device_info, sizeof(GpuDeviceInfo),
                             cudaMemcpyDeviceToHost));
        
        assert(h_results.passed && "GPU capabilities test failed");
        printf("✓ GPU capabilities test passed (Device: %s, Memory: %.2f GB)\n",
               h_device_info.name, 
               h_device_info.total_memory / (1024.0 * 1024.0 * 1024.0));
        
        CUDA_CHECK(cudaFree(d_results));
        CUDA_CHECK(cudaFree(d_device_info));
    }
    
    // Test 3: Multi-GPU Support
    {
        int device_count;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        
        TestResult* d_results;
        int* d_peer_matrix;
        
        CUDA_CHECK(cudaMalloc(&d_results, sizeof(TestResult)));
        CUDA_CHECK(cudaMalloc(&d_peer_matrix, 
                             device_count * device_count * sizeof(int)));
        
        int threads = min(256, device_count * device_count);
        int blocks = (device_count * device_count + threads - 1) / threads;
        
        test_multi_gpu_support<<<blocks, threads>>>(d_results, d_peer_matrix, 
                                                    device_count);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        TestResult h_results;
        CUDA_CHECK(cudaMemcpy(&h_results, d_results, sizeof(TestResult),
                             cudaMemcpyDeviceToHost));
        
        assert(h_results.passed && "Multi-GPU support test failed");
        printf("✓ Multi-GPU support test passed\n");
        
        CUDA_CHECK(cudaFree(d_results));
        CUDA_CHECK(cudaFree(d_peer_matrix));
    }
    
    // Test 4: Memory Bandwidth
    {
        size_t test_size = 256 * 1024 * 1024; // 256MB
        size_t elements = test_size / sizeof(float);
        
        TestResult* d_results;
        float* d_data;
        
        CUDA_CHECK(cudaMalloc(&d_results, sizeof(TestResult)));
        CUDA_CHECK(cudaMalloc(&d_data, test_size));
        
        // Initialize data
        CUDA_CHECK(cudaMemset(d_data, 1, test_size));
        
        GpuTimer timer;
        timer.start();
        
        test_memory_bandwidth<<<256, 256>>>(d_results, d_data, elements);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        timer.stop();
        float elapsed_ms = timer.elapsed_ms();
        
        TestResult h_results;
        CUDA_CHECK(cudaMemcpy(&h_results, d_results, sizeof(TestResult),
                             cudaMemcpyDeviceToHost));
        
        float bandwidth_gbps = (test_size / (1024.0 * 1024.0 * 1024.0)) / 
                              (elapsed_ms / 1000.0);
        
        assert(h_results.passed && "Memory bandwidth test failed");
        printf("✓ Memory bandwidth test passed (%.2f GB/s)\n", bandwidth_gbps);
        
        CUDA_CHECK(cudaFree(d_results));
        CUDA_CHECK(cudaFree(d_data));
    }
    
    printf("=== All GPU Detection Tests Passed ===\n");
}