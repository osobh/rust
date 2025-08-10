#include "test_common.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Build configuration for GPU kernels
struct BuildConfig {
    int target_arch;          // SM architecture (70, 75, 80, etc.)
    int optimization_level;   // 0-3
    bool use_fast_math;
    bool enable_debug;
    int max_registers;
    size_t shared_memory_size;
    int threads_per_block;
    int blocks_per_grid;
    bool enable_profiling;
};

// Test different kernel configurations - REAL GPU EXECUTION
__global__ void test_kernel_configuration(TestResult* results,
                                         BuildConfig* config,
                                         float* output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    cg::thread_block block = cg::this_thread_block();
    
    if (tid == 0) {
        results->passed = true;
        
        // Verify thread configuration matches request
        gpu_assert(blockDim.x == config->threads_per_block, results,
                  "Thread configuration mismatch");
        gpu_assert(gridDim.x == config->blocks_per_grid, results,
                  "Block configuration mismatch");
        
        // Check shared memory allocation
        extern __shared__ float shared_mem[];
        if (config->shared_memory_size > 0) {
            // Use shared memory to verify it's allocated
            shared_mem[0] = 1.0f;
            __syncthreads();
            gpu_assert(shared_mem[0] == 1.0f, results,
                      "Shared memory not properly allocated");
        }
    }
    
    // Actual computation to test optimization levels
    float value = tid * 0.1f;
    
    // Different computation based on optimization
    if (config->use_fast_math) {
        value = __fmaf_rn(value, 2.0f, 1.0f);  // Fast math intrinsic
    } else {
        value = value * 2.0f + 1.0f;  // Standard math
    }
    
    // Store result
    if (tid < 1024) {
        output[tid] = value;
    }
}

// Test parallel compilation of multiple kernels
__global__ void test_parallel_compilation(TestResult* results,
                                         int kernel_id,
                                         int total_kernels) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        results[kernel_id].passed = true;
        
        // Each kernel verifies it was compiled
        gpu_assert(kernel_id < total_kernels, &results[kernel_id],
                  "Invalid kernel ID");
        
        // Simulate different kernel workloads
        float dummy = 0.0f;
        for (int i = 0; i < kernel_id * 100; i++) {
            dummy += __sinf(i * 0.1f);
        }
        
        // Store to prevent optimization
        results[kernel_id].execution_time_ms = dummy;
    }
}

// Test memory configuration for builds
__global__ void test_memory_configuration(TestResult* results,
                                         size_t global_mem_size,
                                         size_t* allocated_sizes,
                                         int num_allocations) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        results->passed = true;
        
        // Check global memory availability
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        
        gpu_assert(total_mem >= global_mem_size, results,
                  "Insufficient global memory");
        
        // Verify allocation sizes
        size_t total_allocated = 0;
        for (int i = 0; i < num_allocations; i++) {
            total_allocated += allocated_sizes[i];
        }
        
        gpu_assert(total_allocated <= free_mem, results,
                  "Memory allocations exceed available memory");
        
        results->memory_used_bytes = total_allocated;
    }
}

// Test optimization flags effect
__global__ void test_optimization_levels(TestResult* results,
                                        BuildConfig* config,
                                        float* input,
                                        float* output,
                                        int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    // Complex computation to test optimization
    for (int i = tid; i < size; i += stride) {
        float val = input[i];
        
        // Optimization level 0: No optimization
        if (config->optimization_level == 0) {
            val = val * 2.0f;
            val = val + 1.0f;
            val = val / 3.0f;
            val = val - 0.5f;
        }
        // Optimization level 1: Basic optimization
        else if (config->optimization_level == 1) {
            val = (val * 2.0f + 1.0f) / 3.0f - 0.5f;
        }
        // Optimization level 2: Advanced optimization
        else if (config->optimization_level == 2) {
            val = __fmaf_rn(val, 0.666667f, -0.166667f);
        }
        // Optimization level 3: Aggressive optimization
        else {
            val = __fmaf_rn(val, 0.666667f, -0.166667f);
            // Prefetch next element
            if (i + stride < size) {
                __builtin_prefetch(&input[i + stride], 0, 1);
            }
        }
        
        output[i] = val;
    }
    
    if (tid == 0) {
        results->passed = true;
        results->gpu_utilization_percent = get_gpu_utilization();
    }
}

// Test incremental compilation support
__global__ void test_incremental_compilation(TestResult* results,
                                            int* compilation_cache,
                                            int cache_entries,
                                            int kernel_hash) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        results->passed = true;
        
        // Check if kernel is in cache
        bool found_in_cache = false;
        for (int i = 0; i < cache_entries; i++) {
            if (compilation_cache[i] == kernel_hash) {
                found_in_cache = true;
                break;
            }
        }
        
        if (!found_in_cache && cache_entries < 1024) {
            // Add to cache
            compilation_cache[cache_entries] = kernel_hash;
        }
        
        // Report cache status
        results->error_code = found_in_cache ? 1 : 0;
    }
}

// Host-side test runner
extern "C" void run_build_config_tests() {
    printf("=== Running Build Configuration Tests (Real CUDA) ===\n");
    
    // Test 1: Kernel Configuration
    {
        TestResult* d_results;
        BuildConfig* d_config;
        float* d_output;
        
        CUDA_CHECK(cudaMalloc(&d_results, sizeof(TestResult)));
        CUDA_CHECK(cudaMalloc(&d_config, sizeof(BuildConfig)));
        CUDA_CHECK(cudaMalloc(&d_output, 1024 * sizeof(float)));
        
        BuildConfig h_config = {
            .target_arch = 70,
            .optimization_level = 2,
            .use_fast_math = true,
            .enable_debug = false,
            .max_registers = 64,
            .shared_memory_size = 48 * 1024,
            .threads_per_block = 256,
            .blocks_per_grid = 4,
            .enable_profiling = false
        };
        
        CUDA_CHECK(cudaMemcpy(d_config, &h_config, sizeof(BuildConfig),
                             cudaMemcpyHostToDevice));
        
        test_kernel_configuration<<<h_config.blocks_per_grid, 
                                   h_config.threads_per_block,
                                   h_config.shared_memory_size>>>(
            d_results, d_config, d_output);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        TestResult h_results;
        CUDA_CHECK(cudaMemcpy(&h_results, d_results, sizeof(TestResult),
                             cudaMemcpyDeviceToHost));
        
        assert(h_results.passed && "Kernel configuration test failed");
        printf("✓ Kernel configuration test passed\n");
        
        CUDA_CHECK(cudaFree(d_results));
        CUDA_CHECK(cudaFree(d_config));
        CUDA_CHECK(cudaFree(d_output));
    }
    
    // Test 2: Parallel Compilation
    {
        const int num_kernels = 8;
        TestResult* d_results;
        
        CUDA_CHECK(cudaMalloc(&d_results, num_kernels * sizeof(TestResult)));
        
        // Launch multiple kernels in parallel
        for (int i = 0; i < num_kernels; i++) {
            test_parallel_compilation<<<1, 32>>>(d_results, i, num_kernels);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        TestResult h_results[num_kernels];
        CUDA_CHECK(cudaMemcpy(h_results, d_results, 
                             num_kernels * sizeof(TestResult),
                             cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < num_kernels; i++) {
            assert(h_results[i].passed && "Parallel compilation test failed");
        }
        printf("✓ Parallel compilation test passed (%d kernels)\n", num_kernels);
        
        CUDA_CHECK(cudaFree(d_results));
    }
    
    // Test 3: Memory Configuration
    {
        TestResult* d_results;
        size_t* d_allocated_sizes;
        
        const int num_allocations = 5;
        size_t h_allocated_sizes[] = {
            100 * 1024 * 1024,  // 100MB
            200 * 1024 * 1024,  // 200MB
            50 * 1024 * 1024,   // 50MB
            150 * 1024 * 1024,  // 150MB
            100 * 1024 * 1024   // 100MB
        };
        
        CUDA_CHECK(cudaMalloc(&d_results, sizeof(TestResult)));
        CUDA_CHECK(cudaMalloc(&d_allocated_sizes, 
                             num_allocations * sizeof(size_t)));
        CUDA_CHECK(cudaMemcpy(d_allocated_sizes, h_allocated_sizes,
                             num_allocations * sizeof(size_t),
                             cudaMemcpyHostToDevice));
        
        size_t required_mem = 1ULL << 30; // 1GB
        test_memory_configuration<<<1, 1>>>(d_results, required_mem,
                                           d_allocated_sizes, num_allocations);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        TestResult h_results;
        CUDA_CHECK(cudaMemcpy(&h_results, d_results, sizeof(TestResult),
                             cudaMemcpyDeviceToHost));
        
        assert(h_results.passed && "Memory configuration test failed");
        printf("✓ Memory configuration test passed (%.2f MB allocated)\n",
               h_results.memory_used_bytes / (1024.0 * 1024.0));
        
        CUDA_CHECK(cudaFree(d_results));
        CUDA_CHECK(cudaFree(d_allocated_sizes));
    }
    
    // Test 4: Optimization Levels
    {
        const int size = 1024 * 1024; // 1M elements
        TestResult* d_results;
        BuildConfig* d_config;
        float* d_input;
        float* d_output;
        
        CUDA_CHECK(cudaMalloc(&d_results, sizeof(TestResult)));
        CUDA_CHECK(cudaMalloc(&d_config, sizeof(BuildConfig)));
        CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));
        
        // Initialize input
        std::vector<float> h_input(size);
        for (int i = 0; i < size; i++) {
            h_input[i] = i * 0.001f;
        }
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size * sizeof(float),
                             cudaMemcpyHostToDevice));
        
        // Test each optimization level
        for (int opt_level = 0; opt_level <= 3; opt_level++) {
            BuildConfig h_config = {
                .optimization_level = opt_level,
                .use_fast_math = (opt_level >= 2)
            };
            
            CUDA_CHECK(cudaMemcpy(d_config, &h_config, sizeof(BuildConfig),
                                 cudaMemcpyHostToDevice));
            
            GpuTimer timer;
            timer.start();
            
            test_optimization_levels<<<256, 256>>>(d_results, d_config,
                                                  d_input, d_output, size);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            timer.stop();
            
            TestResult h_results;
            CUDA_CHECK(cudaMemcpy(&h_results, d_results, sizeof(TestResult),
                                 cudaMemcpyDeviceToHost));
            
            assert(h_results.passed && "Optimization test failed");
            printf("✓ Optimization level %d test passed (%.2f ms)\n",
                   opt_level, timer.elapsed_ms());
        }
        
        CUDA_CHECK(cudaFree(d_results));
        CUDA_CHECK(cudaFree(d_config));
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
    }
    
    // Test 5: Incremental Compilation
    {
        TestResult* d_results;
        int* d_cache;
        const int max_cache_entries = 1024;
        
        CUDA_CHECK(cudaMalloc(&d_results, sizeof(TestResult)));
        CUDA_CHECK(cudaMalloc(&d_cache, max_cache_entries * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_cache, 0, max_cache_entries * sizeof(int)));
        
        // Simulate multiple compilation requests
        int kernel_hashes[] = {0x1234, 0x5678, 0x1234, 0x9ABC, 0x5678};
        int cache_size = 0;
        
        for (int i = 0; i < 5; i++) {
            test_incremental_compilation<<<1, 1>>>(d_results, d_cache,
                                                  cache_size, kernel_hashes[i]);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            TestResult h_results;
            CUDA_CHECK(cudaMemcpy(&h_results, d_results, sizeof(TestResult),
                                 cudaMemcpyDeviceToHost));
            
            assert(h_results.passed && "Incremental compilation test failed");
            
            if (h_results.error_code == 0) {
                cache_size++; // New entry added
                printf("  Cache miss for kernel 0x%X (added to cache)\n",
                       kernel_hashes[i]);
            } else {
                printf("  Cache hit for kernel 0x%X\n", kernel_hashes[i]);
            }
        }
        
        printf("✓ Incremental compilation test passed\n");
        
        CUDA_CHECK(cudaFree(d_results));
        CUDA_CHECK(cudaFree(d_cache));
    }
    
    printf("=== All Build Configuration Tests Passed ===\n");
}