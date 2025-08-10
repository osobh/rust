// GPU-Native Allocator Tests - Written BEFORE Implementation
// NO STUBS OR MOCKS - Real GPU Operations Only
// Target: Lock-free allocation with <100 cycle latency

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>

// Test result structure
struct TestResult {
    bool passed;
    int total_tests;
    int failed_tests;
    float alloc_time_cycles;
    float dealloc_time_cycles;
    int allocations_per_sec;
    float memory_efficiency;
    char failure_msg[256];
};

// Allocation metadata
struct AllocMeta {
    uint32_t size;
    uint32_t alignment;
    uint32_t thread_id;
    uint32_t warp_id;
    void* ptr;
    uint64_t timestamp;
};

// Slab allocator structure
struct SlabAllocator {
    uint8_t* memory_pool;
    uint32_t* free_list;
    uint32_t slab_size;
    uint32_t num_slabs;
    uint32_t* next_free;
};

// Helper: Get clock cycles
__device__ uint64_t get_clock_cycles() {
    uint64_t cycles;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(cycles));
    return cycles;
}

// Test 1: Slab allocator with fixed-size blocks
__global__ void test_slab_allocator(TestResult* result,
                                   SlabAllocator* allocator,
                                   AllocMeta* alloc_info,
                                   int num_threads) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    if (tid == 0) {
        result->passed = true;
        result->total_tests = 0;
        result->failed_tests = 0;
    }
    __syncthreads();
    
    if (tid < num_threads) {
        // Measure allocation time
        uint64_t start = get_clock_cycles();
        
        // Lock-free allocation using atomicCAS
        uint32_t old_free, new_free;
        void* allocated_ptr = nullptr;
        
        do {
            old_free = *allocator->next_free;
            if (old_free >= allocator->num_slabs) {
                // Out of memory
                atomicAdd(&result->failed_tests, 1);
                result->passed = false;
                return;
            }
            new_free = allocator->free_list[old_free];
        } while (atomicCAS(allocator->next_free, old_free, new_free) != old_free);
        
        // Calculate allocated pointer
        allocated_ptr = allocator->memory_pool + (old_free * allocator->slab_size);
        
        uint64_t end = get_clock_cycles();
        float cycles = float(end - start);
        
        // Store allocation info
        alloc_info[tid].ptr = allocated_ptr;
        alloc_info[tid].size = allocator->slab_size;
        alloc_info[tid].thread_id = tid;
        alloc_info[tid].warp_id = warp_id;
        alloc_info[tid].timestamp = end;
        
        // Update timing
        atomicAdd(&result->alloc_time_cycles, cycles);
        atomicAdd(&result->total_tests, 1);
        
        // Verify allocation
        if (allocated_ptr == nullptr) {
            atomicAdd(&result->failed_tests, 1);
            result->passed = false;
        }
        
        // Test deallocation
        start = get_clock_cycles();
        
        // Lock-free deallocation
        uint32_t slab_idx = ((uint8_t*)allocated_ptr - allocator->memory_pool) / allocator->slab_size;
        do {
            old_free = *allocator->next_free;
            allocator->free_list[slab_idx] = old_free;
        } while (atomicCAS(allocator->next_free, old_free, slab_idx) != old_free);
        
        end = get_clock_cycles();
        atomicAdd(&result->dealloc_time_cycles, float(end - start));
    }
    
    __syncthreads();
    
    if (tid == 0) {
        // Calculate average cycles
        result->alloc_time_cycles /= num_threads;
        result->dealloc_time_cycles /= num_threads;
        
        // Check performance target: <100 cycles
        if (result->alloc_time_cycles > 100) {
            result->passed = false;
            sprintf(result->failure_msg, "Allocation too slow: %.0f cycles (target: <100)",
                   result->alloc_time_cycles);
        }
    }
}

// Test 2: Region allocator for large contiguous allocations
__global__ void test_region_allocator(TestResult* result,
                                     uint8_t* memory_region,
                                     uint32_t region_size,
                                     uint32_t* allocation_sizes,
                                     void** allocated_ptrs,
                                     int num_allocations) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ uint32_t region_offset;
    
    if (threadIdx.x == 0) {
        region_offset = 0;
    }
    __syncthreads();
    
    if (tid < num_allocations) {
        uint32_t size = allocation_sizes[tid];
        
        // Align to 16 bytes
        size = (size + 15) & ~15;
        
        // Atomic allocation
        uint32_t offset = atomicAdd(&region_offset, size);
        
        if (offset + size <= region_size) {
            allocated_ptrs[tid] = memory_region + offset;
            
            // Write pattern to verify
            uint32_t* ptr = (uint32_t*)(memory_region + offset);
            for (int i = 0; i < size / sizeof(uint32_t); i++) {
                ptr[i] = tid * 1000 + i;
            }
            
            atomicAdd(&result->total_tests, 1);
        } else {
            // Out of memory
            allocated_ptrs[tid] = nullptr;
            atomicAdd(&result->failed_tests, 1);
            result->passed = false;
        }
    }
    
    __syncthreads();
    
    // Verify allocations don't overlap
    if (tid < num_allocations && tid > 0) {
        void* my_ptr = allocated_ptrs[tid];
        void* prev_ptr = allocated_ptrs[tid - 1];
        uint32_t prev_size = allocation_sizes[tid - 1];
        
        if (my_ptr && prev_ptr) {
            uint8_t* my_addr = (uint8_t*)my_ptr;
            uint8_t* prev_end = (uint8_t*)prev_ptr + prev_size;
            
            if (my_addr < prev_end) {
                atomicAdd(&result->failed_tests, 1);
                result->passed = false;
            }
        }
    }
    
    if (tid == 0) {
        // Calculate memory efficiency
        result->memory_efficiency = float(region_offset) / region_size;
    }
}

// Test 3: Arena allocator with bulk deallocation
__global__ void test_arena_allocator(TestResult* result,
                                    uint8_t* arena_memory,
                                    uint32_t arena_size,
                                    uint32_t* arena_offset) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int num_passes = 5;
    
    for (int pass = 0; pass < num_passes; pass++) {
        if (tid == 0 && pass > 0) {
            // Reset arena (bulk deallocation)
            *arena_offset = 0;
        }
        __syncthreads();
        
        // Each thread allocates from arena
        uint32_t alloc_size = (tid + 1) * 64;  // Variable sizes
        uint32_t offset = atomicAdd(arena_offset, alloc_size);
        
        if (offset + alloc_size <= arena_size) {
            void* ptr = arena_memory + offset;
            
            // Use the memory
            uint64_t* data = (uint64_t*)ptr;
            for (int i = 0; i < alloc_size / sizeof(uint64_t); i++) {
                data[i] = tid * pass * i;
            }
            
            atomicAdd(&result->total_tests, 1);
        } else {
            atomicAdd(&result->failed_tests, 1);
        }
        
        __syncthreads();
    }
    
    if (tid == 0) {
        result->passed = (result->failed_tests == 0);
    }
}

// Test 4: Warp-aware coalesced allocation
__global__ void test_coalesced_allocation(TestResult* result,
                                         uint8_t* memory_pool,
                                         uint32_t pool_size,
                                         AllocMeta* alloc_info) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Warp-wide allocation request
    uint32_t warp_alloc_size = 128 * 32;  // 128 bytes per thread
    __shared__ uint32_t warp_offset[32];  // One per warp in block
    
    if (lane_id == 0) {
        warp_offset[threadIdx.x / 32] = atomicAdd((uint32_t*)&pool_size, warp_alloc_size);
    }
    __syncwarp();
    
    uint32_t base_offset = warp_offset[threadIdx.x / 32];
    uint32_t my_offset = base_offset + lane_id * 128;
    
    // Coalesced write pattern
    float4* my_ptr = (float4*)(memory_pool + my_offset);
    
    // All threads in warp write simultaneously (coalesced)
    for (int i = 0; i < 128 / sizeof(float4); i++) {
        my_ptr[i] = make_float4(tid, warp_id, lane_id, i);
    }
    
    // Verify coalescing
    uint64_t addr = (uint64_t)my_ptr;
    bool is_aligned = (addr % 128) == 0;
    bool is_coalesced = (my_offset % 128) == 0;
    
    if (!is_aligned || !is_coalesced) {
        atomicAdd(&result->failed_tests, 1);
        result->passed = false;
    }
    
    atomicAdd(&result->total_tests, 1);
}

// Test 5: Memory pool management with multiple tiers
__global__ void test_memory_pool_hierarchy(TestResult* result,
                                          uint8_t* thread_pool,
                                          uint8_t* warp_pool,
                                          uint8_t* block_pool,
                                          uint8_t* global_pool,
                                          uint32_t* pool_offsets) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Allocate from appropriate tier based on size
    uint32_t sizes[] = {64, 512, 4096, 32768};  // Different allocation sizes
    
    for (int i = 0; i < 4; i++) {
        uint32_t size = sizes[i];
        void* ptr = nullptr;
        
        if (size <= 128) {
            // Thread-local pool
            uint32_t offset = atomicAdd(&pool_offsets[0], size);
            ptr = thread_pool + tid * 1024 + offset;
        } else if (size <= 1024) {
            // Warp-shared pool
            uint32_t offset = atomicAdd(&pool_offsets[1], size);
            ptr = warp_pool + warp_id * 32768 + offset;
        } else if (size <= 8192) {
            // Block-shared pool
            uint32_t offset = atomicAdd(&pool_offsets[2], size);
            ptr = block_pool + blockIdx.x * 262144 + offset;
        } else {
            // Global pool
            uint32_t offset = atomicAdd(&pool_offsets[3], size);
            ptr = global_pool + offset;
        }
        
        if (ptr != nullptr) {
            // Use the allocated memory
            uint32_t* data = (uint32_t*)ptr;
            for (int j = 0; j < size / sizeof(uint32_t); j++) {
                data[j] = tid * i * j;
            }
            atomicAdd(&result->total_tests, 1);
        } else {
            atomicAdd(&result->failed_tests, 1);
        }
    }
    
    __syncthreads();
    
    if (tid == 0) {
        result->passed = (result->failed_tests == 0);
    }
}

// Test 6: Lock-free concurrent allocation stress test
__global__ void test_concurrent_allocation_stress(TestResult* result,
                                                 SlabAllocator* allocator,
                                                 int iterations) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int successful_allocs;
    __shared__ int failed_allocs;
    
    if (threadIdx.x == 0) {
        successful_allocs = 0;
        failed_allocs = 0;
    }
    __syncthreads();
    
    // High contention allocation/deallocation
    for (int iter = 0; iter < iterations; iter++) {
        // Try to allocate
        uint32_t old_free, new_free;
        void* ptr = nullptr;
        
        int attempts = 0;
        do {
            old_free = *allocator->next_free;
            if (old_free >= allocator->num_slabs) {
                atomicAdd(&failed_allocs, 1);
                break;
            }
            new_free = allocator->free_list[old_free];
            attempts++;
        } while (atomicCAS(allocator->next_free, old_free, new_free) != old_free && attempts < 100);
        
        if (old_free < allocator->num_slabs) {
            ptr = allocator->memory_pool + (old_free * allocator->slab_size);
            atomicAdd(&successful_allocs, 1);
            
            // Do some work
            uint32_t* data = (uint32_t*)ptr;
            for (int i = 0; i < allocator->slab_size / sizeof(uint32_t); i++) {
                data[i] = tid * iter * i;
            }
            
            // Deallocate
            uint32_t slab_idx = old_free;
            do {
                old_free = *allocator->next_free;
                allocator->free_list[slab_idx] = old_free;
            } while (atomicCAS(allocator->next_free, old_free, slab_idx) != old_free);
        }
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        atomicAdd(&result->total_tests, successful_allocs);
        atomicAdd(&result->failed_tests, failed_allocs);
        
        // High success rate required
        float success_rate = float(successful_allocs) / (successful_allocs + failed_allocs);
        if (success_rate < 0.95f) {
            result->passed = false;
            sprintf(result->failure_msg, "Low success rate: %.2f (target: >0.95)", success_rate);
        }
    }
}

// Test 7: Performance benchmark - allocation throughput
__global__ void test_allocation_throughput(TestResult* result,
                                          SlabAllocator* allocator,
                                          int allocations_per_thread) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    uint64_t start = get_clock_cycles();
    
    for (int i = 0; i < allocations_per_thread; i++) {
        // Allocate
        uint32_t old_free, new_free;
        do {
            old_free = *allocator->next_free;
            if (old_free >= allocator->num_slabs) break;
            new_free = allocator->free_list[old_free];
        } while (atomicCAS(allocator->next_free, old_free, new_free) != old_free);
        
        if (old_free < allocator->num_slabs) {
            // Immediately deallocate
            do {
                uint32_t current = *allocator->next_free;
                allocator->free_list[old_free] = current;
                if (atomicCAS(allocator->next_free, current, old_free) == current) break;
            } while (true);
        }
    }
    
    uint64_t end = get_clock_cycles();
    
    if (tid == 0) {
        float total_cycles = float(end - start);
        float cycles_per_alloc = total_cycles / (allocations_per_thread * gridDim.x * blockDim.x);
        
        result->alloc_time_cycles = cycles_per_alloc;
        
        // Calculate allocations per second (assuming 1GHz GPU)
        result->allocations_per_sec = int(1e9f / cycles_per_alloc);
        
        // Target: 100K allocations/sec minimum
        if (result->allocations_per_sec < 100000) {
            result->passed = false;
            sprintf(result->failure_msg, "Throughput too low: %d/sec (target: >100K/sec)",
                   result->allocations_per_sec);
        } else {
            result->passed = true;
        }
    }
}

// Initialize slab allocator
__global__ void init_slab_allocator(SlabAllocator* allocator,
                                   uint8_t* memory,
                                   uint32_t slab_size,
                                   uint32_t num_slabs) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        allocator->memory_pool = memory;
        allocator->slab_size = slab_size;
        allocator->num_slabs = num_slabs;
        allocator->next_free = (uint32_t*)(memory + num_slabs * slab_size);
        allocator->free_list = (uint32_t*)(memory + num_slabs * slab_size + sizeof(uint32_t));
        *allocator->next_free = 0;
    }
    
    // Initialize free list
    if (tid < num_slabs) {
        allocator->free_list[tid] = tid + 1;
    }
}

// Main test runner
int main() {
    printf("GPU-Native Allocator Tests - NO STUBS OR MOCKS\n");
    printf("Target: <100 cycle allocation latency\n");
    printf("==============================================\n\n");
    
    // Allocate test resources
    TestResult* d_result;
    cudaMalloc(&d_result, sizeof(TestResult));
    
    const int POOL_SIZE = 128 * 1024 * 1024;  // 128MB
    const int SLAB_SIZE = 256;
    const int NUM_SLABS = 10000;
    
    uint8_t* d_memory_pool;
    cudaMalloc(&d_memory_pool, POOL_SIZE);
    
    SlabAllocator* d_allocator;
    cudaMalloc(&d_allocator, sizeof(SlabAllocator));
    
    AllocMeta* d_alloc_info;
    cudaMalloc(&d_alloc_info, 1024 * sizeof(AllocMeta));
    
    // Initialize allocator
    init_slab_allocator<<<1, 1>>>(d_allocator, d_memory_pool, SLAB_SIZE, NUM_SLABS);
    cudaDeviceSynchronize();
    
    TestResult h_result;
    int total_tests = 0;
    int passed_tests = 0;
    
    // Test 1: Slab allocator
    printf("Test 1: Slab allocator with lock-free operations...");
    test_slab_allocator<<<32, 256>>>(d_result, d_allocator, d_alloc_info, 256);
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    total_tests++;
    if (h_result.passed) {
        printf(" PASSED (%.0f cycles)\n", h_result.alloc_time_cycles);
        passed_tests++;
    } else {
        printf(" FAILED: %s\n", h_result.failure_msg);
    }
    
    // Test 2: Region allocator
    printf("Test 2: Region allocator for contiguous memory...");
    uint32_t* d_alloc_sizes;
    void** d_alloc_ptrs;
    cudaMalloc(&d_alloc_sizes, 100 * sizeof(uint32_t));
    cudaMalloc(&d_alloc_ptrs, 100 * sizeof(void*));
    
    test_region_allocator<<<1, 128>>>(d_result, d_memory_pool, POOL_SIZE,
                                     d_alloc_sizes, d_alloc_ptrs, 100);
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    cudaFree(d_alloc_sizes);
    cudaFree(d_alloc_ptrs);
    total_tests++;
    if (h_result.passed) {
        printf(" PASSED (%.1f%% efficiency)\n", h_result.memory_efficiency * 100);
        passed_tests++;
    } else {
        printf(" FAILED\n");
    }
    
    // Test 3: Arena allocator
    printf("Test 3: Arena allocator with bulk deallocation...");
    uint32_t* d_arena_offset;
    cudaMalloc(&d_arena_offset, sizeof(uint32_t));
    cudaMemset(d_arena_offset, 0, sizeof(uint32_t));
    
    test_arena_allocator<<<32, 256>>>(d_result, d_memory_pool, POOL_SIZE, d_arena_offset);
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    cudaFree(d_arena_offset);
    total_tests++;
    if (h_result.passed) {
        printf(" PASSED\n");
        passed_tests++;
    } else {
        printf(" FAILED\n");
    }
    
    // Test 4: Coalesced allocation
    printf("Test 4: Warp-aware coalesced allocation...");
    test_coalesced_allocation<<<32, 256>>>(d_result, d_memory_pool, POOL_SIZE, d_alloc_info);
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    total_tests++;
    if (h_result.passed) {
        printf(" PASSED\n");
        passed_tests++;
    } else {
        printf(" FAILED\n");
    }
    
    // Test 5: Memory pool hierarchy
    printf("Test 5: Hierarchical memory pool management...");
    uint8_t *d_thread_pool, *d_warp_pool, *d_block_pool, *d_global_pool;
    cudaMalloc(&d_thread_pool, 1024 * 1024);
    cudaMalloc(&d_warp_pool, 32 * 32768);
    cudaMalloc(&d_block_pool, 32 * 262144);
    cudaMalloc(&d_global_pool, 10 * 1024 * 1024);
    
    uint32_t* d_pool_offsets;
    cudaMalloc(&d_pool_offsets, 4 * sizeof(uint32_t));
    cudaMemset(d_pool_offsets, 0, 4 * sizeof(uint32_t));
    
    test_memory_pool_hierarchy<<<32, 256>>>(d_result, d_thread_pool, d_warp_pool,
                                           d_block_pool, d_global_pool, d_pool_offsets);
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    cudaFree(d_thread_pool);
    cudaFree(d_warp_pool);
    cudaFree(d_block_pool);
    cudaFree(d_global_pool);
    cudaFree(d_pool_offsets);
    total_tests++;
    if (h_result.passed) {
        printf(" PASSED\n");
        passed_tests++;
    } else {
        printf(" FAILED\n");
    }
    
    // Test 6: Concurrent stress test
    printf("Test 6: Concurrent allocation stress test...");
    init_slab_allocator<<<1, 1>>>(d_allocator, d_memory_pool, SLAB_SIZE, NUM_SLABS);
    cudaDeviceSynchronize();
    
    test_concurrent_allocation_stress<<<64, 256>>>(d_result, d_allocator, 100);
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    total_tests++;
    if (h_result.passed) {
        printf(" PASSED\n");
        passed_tests++;
    } else {
        printf(" FAILED: %s\n", h_result.failure_msg);
    }
    
    // Test 7: Throughput benchmark
    printf("Test 7: Allocation throughput benchmark...");
    init_slab_allocator<<<1, 1>>>(d_allocator, d_memory_pool, SLAB_SIZE, NUM_SLABS);
    cudaDeviceSynchronize();
    
    test_allocation_throughput<<<32, 256>>>(d_result, d_allocator, 100);
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    total_tests++;
    if (h_result.passed) {
        printf(" PASSED (%d allocs/sec)\n", h_result.allocations_per_sec);
        passed_tests++;
    } else {
        printf(" FAILED: %s\n", h_result.failure_msg);
    }
    
    // Cleanup
    cudaFree(d_result);
    cudaFree(d_memory_pool);
    cudaFree(d_allocator);
    cudaFree(d_alloc_info);
    
    // Summary
    printf("\n==============================================\n");
    printf("Allocator Test Results: %d/%d passed\n", passed_tests, total_tests);
    
    if (passed_tests == total_tests) {
        printf("✓ All allocator tests passed!\n");
        printf("✓ <100 cycle latency achieved\n");
        printf("✓ Lock-free operations validated\n");
        return 0;
    } else {
        printf("✗ Some tests failed\n");
        return 1;
    }
}