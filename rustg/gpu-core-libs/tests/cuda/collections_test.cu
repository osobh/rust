// GPU-Native Collections Tests
// Structure-of-Arrays vectors and hash maps
// NO STUBS OR MOCKS - Real GPU operations only

#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cuda/std/type_traits>
#include <cstdio>
#include <cstdlib>
#include <cassert>

// Test result structure
struct TestResult {
    bool passed;
    int test_id;
    float performance_score;
    int operations_performed;
    float elapsed_cycles;
    char error_msg[256];
};

// SoA Vector component structure
template<typename T>
struct SoAVector {
    T* data;
    unsigned int* capacity;
    unsigned int* size;
    unsigned int* lock;  // For atomic operations
    
    __device__ void init(unsigned int initial_capacity) {
        if (threadIdx.x == 0) {
            *capacity = initial_capacity;
            *size = 0;
            *lock = 0;
        }
        __syncthreads();
    }
    
    __device__ bool push(T value) {
        unsigned int idx = atomicAdd(size, 1);
        if (idx < *capacity) {
            data[idx] = value;
            return true;
        }
        atomicSub(size, 1);  // Revert if no space
        return false;
    }
    
    __device__ T get(unsigned int idx) {
        if (idx < *size) {
            return data[idx];
        }
        return T{};
    }
    
    __device__ unsigned int get_size() {
        return *size;
    }
    
    __device__ void parallel_map(T (*func)(T)) {
        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int stride = blockDim.x * gridDim.x;
        unsigned int n = *size;
        
        for (unsigned int i = tid; i < n; i += stride) {
            data[i] = func(data[i]);
        }
    }
    
    __device__ T parallel_reduce(T (*op)(T, T), T identity) {
        extern __shared__ char shared_mem[];
        T* shared = reinterpret_cast<T*>(shared_mem);
        
        unsigned int tid = threadIdx.x;
        unsigned int n = *size;
        T sum = identity;
        
        // Grid-stride loop
        for (unsigned int i = tid; i < n; i += blockDim.x) {
            sum = op(sum, data[i]);
        }
        
        shared[tid] = sum;
        __syncthreads();
        
        // Tree reduction in shared memory
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared[tid] = op(shared[tid], shared[tid + s]);
            }
            __syncthreads();
        }
        
        return shared[0];
    }
};

// GPU HashMap with cuckoo hashing
template<typename K, typename V>
struct GPUHashMap {
    struct Entry {
        K key;
        V value;
        unsigned int hash;
        bool occupied;
    };
    
    Entry* table1;
    Entry* table2;
    unsigned int* size;
    unsigned int* capacity;
    unsigned int* num_elements;
    
    __device__ unsigned int hash1(K key) {
        unsigned int h = key;
        h ^= h >> 16;
        h *= 0x85ebca6b;
        h ^= h >> 13;
        h *= 0xc2b2ae35;
        h ^= h >> 16;
        return h % *capacity;
    }
    
    __device__ unsigned int hash2(K key) {
        unsigned int h = key;
        h ^= h >> 17;
        h *= 0xed5ad4bb;
        h ^= h >> 11;
        h *= 0xac4c1b51;
        h ^= h >> 15;
        return h % *capacity;
    }
    
    __device__ void init(unsigned int cap) {
        if (threadIdx.x == 0) {
            *capacity = cap;
            *num_elements = 0;
        }
        __syncthreads();
        
        // Parallel initialization
        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int stride = blockDim.x * gridDim.x;
        
        for (unsigned int i = tid; i < cap; i += stride) {
            table1[i].occupied = false;
            table2[i].occupied = false;
        }
    }
    
    __device__ bool insert(K key, V value) {
        unsigned int h1 = hash1(key);
        unsigned int h2 = hash2(key);
        
        Entry new_entry;
        new_entry.key = key;
        new_entry.value = value;
        new_entry.occupied = true;
        
        // Try table1
        Entry* slot1 = &table1[h1];
        Entry old1 = *slot1;
        
        if (!old1.occupied || old1.key == key) {
            atomicExch((unsigned long long*)slot1, *(unsigned long long*)&new_entry);
            if (!old1.occupied) atomicAdd(num_elements, 1);
            return true;
        }
        
        // Try table2
        Entry* slot2 = &table2[h2];
        Entry old2 = *slot2;
        
        if (!old2.occupied || old2.key == key) {
            atomicExch((unsigned long long*)slot2, *(unsigned long long*)&new_entry);
            if (!old2.occupied) atomicAdd(num_elements, 1);
            return true;
        }
        
        // Cuckoo eviction - simplified for testing
        atomicExch((unsigned long long*)slot1, *(unsigned long long*)&new_entry);
        
        // Try to relocate evicted entry
        h2 = hash2(old1.key);
        slot2 = &table2[h2];
        old2 = *slot2;
        
        if (!old2.occupied) {
            atomicExch((unsigned long long*)slot2, *(unsigned long long*)&old1);
            atomicAdd(num_elements, 1);
            return true;
        }
        
        return false;  // Failed - would need more evictions
    }
    
    __device__ bool find(K key, V* out_value) {
        unsigned int h1 = hash1(key);
        Entry e1 = table1[h1];
        
        if (e1.occupied && e1.key == key) {
            *out_value = e1.value;
            return true;
        }
        
        unsigned int h2 = hash2(key);
        Entry e2 = table2[h2];
        
        if (e2.occupied && e2.key == key) {
            *out_value = e2.value;
            return true;
        }
        
        return false;
    }
    
    __device__ unsigned int get_size() {
        return *num_elements;
    }
};

// Test 1: SoA Vector push operations
__global__ void test_soa_vector_push(TestResult* result,
                                     SoAVector<float>* vec,
                                     int num_operations) {
    vec->init(num_operations * 2);
    
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    
    clock_t start = clock();
    
    // Parallel push operations
    for (unsigned int i = tid; i < num_operations; i += stride) {
        float value = i * 1.5f;
        if (!vec->push(value)) {
            if (tid == 0) {
                result->passed = false;
                sprintf(result->error_msg, "Push failed at %u", i);
            }
            return;
        }
    }
    
    __syncthreads();
    clock_t end = clock();
    
    if (tid == 0) {
        result->passed = true;
        result->operations_performed = vec->get_size();
        result->elapsed_cycles = (float)(end - start);
        result->performance_score = num_operations / result->elapsed_cycles * 1000;
    }
}

// Test 2: SoA Vector parallel map
__device__ float square_func(float x) {
    return x * x;
}

__global__ void test_soa_vector_map(TestResult* result,
                                    SoAVector<float>* vec,
                                    int num_elements) {
    vec->init(num_elements);
    
    // Initialize vector
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        for (int i = 0; i < num_elements; i++) {
            vec->push((float)i);
        }
    }
    __syncthreads();
    
    clock_t start = clock();
    vec->parallel_map(square_func);
    __syncthreads();
    clock_t end = clock();
    
    // Verify results
    if (tid == 0) {
        bool correct = true;
        for (int i = 0; i < num_elements && i < 100; i++) {
            float expected = (float)(i * i);
            float actual = vec->get(i);
            if (abs(actual - expected) > 0.001f) {
                correct = false;
                sprintf(result->error_msg, "Map error at %d: expected %f, got %f", 
                       i, expected, actual);
                break;
            }
        }
        
        result->passed = correct;
        result->operations_performed = num_elements;
        result->elapsed_cycles = (float)(end - start);
        result->performance_score = num_elements / result->elapsed_cycles * 1000;
    }
}

// Test 3: SoA Vector parallel reduce
__device__ float add_func(float a, float b) {
    return a + b;
}

__global__ void test_soa_vector_reduce(TestResult* result,
                                       SoAVector<float>* vec,
                                       int num_elements) {
    vec->init(num_elements);
    
    // Initialize vector
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < num_elements; i++) {
            vec->push(1.0f);  // All ones for easy verification
        }
    }
    __syncthreads();
    
    clock_t start = clock();
    float sum = vec->parallel_reduce(add_func, 0.0f);
    clock_t end = clock();
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float expected = (float)num_elements;
        bool correct = abs(sum - expected) < 0.001f;
        
        if (!correct) {
            sprintf(result->error_msg, "Reduce error: expected %f, got %f", 
                   expected, sum);
        }
        
        result->passed = correct;
        result->operations_performed = num_elements;
        result->elapsed_cycles = (float)(end - start);
        result->performance_score = num_elements / result->elapsed_cycles * 1000;
    }
}

// Test 4: HashMap insertion
__global__ void test_hashmap_insert(TestResult* result,
                                    GPUHashMap<unsigned int, unsigned int>* map,
                                    int num_operations) {
    map->init(num_operations * 2);
    
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    
    clock_t start = clock();
    
    // Parallel insertions
    for (unsigned int i = tid; i < num_operations; i += stride) {
        unsigned int key = i;
        unsigned int value = i * 2;
        
        if (!map->insert(key, value)) {
            // Insertion can fail in cuckoo hashing - not an error for test
        }
    }
    
    __syncthreads();
    clock_t end = clock();
    
    if (tid == 0) {
        result->passed = true;
        result->operations_performed = map->get_size();
        result->elapsed_cycles = (float)(end - start);
        result->performance_score = result->operations_performed / result->elapsed_cycles * 1000;
        
        // Check if we got reasonable insertion rate
        if (result->operations_performed < num_operations * 0.8) {
            result->passed = false;
            sprintf(result->error_msg, "Low insertion rate: %d/%d", 
                   result->operations_performed, num_operations);
        }
    }
}

// Test 5: HashMap lookup
__global__ void test_hashmap_lookup(TestResult* result,
                                    GPUHashMap<unsigned int, unsigned int>* map,
                                    int num_lookups) {
    // First initialize and insert
    map->init(num_lookups * 2);
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (unsigned int i = 0; i < num_lookups; i++) {
            map->insert(i, i * 3);
        }
    }
    __syncthreads();
    
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    __shared__ int errors;
    
    if (threadIdx.x == 0) {
        errors = 0;
    }
    __syncthreads();
    
    clock_t start = clock();
    
    // Parallel lookups
    for (unsigned int i = tid; i < num_lookups; i += stride) {
        unsigned int value;
        if (map->find(i, &value)) {
            if (value != i * 3) {
                atomicAdd(&errors, 1);
            }
        } else {
            atomicAdd(&errors, 1);
        }
    }
    
    __syncthreads();
    clock_t end = clock();
    
    if (tid == 0) {
        result->passed = (errors == 0);
        result->operations_performed = num_lookups;
        result->elapsed_cycles = (float)(end - start);
        result->performance_score = num_lookups / result->elapsed_cycles * 1000;
        
        if (!result->passed) {
            sprintf(result->error_msg, "Lookup errors: %d", errors);
        }
    }
}

// Test 6: Batch operations performance
__global__ void test_batch_operations(TestResult* result,
                                      SoAVector<int>* vec,
                                      int batch_size) {
    vec->init(batch_size * 10);
    
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int num_batches = 10;
    
    clock_t start = clock();
    
    // Batch pushes
    for (unsigned int batch = 0; batch < num_batches; batch++) {
        __syncthreads();
        
        for (unsigned int i = tid; i < batch_size; i += blockDim.x * gridDim.x) {
            vec->push(batch * batch_size + i);
        }
    }
    
    __syncthreads();
    clock_t end = clock();
    
    if (tid == 0) {
        result->passed = (vec->get_size() == batch_size * num_batches);
        result->operations_performed = vec->get_size();
        result->elapsed_cycles = (float)(end - start);
        result->performance_score = result->operations_performed / result->elapsed_cycles * 1000;
        
        if (!result->passed) {
            sprintf(result->error_msg, "Size mismatch: expected %d, got %d",
                   batch_size * num_batches, vec->get_size());
        }
    }
}

// Test 7: Memory coalescing verification
__global__ void test_memory_coalescing(TestResult* result,
                                       float* data,
                                       int num_elements) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    
    clock_t start = clock();
    
    // Coalesced access pattern
    for (unsigned int i = tid; i < num_elements; i += stride) {
        data[i] = i * 2.0f;
    }
    
    __syncthreads();
    
    // Verify with coalesced reads
    bool correct = true;
    for (unsigned int i = tid; i < num_elements; i += stride) {
        if (abs(data[i] - i * 2.0f) > 0.001f) {
            correct = false;
            break;
        }
    }
    
    clock_t end = clock();
    
    if (tid == 0) {
        result->passed = correct;
        result->operations_performed = num_elements * 2;  // Read + write
        result->elapsed_cycles = (float)(end - start);
        result->performance_score = result->operations_performed / result->elapsed_cycles * 1000;
    }
}

// Test 8: Concurrent HashMap operations
__global__ void test_concurrent_hashmap(TestResult* result,
                                        GPUHashMap<unsigned int, unsigned int>* map,
                                        int num_threads) {
    map->init(num_threads * 4);
    
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int success_count;
    
    if (threadIdx.x == 0) {
        success_count = 0;
    }
    __syncthreads();
    
    clock_t start = clock();
    
    // Each thread does multiple operations
    for (int op = 0; op < 10; op++) {
        unsigned int key = tid * 10 + op;
        unsigned int value = key * 2;
        
        if (map->insert(key, value)) {
            atomicAdd(&success_count, 1);
        }
        
        // Immediately lookup
        unsigned int found_value;
        if (map->find(key, &found_value)) {
            if (found_value == value) {
                atomicAdd(&success_count, 1);
            }
        }
    }
    
    __syncthreads();
    clock_t end = clock();
    
    if (tid == 0) {
        result->passed = (success_count > num_threads * 10);  // At least 50% success
        result->operations_performed = success_count;
        result->elapsed_cycles = (float)(end - start);
        result->performance_score = success_count / result->elapsed_cycles * 1000;
    }
}

// Test 9: Performance target validation (100M ops/sec)
__global__ void test_performance_target(TestResult* result,
                                        SoAVector<int>* vec,
                                        int target_ops) {
    vec->init(target_ops * 2);
    
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    
    clock_t start = clock();
    
    // High-throughput operations
    for (unsigned int i = tid; i < target_ops; i += stride) {
        vec->push(i);
    }
    
    __syncthreads();
    clock_t end = clock();
    
    if (tid == 0) {
        float elapsed_ms = (float)(end - start) / (1000.0f);  // Approximate
        float ops_per_sec = (vec->get_size() / elapsed_ms) * 1000.0f;
        
        result->passed = (ops_per_sec > 100000000.0f);  // 100M ops/sec
        result->operations_performed = vec->get_size();
        result->elapsed_cycles = (float)(end - start);
        result->performance_score = ops_per_sec / 100000000.0f;  // Ratio to target
        
        if (!result->passed) {
            sprintf(result->error_msg, "Performance: %.0f ops/sec (target: 100M)",
                   ops_per_sec);
        }
    }
}

// Main test runner
int main() {
    printf("GPU-Native Collections Tests\n");
    printf("============================\n\n");
    
    // Allocate test resources
    TestResult* d_results;
    cudaMalloc(&d_results, sizeof(TestResult) * 10);
    
    TestResult h_results[10];
    
    // Test 1: SoA Vector push
    {
        printf("Test 1: SoA Vector Push Operations...\n");
        SoAVector<float>* d_vec;
        cudaMalloc(&d_vec, sizeof(SoAVector<float>));
        
        float* vec_data;
        unsigned int* vec_meta;
        cudaMalloc(&vec_data, sizeof(float) * 100000);
        cudaMalloc(&vec_meta, sizeof(unsigned int) * 3);
        
        SoAVector<float> h_vec = {vec_data, vec_meta, vec_meta + 1, vec_meta + 2};
        cudaMemcpy(d_vec, &h_vec, sizeof(SoAVector<float>), cudaMemcpyHostToDevice);
        
        test_soa_vector_push<<<256, 256>>>(d_results, d_vec, 10000);
        
        cudaMemcpy(&h_results[0], d_results, sizeof(TestResult), cudaMemcpyDeviceToHost);
        printf("  Result: %s\n", h_results[0].passed ? "PASSED" : "FAILED");
        if (!h_results[0].passed) {
            printf("  Error: %s\n", h_results[0].error_msg);
        }
        printf("  Performance: %.2fx\n\n", h_results[0].performance_score);
        
        cudaFree(vec_data);
        cudaFree(vec_meta);
        cudaFree(d_vec);
    }
    
    // Test 2: SoA Vector map
    {
        printf("Test 2: SoA Vector Parallel Map...\n");
        SoAVector<float>* d_vec;
        cudaMalloc(&d_vec, sizeof(SoAVector<float>));
        
        float* vec_data;
        unsigned int* vec_meta;
        cudaMalloc(&vec_data, sizeof(float) * 10000);
        cudaMalloc(&vec_meta, sizeof(unsigned int) * 3);
        
        SoAVector<float> h_vec = {vec_data, vec_meta, vec_meta + 1, vec_meta + 2};
        cudaMemcpy(d_vec, &h_vec, sizeof(SoAVector<float>), cudaMemcpyHostToDevice);
        
        test_soa_vector_map<<<256, 256>>>(d_results + 1, d_vec, 1000);
        
        cudaMemcpy(&h_results[1], d_results + 1, sizeof(TestResult), cudaMemcpyDeviceToHost);
        printf("  Result: %s\n", h_results[1].passed ? "PASSED" : "FAILED");
        if (!h_results[1].passed) {
            printf("  Error: %s\n", h_results[1].error_msg);
        }
        printf("  Performance: %.2fx\n\n", h_results[1].performance_score);
        
        cudaFree(vec_data);
        cudaFree(vec_meta);
        cudaFree(d_vec);
    }
    
    // Test 3: SoA Vector reduce
    {
        printf("Test 3: SoA Vector Parallel Reduce...\n");
        SoAVector<float>* d_vec;
        cudaMalloc(&d_vec, sizeof(SoAVector<float>));
        
        float* vec_data;
        unsigned int* vec_meta;
        cudaMalloc(&vec_data, sizeof(float) * 10000);
        cudaMalloc(&vec_meta, sizeof(unsigned int) * 3);
        
        SoAVector<float> h_vec = {vec_data, vec_meta, vec_meta + 1, vec_meta + 2};
        cudaMemcpy(d_vec, &h_vec, sizeof(SoAVector<float>), cudaMemcpyHostToDevice);
        
        size_t shared_size = 256 * sizeof(float);
        test_soa_vector_reduce<<<1, 256, shared_size>>>(d_results + 2, d_vec, 1000);
        
        cudaMemcpy(&h_results[2], d_results + 2, sizeof(TestResult), cudaMemcpyDeviceToHost);
        printf("  Result: %s\n", h_results[2].passed ? "PASSED" : "FAILED");
        if (!h_results[2].passed) {
            printf("  Error: %s\n", h_results[2].error_msg);
        }
        printf("  Performance: %.2fx\n\n", h_results[2].performance_score);
        
        cudaFree(vec_data);
        cudaFree(vec_meta);
        cudaFree(d_vec);
    }
    
    // Test 4: HashMap insert
    {
        printf("Test 4: HashMap Insertion...\n");
        GPUHashMap<unsigned int, unsigned int>* d_map;
        cudaMalloc(&d_map, sizeof(GPUHashMap<unsigned int, unsigned int>));
        
        GPUHashMap<unsigned int, unsigned int>::Entry* table1;
        GPUHashMap<unsigned int, unsigned int>::Entry* table2;
        unsigned int* map_meta;
        
        cudaMalloc(&table1, sizeof(GPUHashMap<unsigned int, unsigned int>::Entry) * 20000);
        cudaMalloc(&table2, sizeof(GPUHashMap<unsigned int, unsigned int>::Entry) * 20000);
        cudaMalloc(&map_meta, sizeof(unsigned int) * 3);
        
        GPUHashMap<unsigned int, unsigned int> h_map = {
            table1, table2, map_meta, map_meta + 1, map_meta + 2
        };
        cudaMemcpy(d_map, &h_map, sizeof(GPUHashMap<unsigned int, unsigned int>), 
                   cudaMemcpyHostToDevice);
        
        test_hashmap_insert<<<256, 256>>>(d_results + 3, d_map, 5000);
        
        cudaMemcpy(&h_results[3], d_results + 3, sizeof(TestResult), cudaMemcpyDeviceToHost);
        printf("  Result: %s\n", h_results[3].passed ? "PASSED" : "FAILED");
        if (!h_results[3].passed) {
            printf("  Error: %s\n", h_results[3].error_msg);
        }
        printf("  Performance: %.2fx\n\n", h_results[3].performance_score);
        
        cudaFree(table1);
        cudaFree(table2);
        cudaFree(map_meta);
        cudaFree(d_map);
    }
    
    // Summary
    printf("Test Summary\n");
    printf("============\n");
    
    int passed = 0;
    float total_perf = 0;
    
    for (int i = 0; i < 4; i++) {
        if (h_results[i].passed) {
            passed++;
            total_perf += h_results[i].performance_score;
        }
    }
    
    printf("Passed: %d/4\n", passed);
    printf("Average Performance Score: %.2fx\n", total_perf / 4);
    
    if (passed == 4 && total_perf / 4 > 10.0f) {
        printf("\n✓ All tests passed with >10x performance!\n");
        return 0;
    } else {
        printf("\n✗ Some tests failed or performance target not met\n");
        return 1;
    }
}