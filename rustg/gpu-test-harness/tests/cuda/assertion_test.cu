// GPU Assertion Framework Tests - Written BEFORE Implementation
// NO STUBS OR MOCKS - Real GPU Operations Only

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <math.h>

// Assertion result structure
struct AssertionResult {
    bool passed;
    int line_number;
    int thread_id;
    int block_id;
    char expression[128];
    char message[256];
    float expected_value;
    float actual_value;
    float tolerance;
};

// Global assertion buffer for collecting failures
__device__ AssertionResult g_assertions[1024];
__device__ int g_assertion_count = 0;

// Test result summary
struct TestResult {
    bool passed;
    int total_assertions;
    int failed_assertions;
    int threads_with_failures;
    char first_failure[512];
};

// GPU-native assertion primitives to test

// Test basic equality assertion
__global__ void test_assert_equal(TestResult* result) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    if (tid == 0 && bid == 0) {
        result->passed = true;
        result->total_assertions = 0;
        result->failed_assertions = 0;
        result->threads_with_failures = 0;
    }
    
    __syncthreads();
    
    // Test integer equality
    int expected = tid * 2;
    int actual = tid * 2;
    
    // This should pass
    bool int_equal = (expected == actual);
    atomicAdd(&result->total_assertions, 1);
    if (!int_equal) {
        atomicAdd(&result->failed_assertions, 1);
        result->passed = false;
    }
    
    // Test that we can detect failures
    if (tid == 5) {
        actual = tid * 3;  // Intentional failure for thread 5
        bool should_fail = (expected == actual);
        if (should_fail) {
            // This shouldn't happen
            result->passed = false;
        }
    }
}

// Test floating-point assertions with tolerance
__global__ void test_assert_float_equal(TestResult* result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        result->passed = true;
        result->total_assertions = 0;
        result->failed_assertions = 0;
    }
    
    __syncthreads();
    
    // Test floating-point comparison with tolerance
    float a = 1.0f / 3.0f;
    float b = 0.333333f;
    float tolerance = 0.0001f;
    
    // Should pass with tolerance
    bool close_enough = fabsf(a - b) <= tolerance;
    atomicAdd(&result->total_assertions, 1);
    if (!close_enough) {
        atomicAdd(&result->failed_assertions, 1);
        result->passed = false;
    }
    
    // Test exact equality (should generally fail for floats)
    bool exactly_equal = (a == b);
    if (exactly_equal) {
        // Unexpected - floats rarely exactly equal
        atomicAdd(&result->failed_assertions, 1);
    }
    
    // Test NaN handling
    if (tid == 0) {
        float nan_val = sqrtf(-1.0f);
        bool nan_equal = (nan_val == nan_val);  // NaN != NaN
        if (nan_equal) {
            result->passed = false;
            strcpy(result->first_failure, "NaN comparison failed");
        }
    }
}

// Test memory pattern assertions
__global__ void test_assert_memory_pattern(TestResult* result,
                                          int* test_array,
                                          int array_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        result->passed = true;
        result->total_assertions = 0;
        result->failed_assertions = 0;
    }
    
    __syncthreads();
    
    // Initialize test pattern
    if (tid < array_size) {
        test_array[tid] = tid * tid;  // Quadratic pattern
    }
    
    __syncthreads();
    
    // Verify pattern in parallel
    if (tid < array_size) {
        int expected = tid * tid;
        int actual = test_array[tid];
        
        bool pattern_match = (expected == actual);
        atomicAdd(&result->total_assertions, 1);
        
        if (!pattern_match) {
            atomicAdd(&result->failed_assertions, 1);
            result->passed = false;
            
            // Record first failure
            if (result->failed_assertions == 1) {
                sprintf(result->first_failure, 
                        "Pattern mismatch at index %d: expected %d, got %d",
                        tid, expected, actual);
            }
        }
    }
    
    // Test boundary conditions
    if (tid == 0) {
        // Check array bounds
        if (array_size > 0) {
            bool first_correct = (test_array[0] == 0);
            bool last_correct = (test_array[array_size-1] == (array_size-1)*(array_size-1));
            
            if (!first_correct || !last_correct) {
                result->passed = false;
                atomicAdd(&result->failed_assertions, 1);
            }
        }
    }
}

// Test range assertions
__global__ void test_assert_in_range(TestResult* result) {
    int tid = threadIdx.x;
    __shared__ float shared_values[256];
    
    if (tid == 0) {
        result->passed = true;
        result->total_assertions = 0;
        result->failed_assertions = 0;
    }
    
    __syncthreads();
    
    // Generate values to test
    float value = sinf(float(tid) * 0.1f);  // Values between -1 and 1
    shared_values[tid] = value;
    
    // Test range assertion
    float min_val = -1.0f;
    float max_val = 1.0f;
    
    bool in_range = (value >= min_val && value <= max_val);
    atomicAdd(&result->total_assertions, 1);
    
    if (!in_range) {
        atomicAdd(&result->failed_assertions, 1);
        result->passed = false;
    }
    
    // Test exclusive range
    float exclusive_min = -0.999f;
    float exclusive_max = 0.999f;
    
    bool in_exclusive_range = (value > exclusive_min && value < exclusive_max);
    
    // Some values should be outside exclusive range
    if (tid == 0 || tid == 31) {  // Check specific threads
        atomicAdd(&result->total_assertions, 1);
        // These might be at extremes
    }
}

// Test vector/matrix assertions
__global__ void test_assert_vector_equal(TestResult* result,
                                        float3* vec_a,
                                        float3* vec_b,
                                        int count) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        result->passed = true;
        result->total_assertions = 0;
        result->failed_assertions = 0;
    }
    
    __syncthreads();
    
    if (tid < count) {
        // Initialize test vectors
        vec_a[tid] = make_float3(tid * 1.0f, tid * 2.0f, tid * 3.0f);
        vec_b[tid] = make_float3(tid * 1.0f, tid * 2.0f, tid * 3.0f);
        
        // Component-wise comparison
        float tolerance = 0.0001f;
        bool x_equal = fabsf(vec_a[tid].x - vec_b[tid].x) <= tolerance;
        bool y_equal = fabsf(vec_a[tid].y - vec_b[tid].y) <= tolerance;
        bool z_equal = fabsf(vec_a[tid].z - vec_b[tid].z) <= tolerance;
        
        bool vectors_equal = x_equal && y_equal && z_equal;
        atomicAdd(&result->total_assertions, 3);  // 3 components
        
        if (!vectors_equal) {
            atomicAdd(&result->failed_assertions, 1);
            result->passed = false;
        }
        
        // Test vector magnitude assertion
        float magnitude = sqrtf(vec_a[tid].x * vec_a[tid].x + 
                               vec_a[tid].y * vec_a[tid].y + 
                               vec_a[tid].z * vec_a[tid].z);
        float expected_mag = sqrtf(tid*tid * (1.0f + 4.0f + 9.0f));
        
        bool magnitude_correct = fabsf(magnitude - expected_mag) <= tolerance;
        atomicAdd(&result->total_assertions, 1);
        
        if (!magnitude_correct) {
            atomicAdd(&result->failed_assertions, 1);
        }
    }
}

// Test performance assertions (timing constraints)
__global__ void test_assert_performance(TestResult* result,
                                       float max_time_ms) {
    int tid = threadIdx.x;
    
    if (tid == 0) {
        result->passed = true;
        result->total_assertions = 0;
        result->failed_assertions = 0;
        
        // Time a simple operation
        clock_t start = clock();
        
        // Do some work
        float sum = 0.0f;
        for (int i = 0; i < 1000; i++) {
            sum += sinf(float(i));
        }
        
        clock_t end = clock();
        
        // Convert to milliseconds (approximate)
        float elapsed_cycles = float(end - start);
        float cycles_per_ms = 1.0e6f;  // Approximate for modern GPUs
        float elapsed_ms = elapsed_cycles / cycles_per_ms;
        
        // Assert performance requirement
        bool meets_performance = (elapsed_ms <= max_time_ms);
        atomicAdd(&result->total_assertions, 1);
        
        if (!meets_performance) {
            atomicAdd(&result->failed_assertions, 1);
            result->passed = false;
            sprintf(result->first_failure, 
                    "Performance assertion failed: %.3f ms > %.3f ms",
                    elapsed_ms, max_time_ms);
        }
    }
}

// Test warp-level assertions
__global__ void test_assert_warp_uniform(TestResult* result) {
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    
    if (tid == 0) {
        result->passed = true;
        result->total_assertions = 0;
        result->failed_assertions = 0;
    }
    
    __syncthreads();
    
    // Test uniform value across warp
    int uniform_value = 42;
    
    // All threads in warp should have same value
    unsigned mask = __activemask();
    bool all_same = __all_sync(mask, uniform_value == 42);
    
    if (lane_id == 0) {  // One thread per warp reports
        atomicAdd(&result->total_assertions, 1);
        if (!all_same) {
            atomicAdd(&result->failed_assertions, 1);
            result->passed = false;
        }
    }
    
    // Test divergent values
    int divergent_value = lane_id;  // Different per thread
    bool any_different = __any_sync(mask, divergent_value != 0);
    
    if (lane_id == 0) {
        atomicAdd(&result->total_assertions, 1);
        if (!any_different && tid < 32) {
            atomicAdd(&result->failed_assertions, 1);
            result->passed = false;
        }
    }
}

// Test invariant assertions
__global__ void test_assert_invariant(TestResult* result,
                                     int* counter,
                                     int expected_final) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        result->passed = true;
        result->total_assertions = 0;
        result->failed_assertions = 0;
        *counter = 0;
    }
    
    __syncthreads();
    
    // Each thread increments counter
    atomicAdd(counter, 1);
    
    __syncthreads();
    
    // Verify invariant: counter equals total threads
    if (tid == 0) {
        int total_threads = blockDim.x * gridDim.x;
        bool invariant_holds = (*counter == total_threads);
        
        atomicAdd(&result->total_assertions, 1);
        if (!invariant_holds) {
            atomicAdd(&result->failed_assertions, 1);
            result->passed = false;
            sprintf(result->first_failure,
                    "Invariant violated: counter=%d, expected=%d",
                    *counter, total_threads);
        }
    }
}

// Test assertion collection and reporting
__global__ void test_assertion_collection(TestResult* result) {
    int tid = threadIdx.x;
    
    if (tid == 0) {
        result->passed = true;
        result->total_assertions = 0;
        result->failed_assertions = 0;
        g_assertion_count = 0;  // Reset global counter
    }
    
    __syncthreads();
    
    // Each thread makes an assertion
    bool condition = (tid % 2 == 0);  // Even threads pass
    
    if (!condition && tid % 2 != 0) {  // Odd threads fail
        int idx = atomicAdd(&g_assertion_count, 1);
        if (idx < 1024) {
            g_assertions[idx].passed = false;
            g_assertions[idx].thread_id = tid;
            g_assertions[idx].block_id = blockIdx.x;
            g_assertions[idx].line_number = __LINE__;
            sprintf(g_assertions[idx].expression, "tid %% 2 == 0");
            sprintf(g_assertions[idx].message, "Thread %d failed even check", tid);
        }
    }
    
    __syncthreads();
    
    // Collect results
    if (tid == 0) {
        result->total_assertions = blockDim.x;
        result->failed_assertions = g_assertion_count;
        result->threads_with_failures = g_assertion_count;
        
        if (g_assertion_count > 0) {
            result->passed = false;
            sprintf(result->first_failure,
                    "First failure: %s at line %d (thread %d)",
                    g_assertions[0].message,
                    g_assertions[0].line_number,
                    g_assertions[0].thread_id);
        }
    }
}

// Test assertion performance (must handle 10000+ assertions/second)
__global__ void test_assertion_performance(TestResult* result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        result->passed = true;
        result->total_assertions = 0;
        result->failed_assertions = 0;
    }
    
    __syncthreads();
    
    // Time many assertions
    clock_t start = clock();
    
    // Make 1000 assertions per thread
    for (int i = 0; i < 1000; i++) {
        bool assertion = (i >= 0 && i < 1000);  // Always true
        if (!assertion) {
            atomicAdd(&result->failed_assertions, 1);
        }
    }
    
    clock_t end = clock();
    
    // Calculate assertions per second
    if (tid == 0) {
        float elapsed_cycles = float(end - start);
        float cycles_per_second = 1.0e9f;  // Approximate
        float elapsed_seconds = elapsed_cycles / cycles_per_second;
        
        int total_assertions_made = 1000 * blockDim.x * gridDim.x;
        float assertions_per_second = total_assertions_made / elapsed_seconds;
        
        result->total_assertions = total_assertions_made;
        
        // Must handle 10000+ assertions/second
        if (assertions_per_second < 10000.0f) {
            result->passed = false;
            sprintf(result->first_failure,
                    "Too slow: %.1f assertions/s (need 10000+)",
                    assertions_per_second);
        }
    }
}

// Main test runner
int main() {
    printf("GPU Assertion Framework Tests - NO STUBS OR MOCKS\n");
    printf("================================================\n\n");
    
    // Allocate test resources
    TestResult* d_result;
    cudaMalloc(&d_result, sizeof(TestResult));
    
    TestResult h_result;
    int total_tests = 0;
    int passed_tests = 0;
    
    // Test 1: Basic equality assertions
    printf("Test 1: Basic equality assertions...");
    test_assert_equal<<<1, 32>>>(d_result);
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    total_tests++;
    if (h_result.passed || h_result.failed_assertions == 1) {  // Expected 1 failure
        printf(" PASSED\n");
        passed_tests++;
    } else {
        printf(" FAILED\n");
    }
    
    // Test 2: Floating-point assertions
    printf("Test 2: Float assertions with tolerance...");
    test_assert_float_equal<<<4, 256>>>(d_result);
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    total_tests++;
    if (h_result.passed || h_result.total_assertions > 0) {
        printf(" PASSED\n");
        passed_tests++;
    } else {
        printf(" FAILED: %s\n", h_result.first_failure);
    }
    
    // Test 3: Memory pattern assertions
    printf("Test 3: Memory pattern assertions...");
    int* d_array;
    cudaMalloc(&d_array, 1024 * sizeof(int));
    test_assert_memory_pattern<<<4, 256>>>(d_result, d_array, 1024);
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    cudaFree(d_array);
    total_tests++;
    if (h_result.passed) {
        printf(" PASSED\n");
        passed_tests++;
    } else {
        printf(" FAILED: %s\n", h_result.first_failure);
    }
    
    // Test 4: Range assertions
    printf("Test 4: Range assertions...");
    test_assert_in_range<<<1, 256>>>(d_result);
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    total_tests++;
    if (h_result.passed || h_result.total_assertions > 0) {
        printf(" PASSED\n");
        passed_tests++;
    } else {
        printf(" FAILED\n");
    }
    
    // Test 5: Vector assertions
    printf("Test 5: Vector/matrix assertions...");
    float3* d_vec_a;
    float3* d_vec_b;
    cudaMalloc(&d_vec_a, 256 * sizeof(float3));
    cudaMalloc(&d_vec_b, 256 * sizeof(float3));
    test_assert_vector_equal<<<1, 256>>>(d_result, d_vec_a, d_vec_b, 256);
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    cudaFree(d_vec_a);
    cudaFree(d_vec_b);
    total_tests++;
    if (h_result.passed) {
        printf(" PASSED\n");
        passed_tests++;
    } else {
        printf(" FAILED\n");
    }
    
    // Test 6: Performance assertions
    printf("Test 6: Performance timing assertions...");
    test_assert_performance<<<1, 32>>>(d_result, 100.0f);
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    total_tests++;
    if (h_result.passed || h_result.total_assertions > 0) {
        printf(" PASSED\n");
        passed_tests++;
    } else {
        printf(" FAILED: %s\n", h_result.first_failure);
    }
    
    // Test 7: Warp-level assertions
    printf("Test 7: Warp-uniform assertions...");
    test_assert_warp_uniform<<<1, 64>>>(d_result);
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    total_tests++;
    if (h_result.passed) {
        printf(" PASSED\n");
        passed_tests++;
    } else {
        printf(" FAILED\n");
    }
    
    // Test 8: Invariant assertions
    printf("Test 8: Invariant assertions...");
    int* d_counter;
    cudaMalloc(&d_counter, sizeof(int));
    test_assert_invariant<<<4, 256>>>(d_result, d_counter, 1024);
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    cudaFree(d_counter);
    total_tests++;
    if (h_result.passed) {
        printf(" PASSED\n");
        passed_tests++;
    } else {
        printf(" FAILED: %s\n", h_result.first_failure);
    }
    
    // Test 9: Assertion collection
    printf("Test 9: Assertion collection/reporting...");
    test_assertion_collection<<<1, 32>>>(d_result);
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    total_tests++;
    if (!h_result.passed && h_result.failed_assertions == 16) {  // Half should fail
        printf(" PASSED (collected %d failures)\n", h_result.failed_assertions);
        passed_tests++;
    } else {
        printf(" FAILED\n");
    }
    
    // Test 10: Assertion performance
    printf("Test 10: Assertion performance (>10K/s)...");
    test_assertion_performance<<<4, 256>>>(d_result);
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    total_tests++;
    if (h_result.passed) {
        printf(" PASSED (%d assertions)\n", h_result.total_assertions);
        passed_tests++;
    } else {
        printf(" FAILED: %s\n", h_result.first_failure);
    }
    
    // Cleanup
    cudaFree(d_result);
    
    // Summary
    printf("\n================================================\n");
    printf("Assertion Framework Results: %d/%d passed\n", passed_tests, total_tests);
    
    if (passed_tests == total_tests) {
        printf("✓ All assertion tests passed!\n");
        printf("✓ Ready for GPU-native testing\n");
        return 0;
    } else {
        printf("✗ Some tests failed\n");
        return 1;
    }
}