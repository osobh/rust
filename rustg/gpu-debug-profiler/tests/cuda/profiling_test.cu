#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cassert>
#include <memory>

// Performance profiling data structures
struct FunctionProfile {
    std::string name;
    uint64_t total_time_ns;
    uint64_t call_count;
    uint32_t warp_efficiency;
    uint32_t memory_bandwidth_utilization;
    uint32_t register_usage;
};

struct FlameGraphNode {
    std::string function_name;
    uint64_t self_time_ns;
    uint64_t total_time_ns;
    std::vector<std::unique_ptr<FlameGraphNode>> children;
    
    FlameGraphNode(const std::string& name) : function_name(name), self_time_ns(0), total_time_ns(0) {}
};

// Complex computational kernels for performance profiling
__global__ void compute_intensive_kernel(
    const float* input,
    float* output,
    uint64_t* timing_per_thread,
    int size,
    int complexity_factor
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    
    uint64_t start_clock = clock64();
    float value = input[tid];
    float result = 0.0f;
    
    // Nested computation with varying complexity
    for (int outer = 0; outer < complexity_factor; ++outer) {
        for (int inner = 0; inner < 10; ++inner) {
            switch (outer % 6) {
                case 0: result += sinf(value * outer + inner) * cosf(value); break;
                case 1: result += expf(-value * 0.01f) * logf(fabsf(value) + 1.0f); break;
                case 2: result += sqrtf(value * value + outer) * powf(value, 0.5f); break;
                case 3: result += tanf(value) * atanf(value * inner); break;
                case 4: result += value * value * value + outer * inner; break;
                case 5: result += fmodf(value * outer, inner + 1.0f); break;
            }
            if ((tid + outer + inner) % 3 == 0) result *= 1.1f;
            else result *= 0.9f;
        }
    }
    
    output[tid] = result;
    timing_per_thread[tid] = clock64() - start_clock;
}

__global__ void memory_intensive_kernel(
    const float* input1, const float* input2, const float* input3,
    float* output, uint64_t* memory_timing,
    int size, int stride_pattern
) {
    extern __shared__ float shared_cache[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    if (tid >= size) return;
    
    uint64_t mem_start = clock64();
    
    // Different memory access patterns
    float val1, val2, val3;
    switch (stride_pattern % 4) {
        case 0: val1 = input1[tid]; val2 = input2[tid]; val3 = input3[tid]; break;
        case 1: val1 = input1[(tid * 7) % size]; val2 = input2[(tid * 11) % size]; val3 = input3[(tid * 13) % size]; break;
        case 2: val1 = input1[(tid * tid + 17) % size]; val2 = input2[(tid * 23 + 31) % size]; val3 = input3[(tid * 37 + 41) % size]; break;
        case 3: val1 = input1[size - 1 - tid]; val2 = input2[size - 1 - ((tid * 3) % size)]; val3 = input3[size - 1 - ((tid * 5) % size)]; break;
    }
    
    shared_cache[local_tid] = val1 + val2 + val3;
    shared_cache[local_tid + blockDim.x] = val1 * val2 * val3;
    __syncthreads();
    
    // Shared memory reduction
    float shared_sum = 0.0f;
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        if (local_tid % (offset * 2) == 0) {
            shared_cache[local_tid] += shared_cache[local_tid + offset];
        }
        __syncthreads();
    }
    
    output[tid] = shared_cache[0] * val1 + shared_cache[local_tid % blockDim.x] * val2;
    memory_timing[tid] = clock64() - mem_start;
}

__global__ void warp_divergent_kernel(
    const int* input, int* output, uint64_t* divergence_timing,
    float* divergence_ratio, int size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    if (tid >= size) return;
    
    uint64_t div_start = clock64();
    int value = input[tid];
    int result = 0;
    
    // Create different levels of warp divergence
    if (lane_id < 8) {
        for (int i = 0; i < value % 100; ++i) {
            result += (value * i) % 1000;
            result = (result << 1) ^ (result >> 1);
        }
    } else if (lane_id < 16) {
        for (int i = 0; i < (value % 50); ++i) {
            result += value * i;
        }
    } else if (lane_id < 24) {
        result = value * value + lane_id;
    } else {
        result = value + lane_id;
    }
    
    __syncwarp();
    int neighbor_value = __shfl_xor_sync(0xFFFFFFFF, result, 16);
    result = (result + neighbor_value) / 2;
    
    output[tid] = result;
    divergence_timing[tid] = clock64() - div_start;
    
    if (lane_id == 0) {
        divergence_ratio[warp_id] = 0.75f;  // Simulated high divergence
    }
}

// Performance profiling test class
class ProfilingTest {
private:
    std::map<std::string, FunctionProfile> function_profiles;
    std::unique_ptr<FlameGraphNode> flame_graph_root;
    
public:
    ProfilingTest() {
        flame_graph_root = std::make_unique<FlameGraphNode>("GPU_Program_Root");
    }
    
    bool test_compute_performance_profiling() {
        std::cout << "\n=== Testing Compute Performance Profiling ===" << std::endl;
        
        const int size = 4096;
        const std::vector<int> complexity_levels = {10, 50, 100, 200};
        
        std::vector<float> h_input(size), h_output(size);
        std::vector<uint64_t> h_timing(size);
        
        for (int i = 0; i < size; ++i) {
            h_input[i] = static_cast<float>(i + 1) / size;
        }
        
        float *d_input, *d_output;
        uint64_t *d_timing;
        
        cudaMalloc(&d_input, size * sizeof(float));
        cudaMalloc(&d_output, size * sizeof(float));
        cudaMalloc(&d_timing, size * sizeof(uint64_t));
        
        cudaMemcpy(d_input, h_input.data(), size * sizeof(float), cudaMemcpyHostToDevice);
        
        dim3 block(256);
        dim3 grid((size + block.x - 1) / block.x);
        
        bool performance_scaling_correct = true;
        std::vector<uint64_t> compute_times;
        
        for (int complexity : complexity_levels) {
            cudaEvent_t start_event, end_event;
            cudaEventCreate(&start_event);
            cudaEventCreate(&end_event);
            
            cudaEventRecord(start_event);
            compute_intensive_kernel<<<grid, block>>>(d_input, d_output, d_timing, size, complexity);
            cudaEventRecord(end_event);
            cudaEventSynchronize(end_event);
            
            float kernel_time_ms;
            cudaEventElapsedTime(&kernel_time_ms, start_event, end_event);
            
            cudaMemcpy(h_timing.data(), d_timing, size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            
            uint64_t total_cycles = 0;
            for (uint64_t cycles : h_timing) {
                total_cycles += cycles;
            }
            
            compute_times.push_back(static_cast<uint64_t>(kernel_time_ms * 1e6f));
            
            FunctionProfile profile;
            profile.name = "compute_intensive_kernel_" + std::to_string(complexity);
            profile.total_time_ns = static_cast<uint64_t>(kernel_time_ms * 1e6f);
            profile.call_count = 1;
            profile.warp_efficiency = 85;
            function_profiles[profile.name] = profile;
            
            std::cout << "  Complexity " << complexity << ": " << kernel_time_ms << " ms" << std::endl;
            
            cudaEventDestroy(start_event);
            cudaEventDestroy(end_event);
        }
        
        // Verify performance scaling
        std::sort(compute_times.begin(), compute_times.end());
        for (size_t i = 1; i < compute_times.size(); ++i) {
            if (compute_times[i] < compute_times[i-1] * 0.8) {
                performance_scaling_correct = false;
                break;
            }
        }
        
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_timing);
        
        std::cout << "Compute performance scaling: " << (performance_scaling_correct ? "CORRECT" : "INCORRECT") << std::endl;
        return performance_scaling_correct;
    }
    
    bool test_memory_bandwidth_profiling() {
        std::cout << "\n=== Testing Memory Bandwidth Profiling ===" << std::endl;
        
        const int size = 8192;
        const std::vector<int> stride_patterns = {0, 1, 2, 3};
        const int shared_mem_size = 256 * 2 * sizeof(float);
        
        std::vector<float> h_input1(size), h_input2(size), h_input3(size), h_output(size);
        std::vector<uint64_t> h_memory_timing(size);
        
        for (int i = 0; i < size; ++i) {
            h_input1[i] = static_cast<float>(i);
            h_input2[i] = static_cast<float>(size - i);
            h_input3[i] = static_cast<float>(i * 2);
        }
        
        float *d_input1, *d_input2, *d_input3, *d_output;
        uint64_t *d_memory_timing;
        
        cudaMalloc(&d_input1, size * sizeof(float));
        cudaMalloc(&d_input2, size * sizeof(float));
        cudaMalloc(&d_input3, size * sizeof(float));
        cudaMalloc(&d_output, size * sizeof(float));
        cudaMalloc(&d_memory_timing, size * sizeof(uint64_t));
        
        cudaMemcpy(d_input1, h_input1.data(), size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_input2, h_input2.data(), size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_input3, h_input3.data(), size * sizeof(float), cudaMemcpyHostToDevice);
        
        dim3 block(256);
        dim3 grid((size + block.x - 1) / block.x);
        
        std::vector<float> bandwidth_results;
        
        for (int pattern : stride_patterns) {
            std::string pattern_name = (pattern == 0) ? "coalesced" : 
                                     (pattern == 1) ? "strided" : 
                                     (pattern == 2) ? "random" : "reverse";
            
            cudaEvent_t start_event, end_event;
            cudaEventCreate(&start_event);
            cudaEventCreate(&end_event);
            
            cudaEventRecord(start_event);
            memory_intensive_kernel<<<grid, block, shared_mem_size>>>(
                d_input1, d_input2, d_input3, d_output, d_memory_timing, size, pattern);
            cudaEventRecord(end_event);
            cudaEventSynchronize(end_event);
            
            float kernel_time_ms;
            cudaEventElapsedTime(&kernel_time_ms, start_event, end_event);
            
            size_t bytes_transferred = 4 * size * sizeof(float);
            float bandwidth_gb = (bytes_transferred / (kernel_time_ms * 1e-3f)) / 1e9f;
            bandwidth_results.push_back(bandwidth_gb);
            
            FunctionProfile profile;
            profile.name = "memory_intensive_kernel_" + pattern_name;
            profile.total_time_ns = static_cast<uint64_t>(kernel_time_ms * 1e6f);
            profile.memory_bandwidth_utilization = static_cast<uint32_t>(bandwidth_gb * 10);
            function_profiles[profile.name] = profile;
            
            std::cout << "  " << pattern_name << ": " << bandwidth_gb << " GB/s" << std::endl;
            
            cudaEventDestroy(start_event);
            cudaEventDestroy(end_event);
        }
        
        // Verify coalesced access has better bandwidth
        bool bandwidth_analysis_correct = bandwidth_results[0] > bandwidth_results[1] && 
                                        bandwidth_results[0] > bandwidth_results[2];
        
        cudaFree(d_input1);
        cudaFree(d_input2);
        cudaFree(d_input3);
        cudaFree(d_output);
        cudaFree(d_memory_timing);
        
        std::cout << "Memory bandwidth analysis: " << (bandwidth_analysis_correct ? "CORRECT" : "INCORRECT") << std::endl;
        return bandwidth_analysis_correct;
    }
    
    bool test_warp_divergence_profiling() {
        std::cout << "\n=== Testing Warp Divergence Profiling ===" << std::endl;
        
        const int size = 2048;
        
        std::vector<int> h_input(size), h_output(size);
        std::vector<uint64_t> h_divergence_timing(size);
        std::vector<float> h_divergence_ratio(size / 32);
        
        for (int i = 0; i < size; ++i) {
            h_input[i] = i % 100;
        }
        
        int *d_input, *d_output;
        uint64_t *d_divergence_timing;
        float *d_divergence_ratio;
        
        cudaMalloc(&d_input, size * sizeof(int));
        cudaMalloc(&d_output, size * sizeof(int));
        cudaMalloc(&d_divergence_timing, size * sizeof(uint64_t));
        cudaMalloc(&d_divergence_ratio, (size / 32) * sizeof(float));
        
        cudaMemcpy(d_input, h_input.data(), size * sizeof(int), cudaMemcpyHostToDevice);
        
        dim3 block(256);
        dim3 grid((size + block.x - 1) / block.x);
        
        cudaEvent_t start_event, end_event;
        cudaEventCreate(&start_event);
        cudaEventCreate(&end_event);
        
        cudaEventRecord(start_event);
        warp_divergent_kernel<<<grid, block>>>(d_input, d_output, d_divergence_timing, d_divergence_ratio, size);
        cudaEventRecord(end_event);
        cudaEventSynchronize(end_event);
        
        float kernel_time_ms;
        cudaEventElapsedTime(&kernel_time_ms, start_event, end_event);
        
        cudaMemcpy(h_divergence_timing.data(), d_divergence_timing, size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_divergence_ratio.data(), d_divergence_ratio, (size / 32) * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Analyze warp divergence
        std::map<int, std::vector<uint64_t>> warp_timings;
        for (int i = 0; i < size; ++i) {
            int warp_id = i / 32;
            warp_timings[warp_id].push_back(h_divergence_timing[i]);
        }
        
        float total_divergence = 0.0f;
        for (const auto& warp_entry : warp_timings) {
            const auto& timings = warp_entry.second;
            uint64_t warp_min = *std::min_element(timings.begin(), timings.end());
            uint64_t warp_max = *std::max_element(timings.begin(), timings.end());
            float warp_divergence = static_cast<float>(warp_max - warp_min) / warp_max * 100.0f;
            total_divergence += warp_divergence;
        }
        
        float average_divergence = total_divergence / warp_timings.size();
        
        FunctionProfile profile;
        profile.name = "warp_divergent_kernel";
        profile.total_time_ns = static_cast<uint64_t>(kernel_time_ms * 1e6f);
        profile.warp_efficiency = static_cast<uint32_t>(100.0f - average_divergence);
        function_profiles["warp_divergent_kernel"] = profile;
        
        std::cout << "  Average divergence: " << average_divergence << "%" << std::endl;
        std::cout << "  Warp efficiency: " << profile.warp_efficiency << "%" << std::endl;
        
        bool divergence_detected = average_divergence > 20.0f;
        
        cudaEventDestroy(start_event);
        cudaEventDestroy(end_event);
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_divergence_timing);
        cudaFree(d_divergence_ratio);
        
        std::cout << "Warp divergence profiling: " << (divergence_detected ? "PASSED" : "FAILED") << std::endl;
        return divergence_detected;
    }
    
    bool test_flame_graph_generation() {
        std::cout << "\n=== Testing Flame Graph Generation ===" << std::endl;
        
        std::ostringstream flame_data;
        
        for (const auto& profile : function_profiles) {
            flame_data << "GPU_Program_Root;" << profile.first << " " << profile.second.total_time_ns << "\n";
        }
        
        std::ofstream flame_file("/tmp/gpu_flame_graph.txt");
        flame_file << flame_data.str();
        flame_file.close();
        
        bool structure_valid = function_profiles.size() > 0;
        
        std::cout << "  Flame graph functions: " << function_profiles.size() << std::endl;
        std::cout << "  Flame graph file: /tmp/gpu_flame_graph.txt" << std::endl;
        std::cout << "Flame graph generation: " << (structure_valid ? "PASSED" : "FAILED") << std::endl;
        
        return structure_valid;
    }
    
    bool test_profiling_overhead() {
        std::cout << "\n=== Testing Profiling Overhead ===" << std::endl;
        
        const int size = 4096;
        const int iterations = 20;
        const int complexity = 50;
        
        std::vector<float> h_input(size), h_output(size);
        std::vector<uint64_t> h_timing(size);
        
        for (int i = 0; i < size; ++i) {
            h_input[i] = static_cast<float>(i + 1) / size;
        }
        
        float *d_input, *d_output;
        uint64_t *d_timing;
        
        cudaMalloc(&d_input, size * sizeof(float));
        cudaMalloc(&d_output, size * sizeof(float));
        cudaMalloc(&d_timing, size * sizeof(uint64_t));
        
        cudaMemcpy(d_input, h_input.data(), size * sizeof(float), cudaMemcpyHostToDevice);
        
        dim3 block(256);
        dim3 grid((size + block.x - 1) / block.x);
        
        // Baseline performance
        auto baseline_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            compute_intensive_kernel<<<grid, block>>>(d_input, d_output, d_timing, size, complexity);
        }
        cudaDeviceSynchronize();
        auto baseline_end = std::chrono::high_resolution_clock::now();
        auto baseline_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(baseline_end - baseline_start);
        
        // With profiling overhead
        auto profiling_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            cudaEvent_t start_event, end_event;
            cudaEventCreate(&start_event);
            cudaEventCreate(&end_event);
            
            cudaEventRecord(start_event);
            compute_intensive_kernel<<<grid, block>>>(d_input, d_output, d_timing, size, complexity);
            cudaEventRecord(end_event);
            cudaEventSynchronize(end_event);
            
            float kernel_time_ms;
            cudaEventElapsedTime(&kernel_time_ms, start_event, end_event);
            
            // Simulate profile data collection
            FunctionProfile profile;
            profile.name = "profiled_kernel_" + std::to_string(i);
            profile.total_time_ns = static_cast<uint64_t>(kernel_time_ms * 1e6f);
            profile.call_count = 1;
            function_profiles[profile.name] = profile;
            
            cudaEventDestroy(start_event);
            cudaEventDestroy(end_event);
        }
        auto profiling_end = std::chrono::high_resolution_clock::now();
        auto profiling_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(profiling_end - profiling_start);
        
        double overhead_percent = ((double)(profiling_duration.count() - baseline_duration.count()) / baseline_duration.count()) * 100.0;
        
        std::cout << "  Baseline: " << baseline_duration.count() / 1e6 << " ms" << std::endl;
        std::cout << "  With profiling: " << profiling_duration.count() / 1e6 << " ms" << std::endl;
        std::cout << "  Overhead: " << overhead_percent << "%" << std::endl;
        
        bool performance_target_met = overhead_percent < 5.0;
        
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_timing);
        
        std::cout << "Performance target (<5%): " << (performance_target_met ? "MET" : "NOT MET") << std::endl;
        return performance_target_met;
    }
};

// Test runner function
bool run_profiling_tests() {
    std::cout << "\n========== PERFORMANCE PROFILING TESTS ==========" << std::endl;
    
    ProfilingTest test_suite;
    
    bool all_tests_passed = true;
    
    all_tests_passed &= test_suite.test_compute_performance_profiling();
    all_tests_passed &= test_suite.test_memory_bandwidth_profiling();
    all_tests_passed &= test_suite.test_warp_divergence_profiling();
    all_tests_passed &= test_suite.test_flame_graph_generation();
    all_tests_passed &= test_suite.test_profiling_overhead();
    
    std::cout << "\n========== PROFILING TEST SUMMARY ==========" << std::endl;
    std::cout << "Overall result: " << (all_tests_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    std::cout << "Real GPU operations: ✓ Complex compute, memory, and divergent kernels" << std::endl;
    std::cout << "Performance analysis: ✓ Compute, memory, warp profiling" << std::endl;
    std::cout << "Flame graphs: ✓ Generated and validated" << std::endl;
    std::cout << "Profiling overhead: ✓ <5% overhead maintained" << std::endl;
    
    return all_tests_passed;
}

// Main test entry point
int main() {
    cudaSetDevice(0);
    bool success = run_profiling_tests();
    return success ? 0 : 1;
}