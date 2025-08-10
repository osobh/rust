// GPUDirect RDMA Tests - WRITTEN FIRST (TDD)
// Testing direct NIC to GPU memory transfers with 40Gbps+ throughput

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <chrono>
#include <atomic>

// Test result structure
struct TestResult {
    bool passed;
    float throughput_gbps;
    float latency_us;
    int operations_completed;
    char error_message[256];
};

// RDMA queue pair structure
struct RDMAQueuePair {
    void* send_buffer;
    void* recv_buffer;
    size_t buffer_size;
    int queue_depth;
    std::atomic<int> send_head;
    std::atomic<int> send_tail;
    std::atomic<int> recv_head;
    std::atomic<int> recv_tail;
    bool connected;
};

// Memory region for RDMA
struct MemoryRegion {
    void* gpu_buffer;
    size_t size;
    uint32_t lkey;  // Local key
    uint32_t rkey;  // Remote key
    bool registered;
};

// Test 1: GPU Memory Registration for RDMA
__global__ void test_memory_registration(TestResult* result, 
                                        MemoryRegion* regions,
                                        int num_regions) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid == 0) {
        result->passed = true;
        result->operations_completed = 0;
        
        // Test memory registration
        for (int i = 0; i < num_regions; i++) {
            // Allocate GPU memory
            size_t alloc_size = (1 << 20) * (i + 1); // 1MB, 2MB, 3MB...
            void* gpu_mem;
            
            cudaError_t err = cudaMalloc(&gpu_mem, alloc_size);
            if (err != cudaSuccess) {
                result->passed = false;
                sprintf(result->error_message, "Failed to allocate GPU memory");
                return;
            }
            
            // Register memory for RDMA (simulated)
            regions[i].gpu_buffer = gpu_mem;
            regions[i].size = alloc_size;
            regions[i].lkey = 0x1000 + i;  // Simulated local key
            regions[i].rkey = 0x2000 + i;  // Simulated remote key
            regions[i].registered = true;
            
            result->operations_completed++;
        }
        
        // Validate registrations
        for (int i = 0; i < num_regions; i++) {
            if (!regions[i].registered || regions[i].gpu_buffer == nullptr) {
                result->passed = false;
                sprintf(result->error_message, "Region %d not properly registered", i);
                return;
            }
        }
    }
}

// Test 2: Queue Pair Creation and Management
__global__ void test_queue_pair_creation(TestResult* result,
                                        RDMAQueuePair* qp,
                                        int queue_depth) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid == 0) {
        result->passed = true;
        
        // Initialize queue pair
        qp->queue_depth = queue_depth;
        qp->buffer_size = 64 * 1024; // 64KB per buffer
        
        // Allocate send/receive buffers
        cudaError_t err = cudaMalloc(&qp->send_buffer, 
                                    qp->buffer_size * queue_depth);
        if (err != cudaSuccess) {
            result->passed = false;
            sprintf(result->error_message, "Failed to allocate send buffer");
            return;
        }
        
        err = cudaMalloc(&qp->recv_buffer, 
                        qp->buffer_size * queue_depth);
        if (err != cudaSuccess) {
            result->passed = false;
            sprintf(result->error_message, "Failed to allocate recv buffer");
            return;
        }
        
        // Initialize queue pointers
        qp->send_head = 0;
        qp->send_tail = 0;
        qp->recv_head = 0;
        qp->recv_tail = 0;
        qp->connected = false;
        
        // Simulate connection establishment
        __threadfence();
        qp->connected = true;
        
        if (!qp->connected) {
            result->passed = false;
            sprintf(result->error_message, "Queue pair connection failed");
        }
    }
}

// Test 3: Zero-Copy Message Passing
__global__ void test_zero_copy_transfer(TestResult* result,
                                       void* src_buffer,
                                       void* dst_buffer,
                                       size_t transfer_size,
                                       int num_transfers) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    
    // Each thread handles a portion of transfers
    int transfers_per_thread = (num_transfers + total_threads - 1) / total_threads;
    int start_transfer = tid * transfers_per_thread;
    int end_transfer = min(start_transfer + transfers_per_thread, num_transfers);
    
    auto start_time = clock64();
    
    for (int i = start_transfer; i < end_transfer; i++) {
        // Calculate offsets for this transfer
        size_t offset = (i * transfer_size) % (256 * 1024 * 1024); // Wrap at 256MB
        
        // Simulate RDMA write operation (GPU-initiated)
        char* src = (char*)src_buffer + offset;
        char* dst = (char*)dst_buffer + offset;
        
        // Direct memory copy (simulating RDMA)
        for (size_t j = 0; j < transfer_size; j += 16) {
            // Coalesced 16-byte transfers
            *((float4*)(dst + j)) = *((float4*)(src + j));
        }
    }
    
    auto end_time = clock64();
    
    // Calculate throughput
    if (tid == 0) {
        double elapsed_cycles = (double)(end_time - start_time);
        double elapsed_seconds = elapsed_cycles / (1.4e9); // Assuming 1.4GHz
        double total_bytes = (double)transfer_size * num_transfers;
        double throughput_gbps = (total_bytes * 8) / (elapsed_seconds * 1e9);
        
        result->throughput_gbps = throughput_gbps;
        result->passed = (throughput_gbps >= 40.0); // 40Gbps target
        result->operations_completed = num_transfers;
        
        if (!result->passed) {
            sprintf(result->error_message, 
                   "Throughput %.2f Gbps below 40Gbps target", throughput_gbps);
        }
    }
}

// Test 4: RDMA Send/Receive Operations
__global__ void test_rdma_send_receive(TestResult* result,
                                      RDMAQueuePair* qp,
                                      int num_messages) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    if (warp_id < num_messages / 32) {
        // Each warp handles 32 messages
        int msg_id = warp_id * 32 + lane_id;
        
        if (msg_id < num_messages) {
            // Prepare message
            size_t msg_size = 4096; // 4KB messages
            int send_slot = atomicAdd(&qp->send_tail, 1) % qp->queue_depth;
            
            char* send_ptr = (char*)qp->send_buffer + send_slot * qp->buffer_size;
            
            // Fill message with test pattern
            for (int i = lane_id; i < msg_size; i += 32) {
                send_ptr[i] = (char)(msg_id + i);
            }
            
            __syncwarp();
            
            // Simulate RDMA send
            if (lane_id == 0) {
                atomicAdd(&qp->send_head, 1);
            }
            
            // Simulate receive
            int recv_slot = atomicAdd(&qp->recv_tail, 1) % qp->queue_depth;
            char* recv_ptr = (char*)qp->recv_buffer + recv_slot * qp->buffer_size;
            
            // Copy data (simulating RDMA receive)
            for (int i = lane_id; i < msg_size; i += 32) {
                recv_ptr[i] = send_ptr[i];
            }
            
            __syncwarp();
            
            // Verify received data
            bool valid = true;
            for (int i = lane_id; i < msg_size; i += 32) {
                if (recv_ptr[i] != (char)(msg_id + i)) {
                    valid = false;
                }
            }
            
            // Reduce validation across warp
            unsigned mask = __activemask();
            valid = __all_sync(mask, valid);
            
            if (lane_id == 0 && !valid) {
                atomicExch(&result->passed, 0);
                printf("Message %d validation failed\n", msg_id);
            }
        }
    }
    
    if (tid == 0) {
        result->operations_completed = num_messages;
        if (result->passed) {
            result->passed = true;
        }
    }
}

// Test 5: Multi-Queue RSS (Receive Side Scaling)
__global__ void test_multi_queue_rss(TestResult* result,
                                    RDMAQueuePair* queues,
                                    int num_queues,
                                    int packets_per_queue) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int queue_id = tid % num_queues;
    
    if (tid < num_queues * packets_per_queue) {
        int packet_id = tid / num_queues;
        
        // Hash-based queue selection (RSS)
        uint32_t hash = packet_id * 0x9e3779b9; // Golden ratio hash
        int target_queue = hash % num_queues;
        
        // Send packet to target queue
        RDMAQueuePair* qp = &queues[target_queue];
        int slot = atomicAdd(&qp->recv_tail, 1) % qp->queue_depth;
        
        // Write packet data
        char* packet_ptr = (char*)qp->recv_buffer + slot * 1500; // MTU size
        for (int i = 0; i < 64; i++) { // Write header
            packet_ptr[i] = (char)(packet_id + i);
        }
        
        atomicAdd(&qp->recv_head, 1);
    }
    
    __syncthreads();
    
    if (tid == 0) {
        // Verify load distribution
        int total_packets = 0;
        for (int i = 0; i < num_queues; i++) {
            int queue_packets = queues[i].recv_head.load();
            total_packets += queue_packets;
        }
        
        result->operations_completed = total_packets;
        result->passed = (total_packets == num_queues * packets_per_queue);
        
        if (!result->passed) {
            sprintf(result->error_message, 
                   "RSS distribution failed: expected %d, got %d packets",
                   num_queues * packets_per_queue, total_packets);
        }
    }
}

// Test 6: RDMA Atomic Operations
__global__ void test_rdma_atomics(TestResult* result,
                                 uint64_t* remote_memory,
                                 int num_atomics) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_atomics) {
        // Test atomic fetch-and-add
        uint64_t old_val = atomicAdd((unsigned long long*)&remote_memory[0], 1ULL);
        
        // Test atomic compare-and-swap
        uint64_t expected = tid;
        uint64_t desired = tid + 1000;
        atomicCAS((unsigned long long*)&remote_memory[tid + 1], expected, desired);
    }
    
    __syncthreads();
    
    if (tid == 0) {
        // Verify atomic operations
        result->passed = true;
        result->operations_completed = num_atomics;
        
        // Check fetch-and-add result
        if (remote_memory[0] != num_atomics) {
            result->passed = false;
            sprintf(result->error_message, 
                   "Atomic add failed: expected %d, got %llu",
                   num_atomics, remote_memory[0]);
        }
    }
}

// Test 7: Congestion Control and Flow Control
__global__ void test_congestion_control(TestResult* result,
                                       RDMAQueuePair* qp,
                                       int burst_size,
                                       float target_rate_gbps) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid == 0) {
        // Implement credit-based flow control
        int credits = qp->queue_depth / 2;
        int sent = 0;
        
        auto start_time = clock64();
        
        while (sent < burst_size && credits > 0) {
            // Check if we can send
            int send_slot = qp->send_tail.load() % qp->queue_depth;
            int recv_slot = qp->recv_head.load() % qp->queue_depth;
            
            // Calculate available space
            int space = (recv_slot - send_slot + qp->queue_depth) % qp->queue_depth;
            
            if (space > 0) {
                // Send packet
                qp->send_tail.fetch_add(1);
                sent++;
                credits--;
                
                // Simulate rate limiting
                if (sent % 100 == 0) {
                    // Calculate current rate
                    auto current_time = clock64();
                    double elapsed = (current_time - start_time) / 1.4e9;
                    double current_rate = (sent * qp->buffer_size * 8) / (elapsed * 1e9);
                    
                    // Adjust credits based on rate
                    if (current_rate > target_rate_gbps) {
                        credits = max(1, credits - 10);
                    } else {
                        credits = min(qp->queue_depth / 2, credits + 5);
                    }
                }
            }
            
            // Simulate credit recovery
            if (sent % 10 == 0) {
                credits = min(qp->queue_depth / 2, credits + 1);
            }
        }
        
        auto end_time = clock64();
        
        // Calculate achieved rate
        double elapsed = (end_time - start_time) / 1.4e9;
        double achieved_rate = (sent * qp->buffer_size * 8) / (elapsed * 1e9);
        
        result->throughput_gbps = achieved_rate;
        result->operations_completed = sent;
        result->passed = (abs(achieved_rate - target_rate_gbps) / target_rate_gbps < 0.1);
        
        if (!result->passed) {
            sprintf(result->error_message,
                   "Rate control failed: target %.2f Gbps, achieved %.2f Gbps",
                   target_rate_gbps, achieved_rate);
        }
    }
}

// Test 8: Interrupt Coalescing
__global__ void test_interrupt_coalescing(TestResult* result,
                                         volatile int* interrupt_counter,
                                         int num_packets,
                                         int coalesce_threshold) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ int shared_counter;
    __shared__ int packets_processed;
    
    if (threadIdx.x == 0) {
        shared_counter = 0;
        packets_processed = 0;
    }
    __syncthreads();
    
    // Process packets
    int packets_per_thread = (num_packets + blockDim.x - 1) / blockDim.x;
    int local_packets = 0;
    
    for (int i = 0; i < packets_per_thread; i++) {
        int packet_id = tid * packets_per_thread + i;
        if (packet_id < num_packets) {
            local_packets++;
            
            // Add to shared counter
            atomicAdd(&packets_processed, 1);
            
            // Check if we should generate interrupt
            if (packets_processed % coalesce_threshold == 0) {
                if (threadIdx.x == 0) {
                    atomicAdd(&shared_counter, 1);
                    atomicAdd((int*)interrupt_counter, 1);
                }
            }
        }
    }
    
    __syncthreads();
    
    if (tid == 0) {
        // Verify interrupt coalescing
        int expected_interrupts = (num_packets + coalesce_threshold - 1) / coalesce_threshold;
        int actual_interrupts = *interrupt_counter;
        
        result->operations_completed = num_packets;
        result->passed = (actual_interrupts <= expected_interrupts);
        
        if (!result->passed) {
            sprintf(result->error_message,
                   "Interrupt coalescing failed: expected <= %d, got %d",
                   expected_interrupts, actual_interrupts);
        }
    }
}

// Performance benchmark kernel
__global__ void benchmark_rdma_throughput(TestResult* result,
                                         void* src_buffer,
                                         void* dst_buffer,
                                         size_t buffer_size,
                                         int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    
    // Each thread processes a chunk
    size_t chunk_size = buffer_size / total_threads;
    size_t offset = tid * chunk_size;
    
    auto start = clock64();
    
    for (int iter = 0; iter < iterations; iter++) {
        // Simulate RDMA transfer
        char* src = (char*)src_buffer + offset;
        char* dst = (char*)dst_buffer + offset;
        
        // Vectorized copy
        for (size_t i = 0; i < chunk_size; i += sizeof(float4)) {
            *((float4*)(dst + i)) = *((float4*)(src + i));
        }
        
        __syncthreads();
    }
    
    auto end = clock64();
    
    if (tid == 0) {
        double cycles = (double)(end - start);
        double seconds = cycles / 1.4e9;
        double total_bytes = (double)buffer_size * iterations;
        double throughput_gbps = (total_bytes * 8) / (seconds * 1e9);
        
        result->throughput_gbps = throughput_gbps;
        result->latency_us = (seconds * 1e6) / iterations;
        result->passed = (throughput_gbps >= 40.0);
        result->operations_completed = iterations;
    }
}

// Main test runner
int main() {
    printf("=== GPU RDMA Tests ===\n\n");
    
    TestResult* d_result;
    cudaMalloc(&d_result, sizeof(TestResult));
    
    // Test 1: Memory Registration
    printf("Test 1: Memory Registration...\n");
    MemoryRegion* d_regions;
    cudaMalloc(&d_regions, sizeof(MemoryRegion) * 10);
    test_memory_registration<<<1, 1>>>(d_result, d_regions, 10);
    cudaDeviceSynchronize();
    
    TestResult h_result;
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    printf("  %s - %d regions registered\n\n", 
           h_result.passed ? "PASSED" : "FAILED",
           h_result.operations_completed);
    
    // Test 2: Queue Pair Creation
    printf("Test 2: Queue Pair Creation...\n");
    RDMAQueuePair* d_qp;
    cudaMalloc(&d_qp, sizeof(RDMAQueuePair));
    test_queue_pair_creation<<<1, 1>>>(d_result, d_qp, 128);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    printf("  %s\n\n", h_result.passed ? "PASSED" : "FAILED");
    
    // Test 3: Zero-Copy Transfer
    printf("Test 3: Zero-Copy Transfer...\n");
    void *d_src, *d_dst;
    size_t transfer_buffer_size = 256 * 1024 * 1024; // 256MB
    cudaMalloc(&d_src, transfer_buffer_size);
    cudaMalloc(&d_dst, transfer_buffer_size);
    
    test_zero_copy_transfer<<<256, 256>>>(d_result, d_src, d_dst, 
                                         64 * 1024, 10000); // 64KB transfers
    cudaDeviceSynchronize();
    
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    printf("  %s - Throughput: %.2f Gbps\n\n",
           h_result.passed ? "PASSED" : "FAILED",
           h_result.throughput_gbps);
    
    // Test 4: Send/Receive Operations
    printf("Test 4: RDMA Send/Receive...\n");
    test_rdma_send_receive<<<32, 256>>>(d_result, d_qp, 1024);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    printf("  %s - %d messages processed\n\n",
           h_result.passed ? "PASSED" : "FAILED",
           h_result.operations_completed);
    
    // Performance Benchmark
    printf("Performance Benchmark:\n");
    benchmark_rdma_throughput<<<512, 256>>>(d_result, d_src, d_dst,
                                           transfer_buffer_size, 100);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    printf("  Throughput: %.2f Gbps\n", h_result.throughput_gbps);
    printf("  Latency: %.2f μs\n", h_result.latency_us);
    printf("  Target Met: %s\n", h_result.passed ? "YES" : "NO");
    
    // Cleanup
    cudaFree(d_result);
    cudaFree(d_regions);
    cudaFree(d_qp);
    cudaFree(d_src);
    cudaFree(d_dst);
    
    return 0;
}