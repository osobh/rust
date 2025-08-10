// GPU Scheduler Tests - Written BEFORE Implementation  
// NO STUBS OR MOCKS - Real GPU Operations Only
// Target: <1μs scheduling decision, 95% SM utilization

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
    float scheduling_latency_us;
    float sm_utilization;
    int concurrent_tasks;
    int tasks_per_sec;
    char failure_msg[256];
};

// Task descriptor
struct Task {
    uint32_t task_id;
    uint32_t priority;
    uint32_t resource_requirements;
    uint32_t dependencies[4];
    uint32_t dep_count;
    void (*kernel_func)(void*);
    void* args;
    uint64_t enqueue_time;
    uint64_t start_time;
    uint64_t end_time;
    uint32_t status;  // 0=pending, 1=ready, 2=running, 3=complete
};

// Work queue structure
struct WorkQueue {
    Task* tasks;
    uint32_t* head;
    uint32_t* tail;
    uint32_t capacity;
    uint32_t* priority_index;
    uint32_t num_priorities;
};

// Persistent kernel state
struct PersistentKernelState {
    volatile int* should_exit;
    WorkQueue* work_queue;
    uint32_t* active_tasks;
    uint32_t max_concurrent;
    uint64_t* total_tasks_processed;
};

// Helper: Get microseconds
__device__ float get_microseconds(uint64_t cycles) {
    // Assuming 1GHz GPU clock
    return float(cycles) / 1000.0f;
}

// Test 1: Work queue operations (enqueue/dequeue)
__global__ void test_work_queue_operations(TestResult* result,
                                          WorkQueue* queue,
                                          Task* test_tasks,
                                          int num_tasks) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = tid / 32;
    
    if (tid == 0) {
        result->passed = true;
        result->total_tests = 0;
        result->failed_tests = 0;
    }
    __syncthreads();
    
    if (tid < num_tasks) {
        // Enqueue task with lock-free operation
        uint32_t old_tail, new_tail;
        uint64_t start = clock64();
        
        do {
            old_tail = *queue->tail;
            new_tail = (old_tail + 1) % queue->capacity;
            
            // Check if queue is full
            if (new_tail == *queue->head) {
                atomicAdd(&result->failed_tests, 1);
                return;
            }
        } while (atomicCAS(queue->tail, old_tail, new_tail) != old_tail);
        
        // Add task to queue
        queue->tasks[old_tail] = test_tasks[tid];
        queue->tasks[old_tail].enqueue_time = clock64();
        
        uint64_t end = clock64();
        
        // Calculate latency
        float latency = get_microseconds(end - start);
        atomicAdd(&result->scheduling_latency_us, latency);
        atomicAdd(&result->total_tests, 1);
        
        // Verify enqueue succeeded
        if (queue->tasks[old_tail].task_id != test_tasks[tid].task_id) {
            atomicAdd(&result->failed_tests, 1);
            result->passed = false;
        }
    }
    
    __syncthreads();
    
    // Now dequeue tasks
    if (tid < num_tasks) {
        uint32_t old_head, new_head;
        Task dequeued_task;
        
        uint64_t start = clock64();
        
        do {
            old_head = *queue->head;
            
            // Check if queue is empty
            if (old_head == *queue->tail) {
                break;
            }
            
            new_head = (old_head + 1) % queue->capacity;
            
            // Try to dequeue
            if (atomicCAS(queue->head, old_head, new_head) == old_head) {
                dequeued_task = queue->tasks[old_head];
                break;
            }
        } while (true);
        
        uint64_t end = clock64();
        float latency = get_microseconds(end - start);
        
        // Update average latency
        atomicAdd(&result->scheduling_latency_us, latency);
        
        // Verify dequeue
        if (dequeued_task.task_id == 0) {
            atomicAdd(&result->failed_tests, 1);
        }
    }
    
    __syncthreads();
    
    if (tid == 0) {
        result->scheduling_latency_us /= (num_tasks * 2);  // Average for enqueue+dequeue
        
        // Check performance target: <1μs
        if (result->scheduling_latency_us > 1.0f) {
            result->passed = false;
            sprintf(result->failure_msg, "Scheduling too slow: %.2fμs (target: <1μs)",
                   result->scheduling_latency_us);
        }
    }
}

// Test 2: Priority scheduling
__global__ void test_priority_scheduling(TestResult* result,
                                        WorkQueue* queue,
                                        Task* tasks,
                                        int num_tasks) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int priority_order[256];
    
    if (tid < num_tasks) {
        // Create tasks with different priorities
        tasks[tid].task_id = tid;
        tasks[tid].priority = (num_tasks - tid) % 4;  // 0-3 priority levels
        tasks[tid].status = 1;  // Ready
        
        // Insert into priority queue
        uint32_t priority = tasks[tid].priority;
        uint32_t slot = atomicAdd(&queue->priority_index[priority], 1);
        
        // Store in priority order
        priority_order[slot] = tid;
    }
    
    __syncthreads();
    
    // Verify priority ordering
    if (tid == 0) {
        int last_priority = -1;
        bool ordered = true;
        
        for (int i = 0; i < num_tasks; i++) {
            int task_idx = priority_order[i];
            int priority = tasks[task_idx].priority;
            
            if (priority < last_priority) {
                ordered = false;
                break;
            }
            last_priority = priority;
        }
        
        if (ordered) {
            result->passed = true;
            result->total_tests++;
        } else {
            result->passed = false;
            result->failed_tests++;
            strcpy(result->failure_msg, "Priority ordering violated");
        }
    }
}

// Test 3: Work stealing between warps
__global__ void test_work_stealing(TestResult* result,
                                  WorkQueue* local_queues,
                                  int num_warps,
                                  int tasks_per_warp) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    __shared__ int warp_task_count[32];
    
    if (lane_id == 0) {
        warp_task_count[warp_id] = 0;
    }
    __syncwarp();
    
    // Initially distribute tasks unevenly
    if (warp_id == 0 && lane_id < tasks_per_warp * 2) {
        // First warp gets double tasks
        atomicAdd(&warp_task_count[0], 1);
    } else if (warp_id > 0 && lane_id < tasks_per_warp / 2) {
        // Other warps get half tasks
        atomicAdd(&warp_task_count[warp_id], 1);
    }
    
    __syncthreads();
    
    // Work stealing phase
    if (lane_id == 0) {
        int my_count = warp_task_count[warp_id];
        
        // If underloaded, steal from overloaded warps
        if (my_count < tasks_per_warp) {
            for (int w = 0; w < num_warps; w++) {
                if (warp_task_count[w] > tasks_per_warp) {
                    // Attempt to steal
                    int steal_amount = min(warp_task_count[w] - tasks_per_warp,
                                          tasks_per_warp - my_count);
                    
                    int old_count = atomicAdd(&warp_task_count[w], -steal_amount);
                    if (old_count >= steal_amount) {
                        atomicAdd(&warp_task_count[warp_id], steal_amount);
                        break;
                    } else {
                        // Rollback if failed
                        atomicAdd(&warp_task_count[w], steal_amount);
                    }
                }
            }
        }
    }
    
    __syncthreads();
    
    // Check load balancing
    if (tid == 0) {
        int min_count = tasks_per_warp * num_warps;
        int max_count = 0;
        
        for (int w = 0; w < num_warps; w++) {
            min_count = min(min_count, warp_task_count[w]);
            max_count = max(max_count, warp_task_count[w]);
        }
        
        float imbalance = float(max_count - min_count) / tasks_per_warp;
        
        if (imbalance < 0.2f) {  // Less than 20% imbalance
            result->passed = true;
            result->total_tests++;
        } else {
            result->passed = false;
            result->failed_tests++;
            sprintf(result->failure_msg, "Load imbalance: %.1f%% (target: <20%%)",
                   imbalance * 100);
        }
    }
}

// Test 4: Persistent kernel execution
__global__ void test_persistent_kernel(PersistentKernelState* state,
                                      TestResult* result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = tid / 32;
    
    // Persistent kernel loop
    while (!(*state->should_exit)) {
        // Try to get work
        uint32_t old_head = *state->work_queue->head;
        uint32_t tail = *state->work_queue->tail;
        
        if (old_head != tail) {
            uint32_t new_head = (old_head + 1) % state->work_queue->capacity;
            
            // Try to claim task
            if (atomicCAS(state->work_queue->head, old_head, new_head) == old_head) {
                Task& task = state->work_queue->tasks[old_head];
                
                // Execute task
                task.start_time = clock64();
                
                // Simulate work
                float sum = 0;
                for (int i = 0; i < 1000; i++) {
                    sum += tid * i * 0.001f;
                }
                
                task.end_time = clock64();
                task.status = 3;  // Complete
                
                atomicAdd((unsigned long long*)state->total_tasks_processed, 1);
                atomicAdd(&state->active_tasks[warp_id], 1);
            }
        }
        
        // Cooperative yielding
        if (tid % 32 == 0) {
            __threadfence();
        }
    }
    
    // Report statistics
    if (tid == 0) {
        result->concurrent_tasks = 0;
        for (int i = 0; i < 32; i++) {
            result->concurrent_tasks += state->active_tasks[i];
        }
        
        result->passed = (*state->total_tasks_processed > 0);
        result->total_tests = 1;
        
        if (!result->passed) {
            strcpy(result->failure_msg, "No tasks processed by persistent kernel");
            result->failed_tests = 1;
        }
    }
}

// Test 5: Dependency resolution
__global__ void test_dependency_resolution(TestResult* result,
                                          Task* tasks,
                                          int num_tasks) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int ready_count;
    __shared__ int completed[256];
    
    if (threadIdx.x == 0) {
        ready_count = 0;
    }
    __syncthreads();
    
    // Mark initial tasks without dependencies as ready
    if (tid < num_tasks) {
        if (tasks[tid].dep_count == 0) {
            tasks[tid].status = 1;  // Ready
            atomicAdd(&ready_count, 1);
        } else {
            tasks[tid].status = 0;  // Pending
        }
        completed[tid] = 0;
    }
    
    __syncthreads();
    
    // Simulate execution with dependency resolution
    int iterations = 0;
    while (ready_count > 0 && iterations < 100) {
        if (tid < num_tasks && tasks[tid].status == 1) {
            // "Execute" task
            tasks[tid].status = 3;  // Complete
            completed[tid] = 1;
            atomicSub(&ready_count, 1);
            
            // Check if any tasks depend on this one
            for (int i = 0; i < num_tasks; i++) {
                if (tasks[i].status == 0) {  // Pending
                    bool deps_met = true;
                    for (int d = 0; d < tasks[i].dep_count; d++) {
                        if (!completed[tasks[i].dependencies[d]]) {
                            deps_met = false;
                            break;
                        }
                    }
                    
                    if (deps_met) {
                        tasks[i].status = 1;  // Ready
                        atomicAdd(&ready_count, 1);
                    }
                }
            }
        }
        
        __syncthreads();
        iterations++;
    }
    
    // Check all tasks completed
    if (tid == 0) {
        int complete_count = 0;
        for (int i = 0; i < num_tasks; i++) {
            if (tasks[i].status == 3) {
                complete_count++;
            }
        }
        
        if (complete_count == num_tasks) {
            result->passed = true;
            result->total_tests = 1;
        } else {
            result->passed = false;
            result->failed_tests = 1;
            sprintf(result->failure_msg, "Dependency resolution failed: %d/%d complete",
                   complete_count, num_tasks);
        }
    }
}

// Test 6: SM utilization measurement
__global__ void test_sm_utilization(TestResult* result,
                                   int workload_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ uint64_t warp_active_cycles[32];
    __shared__ uint64_t total_cycles;
    
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    if (lane_id == 0) {
        warp_active_cycles[warp_id] = 0;
    }
    __syncthreads();
    
    uint64_t start = clock64();
    
    // Simulate varying workload
    float work_result = 0;
    int my_work = workload_size * (1 + (tid % 4));  // Uneven work distribution
    
    uint64_t work_start = clock64();
    for (int i = 0; i < my_work; i++) {
        work_result += __sinf(tid * i * 0.001f);
    }
    uint64_t work_end = clock64();
    
    // Track active cycles per warp
    if (lane_id == 0) {
        warp_active_cycles[warp_id] = work_end - work_start;
    }
    
    __syncthreads();
    
    uint64_t end = clock64();
    
    if (tid == 0) {
        total_cycles = end - start;
        
        // Calculate SM utilization
        uint64_t total_active = 0;
        for (int w = 0; w < blockDim.x / 32; w++) {
            total_active += warp_active_cycles[w];
        }
        
        float utilization = float(total_active) / (total_cycles * (blockDim.x / 32));
        result->sm_utilization = utilization;
        
        // Target: >95% utilization
        if (utilization > 0.95f) {
            result->passed = true;
            result->total_tests = 1;
        } else {
            result->passed = false;
            result->failed_tests = 1;
            sprintf(result->failure_msg, "Low SM utilization: %.1f%% (target: >95%%)",
                   utilization * 100);
        }
    }
}

// Test 7: Throughput benchmark
__global__ void test_scheduler_throughput(TestResult* result,
                                         WorkQueue* queue,
                                         int num_tasks,
                                         int iterations) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ uint64_t start_time;
    __shared__ uint64_t end_time;
    __shared__ int tasks_processed;
    
    if (threadIdx.x == 0) {
        tasks_processed = 0;
        start_time = clock64();
    }
    __syncthreads();
    
    // High-throughput scheduling test
    for (int iter = 0; iter < iterations; iter++) {
        // Enqueue
        uint32_t old_tail = atomicAdd(queue->tail, 1) % queue->capacity;
        queue->tasks[old_tail].task_id = tid * iterations + iter;
        queue->tasks[old_tail].priority = tid % 4;
        queue->tasks[old_tail].status = 1;
        
        // Immediately dequeue
        uint32_t old_head = atomicAdd(queue->head, 1) % queue->capacity;
        if (old_head < *queue->tail) {
            Task task = queue->tasks[old_head];
            
            // Minimal processing
            int dummy = task.task_id * task.priority;
            
            atomicAdd(&tasks_processed, 1);
        }
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        end_time = clock64();
        
        float elapsed_us = get_microseconds(end_time - start_time);
        float elapsed_sec = elapsed_us / 1e6f;
        
        result->tasks_per_sec = int(tasks_processed / elapsed_sec);
        
        // Target: 10K tasks/sec minimum
        if (result->tasks_per_sec >= 10000) {
            result->passed = true;
            result->total_tests = 1;
        } else {
            result->passed = false;
            result->failed_tests = 1;
            sprintf(result->failure_msg, "Low throughput: %d tasks/sec (target: >10K)",
                   result->tasks_per_sec);
        }
    }
}

// Initialize work queue
__global__ void init_work_queue(WorkQueue* queue, Task* task_buffer, 
                               uint32_t capacity) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        queue->tasks = task_buffer;
        queue->capacity = capacity;
        queue->head = (uint32_t*)malloc(sizeof(uint32_t));
        queue->tail = (uint32_t*)malloc(sizeof(uint32_t));
        queue->priority_index = (uint32_t*)malloc(4 * sizeof(uint32_t));
        *queue->head = 0;
        *queue->tail = 0;
        queue->num_priorities = 4;
        
        for (int i = 0; i < 4; i++) {
            queue->priority_index[i] = 0;
        }
    }
}

// Main test runner
int main() {
    printf("GPU Scheduler Tests - NO STUBS OR MOCKS\n");
    printf("Target: <1μs scheduling, 95%% SM utilization\n");
    printf("==========================================\n\n");
    
    // Allocate test resources
    TestResult* d_result;
    cudaMalloc(&d_result, sizeof(TestResult));
    
    const int MAX_TASKS = 10000;
    Task* d_tasks;
    cudaMalloc(&d_tasks, MAX_TASKS * sizeof(Task));
    
    WorkQueue* d_queue;
    cudaMalloc(&d_queue, sizeof(WorkQueue));
    
    Task* d_task_buffer;
    cudaMalloc(&d_task_buffer, MAX_TASKS * sizeof(Task));
    
    // Initialize queue
    init_work_queue<<<1, 1>>>(d_queue, d_task_buffer, MAX_TASKS);
    cudaDeviceSynchronize();
    
    TestResult h_result;
    int total_tests = 0;
    int passed_tests = 0;
    
    // Test 1: Work queue operations
    printf("Test 1: Lock-free work queue operations...");
    test_work_queue_operations<<<32, 256>>>(d_result, d_queue, d_tasks, 256);
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    total_tests++;
    if (h_result.passed) {
        printf(" PASSED (%.2fμs latency)\n", h_result.scheduling_latency_us);
        passed_tests++;
    } else {
        printf(" FAILED: %s\n", h_result.failure_msg);
    }
    
    // Test 2: Priority scheduling
    printf("Test 2: Priority-based task scheduling...");
    test_priority_scheduling<<<1, 256>>>(d_result, d_queue, d_tasks, 256);
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    total_tests++;
    if (h_result.passed) {
        printf(" PASSED\n");
        passed_tests++;
    } else {
        printf(" FAILED: %s\n", h_result.failure_msg);
    }
    
    // Test 3: Work stealing
    printf("Test 3: Work stealing between warps...");
    WorkQueue* d_local_queues;
    cudaMalloc(&d_local_queues, 32 * sizeof(WorkQueue));
    
    test_work_stealing<<<1, 256>>>(d_result, d_local_queues, 8, 32);
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    cudaFree(d_local_queues);
    total_tests++;
    if (h_result.passed) {
        printf(" PASSED\n");
        passed_tests++;
    } else {
        printf(" FAILED: %s\n", h_result.failure_msg);
    }
    
    // Test 4: Persistent kernel
    printf("Test 4: Persistent kernel execution...");
    PersistentKernelState* d_state;
    cudaMalloc(&d_state, sizeof(PersistentKernelState));
    
    int* d_should_exit;
    cudaMalloc(&d_should_exit, sizeof(int));
    cudaMemset(d_should_exit, 0, sizeof(int));
    
    uint32_t* d_active_tasks;
    cudaMalloc(&d_active_tasks, 32 * sizeof(uint32_t));
    cudaMemset(d_active_tasks, 0, 32 * sizeof(uint32_t));
    
    uint64_t* d_total_processed;
    cudaMalloc(&d_total_processed, sizeof(uint64_t));
    cudaMemset(d_total_processed, 0, sizeof(uint64_t));
    
    // Setup persistent kernel state
    PersistentKernelState h_state;
    h_state.should_exit = d_should_exit;
    h_state.work_queue = d_queue;
    h_state.active_tasks = d_active_tasks;
    h_state.max_concurrent = 1024;
    h_state.total_tasks_processed = d_total_processed;
    cudaMemcpy(d_state, &h_state, sizeof(PersistentKernelState), cudaMemcpyHostToDevice);
    
    // Add some tasks
    for (int i = 0; i < 100; i++) {
        Task h_task;
        h_task.task_id = i;
        h_task.priority = i % 4;
        h_task.status = 1;
        cudaMemcpy(&d_task_buffer[i], &h_task, sizeof(Task), cudaMemcpyHostToDevice);
    }
    
    // Launch persistent kernel
    test_persistent_kernel<<<32, 256>>>(d_state, d_result);
    
    // Let it run briefly
    cudaDeviceSynchronize();
    
    // Signal exit
    int exit_flag = 1;
    cudaMemcpy(d_should_exit, &exit_flag, sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    cudaFree(d_state);
    cudaFree(d_should_exit);
    cudaFree(d_active_tasks);
    cudaFree(d_total_processed);
    total_tests++;
    if (h_result.passed) {
        printf(" PASSED (%d concurrent tasks)\n", h_result.concurrent_tasks);
        passed_tests++;
    } else {
        printf(" FAILED: %s\n", h_result.failure_msg);
    }
    
    // Test 5: Dependency resolution
    printf("Test 5: Task dependency resolution...");
    
    // Create tasks with dependencies
    Task h_tasks[100];
    for (int i = 0; i < 100; i++) {
        h_tasks[i].task_id = i;
        h_tasks[i].priority = i % 4;
        h_tasks[i].dep_count = 0;
        
        // Add some dependencies
        if (i > 0) {
            h_tasks[i].dependencies[0] = i - 1;
            h_tasks[i].dep_count = 1;
        }
        if (i > 10) {
            h_tasks[i].dependencies[1] = i - 10;
            h_tasks[i].dep_count = 2;
        }
    }
    cudaMemcpy(d_tasks, h_tasks, 100 * sizeof(Task), cudaMemcpyHostToDevice);
    
    test_dependency_resolution<<<1, 256>>>(d_result, d_tasks, 100);
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    total_tests++;
    if (h_result.passed) {
        printf(" PASSED\n");
        passed_tests++;
    } else {
        printf(" FAILED: %s\n", h_result.failure_msg);
    }
    
    // Test 6: SM utilization
    printf("Test 6: SM utilization measurement...");
    test_sm_utilization<<<32, 256>>>(d_result, 10000);
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    total_tests++;
    if (h_result.passed) {
        printf(" PASSED (%.1f%% utilization)\n", h_result.sm_utilization * 100);
        passed_tests++;
    } else {
        printf(" FAILED: %s\n", h_result.failure_msg);
    }
    
    // Test 7: Throughput
    printf("Test 7: Scheduler throughput benchmark...");
    test_scheduler_throughput<<<32, 256>>>(d_result, d_queue, 1000, 100);
    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    total_tests++;
    if (h_result.passed) {
        printf(" PASSED (%d tasks/sec)\n", h_result.tasks_per_sec);
        passed_tests++;
    } else {
        printf(" FAILED: %s\n", h_result.failure_msg);
    }
    
    // Cleanup
    cudaFree(d_result);
    cudaFree(d_tasks);
    cudaFree(d_queue);
    cudaFree(d_task_buffer);
    
    // Summary
    printf("\n==========================================\n");
    printf("Scheduler Test Results: %d/%d passed\n", passed_tests, total_tests);
    
    if (passed_tests == total_tests) {
        printf("✓ All scheduler tests passed!\n");
        printf("✓ <1μs scheduling latency achieved\n");
        printf("✓ 95%% SM utilization validated\n");
        return 0;
    } else {
        printf("✗ Some tests failed\n");
        return 1;
    }
}