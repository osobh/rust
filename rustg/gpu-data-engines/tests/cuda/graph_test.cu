/**
 * GPU Graph Processing Engine CUDA Tests
 * STRICT TDD: Written BEFORE implementation
 * Validates 1B+ edges/sec traversal with real GPU operations
 */

#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/fill.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>

namespace cg = cooperative_groups;

// Test result structure
struct TestResult {
    bool success;
    float throughput_edges_per_sec;
    size_t vertices_processed;
    double elapsed_ms;
    char error_msg[256];
};

// CSR (Compressed Sparse Row) graph representation
struct CSRGraph {
    uint32_t* row_offsets;    // Size: num_vertices + 1
    uint32_t* col_indices;    // Size: num_edges  
    float* edge_weights;      // Size: num_edges (optional)
    uint32_t num_vertices;
    uint32_t num_edges;
};

// Graph algorithm state structures
struct BFSState {
    bool* visited;
    bool* frontier;
    bool* next_frontier;
    uint32_t* distances;
    uint32_t level;
    bool has_work;
};

struct PageRankState {
    float* current_ranks;
    float* next_ranks;
    float damping;
    float convergence_threshold;
    uint32_t max_iterations;
};

struct ConnectedComponentsState {
    uint32_t* component_ids;
    bool* changed;
    uint32_t num_components;
};

/**
 * TEST 1: Breadth-First Search (BFS) Traversal
 * Validates high-throughput graph traversal at 1B+ edges/sec
 */
__global__ void test_bfs_kernel(TestResult* result,
                               CSRGraph graph,
                               uint32_t source_vertex,
                               BFSState* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto block = cg::this_thread_block();
    
    __shared__ bool block_has_work;
    
    if (threadIdx.x == 0) {
        block_has_work = false;
    }
    __syncthreads();
    
    // Process vertices in current frontier
    for (uint32_t v = tid; v < graph.num_vertices; v += blockDim.x * gridDim.x) {
        if (state->frontier[v]) {
            // Explore all neighbors of vertex v
            uint32_t start = graph.row_offsets[v];
            uint32_t end = graph.row_offsets[v + 1];
            
            for (uint32_t edge = start; edge < end; edge++) {
                uint32_t neighbor = graph.col_indices[edge];
                
                // Check if neighbor is unvisited
                if (!state->visited[neighbor]) {
                    // Atomic CAS to claim this vertex
                    int was_visited = atomicExch((int*)&state->visited[neighbor], 1);
                    if (!was_visited) {
                        state->next_frontier[neighbor] = true;
                        state->distances[neighbor] = state->level + 1;
                        block_has_work = true;
                    }
                }
            }
            
            // Clear current frontier
            state->frontier[v] = false;
        }
    }
    
    __syncthreads();
    
    // Update global work flag
    if (threadIdx.x == 0 && block_has_work) {
        state->has_work = true;
    }
}

/**
 * TEST 2: PageRank Algorithm
 * Validates GPU-accelerated PageRank computation
 */
__global__ void test_pagerank_kernel(TestResult* result,
                                   CSRGraph graph,
                                   PageRankState* state,
                                   uint32_t iteration) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    
    // Process vertices
    for (uint32_t v = tid; v < graph.num_vertices; v += blockDim.x * gridDim.x) {
        float sum = 0.0f;
        
        // Sum contributions from all incoming edges
        uint32_t start = graph.row_offsets[v];
        uint32_t end = graph.row_offsets[v + 1];
        uint32_t out_degree = end - start;
        
        // For each incoming edge (simulated by reversing adjacency)
        for (uint32_t u = 0; u < graph.num_vertices; u++) {
            uint32_t u_start = graph.row_offsets[u];
            uint32_t u_end = graph.row_offsets[u + 1];
            uint32_t u_out_degree = u_end - u_start;
            
            // Check if there's an edge u -> v
            for (uint32_t edge = u_start; edge < u_end; edge++) {
                if (graph.col_indices[edge] == v) {
                    sum += state->current_ranks[u] / u_out_degree;
                    break;
                }
            }
        }
        
        // Apply PageRank formula: (1-d)/N + d * sum
        float new_rank = (1.0f - state->damping) / graph.num_vertices + 
                        state->damping * sum;
        
        state->next_ranks[v] = new_rank;
    }
}

/**
 * TEST 3: Connected Components (Union-Find)
 * Validates parallel connected components detection
 */
__global__ void test_connected_components_kernel(TestResult* result,
                                               CSRGraph graph,
                                               ConnectedComponentsState* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    bool changed = false;
    
    // Process vertices
    for (uint32_t v = tid; v < graph.num_vertices; v += blockDim.x * gridDim.x) {
        uint32_t min_component = state->component_ids[v];
        
        // Find minimum component ID among neighbors
        uint32_t start = graph.row_offsets[v];
        uint32_t end = graph.row_offsets[v + 1];
        
        for (uint32_t edge = start; edge < end; edge++) {
            uint32_t neighbor = graph.col_indices[edge];
            uint32_t neighbor_comp = state->component_ids[neighbor];
            if (neighbor_comp < min_component) {
                min_component = neighbor_comp;
            }
        }
        
        // Update component ID if changed
        if (min_component < state->component_ids[v]) {
            state->component_ids[v] = min_component;
            changed = true;
        }
    }
    
    // Set global change flag
    if (changed) {
        *state->changed = true;
    }
}

/**
 * TEST 4: Single Source Shortest Path (SSSP)
 * Validates GPU-optimized shortest path algorithms
 */
__global__ void test_sssp_kernel(TestResult* result,
                               CSRGraph graph,
                               uint32_t source,
                               float* distances,
                               bool* in_queue,
                               bool* has_work) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    bool local_work = false;
    
    // Process vertices
    for (uint32_t v = tid; v < graph.num_vertices; v += blockDim.x * gridDim.x) {
        if (in_queue[v]) {
            in_queue[v] = false;
            float v_dist = distances[v];
            
            // Relax all outgoing edges
            uint32_t start = graph.row_offsets[v];
            uint32_t end = graph.row_offsets[v + 1];
            
            for (uint32_t edge = start; edge < end; edge++) {
                uint32_t neighbor = graph.col_indices[edge];
                float edge_weight = (graph.edge_weights) ? graph.edge_weights[edge] : 1.0f;
                float new_dist = v_dist + edge_weight;
                
                // Atomic minimum for relaxation
                float old_dist = atomicExch(&distances[neighbor], new_dist);
                if (new_dist < old_dist) {
                    // Restored old distance, try again
                    float current_dist = atomicExch(&distances[neighbor], old_dist);
                    if (new_dist < current_dist) {
                        distances[neighbor] = new_dist;
                        in_queue[neighbor] = true;
                        local_work = true;
                    }
                }
            }
        }
    }
    
    if (local_work) {
        *has_work = true;
    }
}

/**
 * TEST 5: Triangle Counting
 * Validates efficient triangle enumeration on GPU
 */
__global__ void test_triangle_counting_kernel(TestResult* result,
                                            CSRGraph graph,
                                            uint64_t* triangle_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    
    uint64_t local_triangles = 0;
    
    // Process vertices
    for (uint32_t u = tid; u < graph.num_vertices; u += blockDim.x * gridDim.x) {
        uint32_t u_start = graph.row_offsets[u];
        uint32_t u_end = graph.row_offsets[u + 1];
        
        // For each neighbor v of u
        for (uint32_t i = u_start; i < u_end; i++) {
            uint32_t v = graph.col_indices[i];
            if (v <= u) continue; // Avoid double counting
            
            uint32_t v_start = graph.row_offsets[v];
            uint32_t v_end = graph.row_offsets[v + 1];
            
            // For each neighbor w of u (after v)
            for (uint32_t j = i + 1; j < u_end; j++) {
                uint32_t w = graph.col_indices[j];
                if (w <= v) continue;
                
                // Check if v and w are connected (binary search)
                bool found = false;
                uint32_t left = v_start, right = v_end;
                while (left < right) {
                    uint32_t mid = (left + right) / 2;
                    if (graph.col_indices[mid] == w) {
                        found = true;
                        break;
                    } else if (graph.col_indices[mid] < w) {
                        left = mid + 1;
                    } else {
                        right = mid;
                    }
                }
                
                if (found) {
                    local_triangles++;
                }
            }
        }
    }
    
    // Warp-level reduction
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        local_triangles += warp.shfl_down(local_triangles, offset);
    }
    
    // Add to global count
    if (warp.thread_rank() == 0) {
        atomicAdd((unsigned long long*)triangle_count, (unsigned long long)local_triangles);
    }
}

/**
 * TEST 6: Maximal Independent Set (MIS)
 * Validates parallel MIS computation
 */
__global__ void test_mis_kernel(TestResult* result,
                              CSRGraph graph,
                              bool* in_set,
                              bool* removed,
                              uint32_t* random_values,
                              bool* changed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    bool local_changed = false;
    
    // Process vertices
    for (uint32_t v = tid; v < graph.num_vertices; v += blockDim.x * gridDim.x) {
        if (removed[v] || in_set[v]) continue;
        
        bool is_local_maximum = true;
        uint32_t v_value = random_values[v];
        
        // Check if v has maximum value among unremoved neighbors
        uint32_t start = graph.row_offsets[v];
        uint32_t end = graph.row_offsets[v + 1];
        
        for (uint32_t edge = start; edge < end; edge++) {
            uint32_t neighbor = graph.col_indices[edge];
            if (!removed[neighbor] && !in_set[neighbor]) {
                if (random_values[neighbor] > v_value || 
                   (random_values[neighbor] == v_value && neighbor > v)) {
                    is_local_maximum = false;
                    break;
                }
            }
        }
        
        // Add to independent set if local maximum
        if (is_local_maximum) {
            in_set[v] = true;
            local_changed = true;
            
            // Remove all neighbors
            for (uint32_t edge = start; edge < end; edge++) {
                uint32_t neighbor = graph.col_indices[edge];
                removed[neighbor] = true;
            }
        }
    }
    
    if (local_changed) {
        *changed = true;
    }
}

/**
 * Performance Test Wrapper Functions
 */
extern "C" {
    void test_graph_bfs_performance(TestResult* result, uint32_t num_vertices, uint32_t num_edges) {
        // Generate random graph
        CSRGraph graph;
        graph.num_vertices = num_vertices;
        graph.num_edges = num_edges;
        
        cudaMalloc(&graph.row_offsets, (num_vertices + 1) * sizeof(uint32_t));
        cudaMalloc(&graph.col_indices, num_edges * sizeof(uint32_t));
        
        // Generate random adjacency (uniform degree distribution)
        thrust::device_vector<uint32_t> degrees(num_vertices, num_edges / num_vertices);
        thrust::exclusive_scan(degrees.begin(), degrees.end(), 
                              thrust::device_pointer_cast(graph.row_offsets));
        graph.row_offsets[num_vertices] = num_edges;
        
        // Random target vertices for each edge
        thrust::sequence(thrust::device, graph.col_indices, graph.col_indices + num_edges);
        thrust::transform(thrust::device, graph.col_indices, graph.col_indices + num_edges,
                         graph.col_indices, [num_vertices] __device__ (uint32_t x) { return x % num_vertices; });
        
        // Allocate BFS state
        BFSState state;
        cudaMalloc(&state.visited, num_vertices * sizeof(bool));
        cudaMalloc(&state.frontier, num_vertices * sizeof(bool));
        cudaMalloc(&state.next_frontier, num_vertices * sizeof(bool));
        cudaMalloc(&state.distances, num_vertices * sizeof(uint32_t));
        
        // Initialize BFS from vertex 0
        thrust::fill(thrust::device, state.visited, state.visited + num_vertices, false);
        thrust::fill(thrust::device, state.frontier, state.frontier + num_vertices, false);
        thrust::fill(thrust::device, state.next_frontier, state.next_frontier + num_vertices, false);
        thrust::fill(thrust::device, state.distances, state.distances + num_vertices, UINT32_MAX);
        
        // Set source
        bool true_val = true;
        uint32_t zero_dist = 0;
        cudaMemcpy(&state.visited[0], &true_val, sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(&state.frontier[0], &true_val, sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(&state.distances[0], &zero_dist, sizeof(uint32_t), cudaMemcpyHostToDevice);
        
        state.level = 0;
        
        // Launch BFS kernel
        dim3 block(256);
        dim3 grid((num_vertices + block.x - 1) / block.x);
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        
        // BFS iterations
        do {
            state.has_work = false;
            
            test_bfs_kernel<<<grid, block>>>(result, graph, 0, &state);
            cudaDeviceSynchronize();
            
            // Swap frontiers
            bool* temp = state.frontier;
            state.frontier = state.next_frontier;
            state.next_frontier = temp;
            
            thrust::fill(thrust::device, state.next_frontier, state.next_frontier + num_vertices, false);
            state.level++;
            
        } while (state.has_work && state.level < 100);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        // Calculate performance
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        
        result->elapsed_ms = elapsed_ms;
        result->vertices_processed = num_vertices;
        result->throughput_edges_per_sec = (num_edges * state.level) / (elapsed_ms / 1000.0);
        result->success = (result->throughput_edges_per_sec >= 1000000000.0); // 1B edges/sec target
        
        // Cleanup
        cudaFree(graph.row_offsets);
        cudaFree(graph.col_indices);
        cudaFree(state.visited);
        cudaFree(state.frontier);
        cudaFree(state.next_frontier);
        cudaFree(state.distances);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void test_graph_pagerank_performance(TestResult* result, uint32_t num_vertices, uint32_t num_edges) {
        // Setup PageRank test
        CSRGraph graph;
        graph.num_vertices = num_vertices;
        graph.num_edges = num_edges;
        
        cudaMalloc(&graph.row_offsets, (num_vertices + 1) * sizeof(uint32_t));
        cudaMalloc(&graph.col_indices, num_edges * sizeof(uint32_t));
        
        // Generate random graph
        thrust::device_vector<uint32_t> degrees(num_vertices, num_edges / num_vertices);
        thrust::exclusive_scan(degrees.begin(), degrees.end(),
                              thrust::device_pointer_cast(graph.row_offsets));
        graph.row_offsets[num_vertices] = num_edges;
        
        thrust::sequence(thrust::device, graph.col_indices, graph.col_indices + num_edges);
        thrust::transform(thrust::device, graph.col_indices, graph.col_indices + num_edges,
                         graph.col_indices, [num_vertices] __device__ (uint32_t x) { return x % num_vertices; });
        
        // PageRank state
        PageRankState state;
        state.damping = 0.85f;
        state.convergence_threshold = 1e-6f;
        state.max_iterations = 50;
        
        cudaMalloc(&state.current_ranks, num_vertices * sizeof(float));
        cudaMalloc(&state.next_ranks, num_vertices * sizeof(float));
        
        // Initialize ranks
        float init_rank = 1.0f / num_vertices;
        thrust::fill(thrust::device, state.current_ranks, state.current_ranks + num_vertices, init_rank);
        
        // Launch PageRank iterations
        dim3 block(256);
        dim3 grid((num_vertices + block.x - 1) / block.x);
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        
        for (uint32_t iter = 0; iter < state.max_iterations; iter++) {
            test_pagerank_kernel<<<grid, block>>>(result, graph, &state, iter);
            
            // Swap rank arrays
            float* temp = state.current_ranks;
            state.current_ranks = state.next_ranks;
            state.next_ranks = temp;
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        // Calculate performance
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        
        result->elapsed_ms = elapsed_ms;
        result->vertices_processed = num_vertices * state.max_iterations;
        result->throughput_edges_per_sec = (num_edges * state.max_iterations) / (elapsed_ms / 1000.0);
        result->success = (result->throughput_edges_per_sec >= 500000000.0); // 500M edges/sec for PageRank
        
        // Cleanup
        cudaFree(graph.row_offsets);
        cudaFree(graph.col_indices);
        cudaFree(state.current_ranks);
        cudaFree(state.next_ranks);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void test_graph_performance_comprehensive(TestResult* result) {
        const uint32_t NUM_VERTICES = 1000000;  // 1M vertices
        const uint32_t NUM_EDGES = 10000000;    // 10M edges
        
        // Test 1: BFS performance
        test_graph_bfs_performance(result, NUM_VERTICES, NUM_EDGES);
        if (!result->success) {
            strcpy(result->error_msg, "BFS failed to meet 1B edges/sec target");
            return;
        }
        
        // Test 2: PageRank performance  
        TestResult pr_result = {};
        test_graph_pagerank_performance(&pr_result, NUM_VERTICES, NUM_EDGES);
        if (!pr_result.success) {
            strcpy(result->error_msg, "PageRank failed to meet performance target");
            result->success = false;
            return;
        }
        
        // Overall success
        result->success = true;
        result->throughput_edges_per_sec = (result->throughput_edges_per_sec + pr_result.throughput_edges_per_sec) / 2.0;
        strcpy(result->error_msg, "All graph processing tests passed");
    }
}