#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include "../../../include/gpu_types.h"

namespace cg = cooperative_groups;

namespace rustg {

// Shared memory for BFS
struct BFSSharedMem {
    uint32_t frontier[256];       // Current frontier nodes
    uint32_t next_frontier[256];  // Next level frontier
    uint32_t frontier_size;
    uint32_t next_frontier_size;
    bool done;
};

// Parallel BFS kernel using frontier-based approach
__global__ void parallel_bfs_kernel(
    const uint32_t* csr_row_offsets,
    const uint32_t* csr_col_indices,
    uint32_t num_nodes,
    uint32_t start_node,
    int32_t* distances,
    uint32_t* predecessors,
    bool* visited
) {
    extern __shared__ char shared_mem_raw[];
    BFSSharedMem* shared = reinterpret_cast<BFSSharedMem*>(shared_mem_raw);
    
    const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // Initialize
    if (tid == 0) {
        shared->frontier[0] = start_node;
        shared->frontier_size = 1;
        shared->next_frontier_size = 0;
        shared->done = false;
        
        distances[start_node] = 0;
        predecessors[start_node] = start_node;
        visited[start_node] = true;
    }
    
    // Initialize distances
    for (uint32_t i = tid; i < num_nodes; i += blockDim.x) {
        if (i != start_node) {
            distances[i] = -1;
            predecessors[i] = UINT32_MAX;
            visited[i] = false;
        }
    }
    __syncthreads();
    
    int32_t level = 0;
    
    // BFS main loop
    while (!shared->done) {
        if (tid == 0) {
            shared->next_frontier_size = 0;
        }
        __syncthreads();
        
        // Process current frontier in parallel
        for (uint32_t f = 0; f < shared->frontier_size; f += blockDim.x) {
            uint32_t frontier_idx = f + tid;
            
            if (frontier_idx < shared->frontier_size) {
                uint32_t node = shared->frontier[frontier_idx];
                
                // Get neighbors
                uint32_t row_start = csr_row_offsets[node];
                uint32_t row_end = csr_row_offsets[node + 1];
                
                // Process neighbors
                for (uint32_t n = row_start; n < row_end; ++n) {
                    uint32_t neighbor = csr_col_indices[n];
                    
                    // Atomic check and update
                    bool old_visited = atomicExch((int*)&visited[neighbor], 1);
                    
                    if (!old_visited) {
                        distances[neighbor] = level + 1;
                        predecessors[neighbor] = node;
                        
                        // Add to next frontier
                        uint32_t pos = atomicAdd(&shared->next_frontier_size, 1);
                        if (pos < 256) {
                            shared->next_frontier[pos] = neighbor;
                        }
                    }
                }
            }
        }
        __syncthreads();
        
        // Swap frontiers
        if (tid == 0) {
            shared->frontier_size = shared->next_frontier_size;
            shared->done = (shared->frontier_size == 0);
        }
        __syncthreads();
        
        // Copy next frontier to current
        for (uint32_t i = tid; i < shared->frontier_size; i += blockDim.x) {
            shared->frontier[i] = shared->next_frontier[i];
        }
        __syncthreads();
        
        level++;
    }
}

// Optimized BFS using warp-centric approach
__global__ void bfs_warp_centric_kernel(
    const uint32_t* csr_row_offsets,
    const uint32_t* csr_col_indices,
    uint32_t num_nodes,
    uint32_t start_node,
    int32_t* distances,
    uint32_t* predecessors
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    const uint32_t num_warps = gridDim.x * blockDim.x / 32;
    
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
    
    __shared__ uint32_t frontier_shared[1024];
    __shared__ uint32_t frontier_size_shared;
    __shared__ bool done_shared;
    
    // Initialize
    if (tid == 0) {
        frontier_shared[0] = start_node;
        frontier_size_shared = 1;
        done_shared = false;
        distances[start_node] = 0;
        predecessors[start_node] = start_node;
    }
    
    // Initialize all distances
    for (uint32_t i = tid; i < num_nodes; i += blockDim.x * gridDim.x) {
        if (i != start_node) {
            distances[i] = -1;
            predecessors[i] = UINT32_MAX;
        }
    }
    __syncthreads();
    
    int32_t level = 0;
    
    while (!done_shared) {
        __shared__ uint32_t next_frontier_size;
        
        if (tid == 0) {
            next_frontier_size = 0;
        }
        __syncthreads();
        
        uint32_t frontier_size = frontier_size_shared;
        
        // Each warp processes different frontier nodes
        for (uint32_t f = warp_id; f < frontier_size; f += num_warps) {
            uint32_t node = frontier_shared[f];
            
            uint32_t row_start = csr_row_offsets[node];
            uint32_t row_end = csr_row_offsets[node + 1];
            uint32_t degree = row_end - row_start;
            
            // Warp processes neighbors in parallel
            for (uint32_t n = row_start + lane_id; n < row_end; n += 32) {
                uint32_t neighbor = csr_col_indices[n];
                
                // Check if already visited
                int32_t old_dist = atomicCAS(&distances[neighbor], -1, level + 1);
                
                if (old_dist == -1) {
                    predecessors[neighbor] = node;
                    
                    // Add to next frontier
                    uint32_t pos = atomicAdd(&next_frontier_size, 1);
                    if (pos < 1024) {
                        frontier_shared[pos] = neighbor;
                    }
                }
            }
        }
        __syncthreads();
        
        if (tid == 0) {
            frontier_size_shared = next_frontier_size;
            done_shared = (next_frontier_size == 0);
        }
        __syncthreads();
        
        level++;
    }
}

// Cycle detection using DFS coloring
__global__ void cycle_detection_kernel(
    const uint32_t* csr_row_offsets,
    const uint32_t* csr_col_indices,
    uint32_t num_nodes,
    uint8_t* colors,  // 0=white, 1=gray, 2=black
    bool* has_cycle,
    uint32_t* cycle_nodes
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize colors
    for (uint32_t i = tid; i < num_nodes; i += blockDim.x * gridDim.x) {
        colors[i] = 0; // White
    }
    __syncthreads();
    
    // DFS from each unvisited node
    for (uint32_t start = tid; start < num_nodes; start += blockDim.x * gridDim.x) {
        if (colors[start] == 0) {
            // Stack for DFS (simplified, fixed size)
            uint32_t stack[64];
            uint32_t stack_top = 0;
            
            stack[stack_top++] = start;
            colors[start] = 1; // Gray
            
            while (stack_top > 0 && !(*has_cycle)) {
                uint32_t node = stack[stack_top - 1];
                bool all_neighbors_processed = true;
                
                uint32_t row_start = csr_row_offsets[node];
                uint32_t row_end = csr_row_offsets[node + 1];
                
                for (uint32_t n = row_start; n < row_end; ++n) {
                    uint32_t neighbor = csr_col_indices[n];
                    uint8_t neighbor_color = colors[neighbor];
                    
                    if (neighbor_color == 1) {
                        // Found cycle (gray node)
                        *has_cycle = true;
                        
                        // Record cycle
                        if (cycle_nodes != nullptr) {
                            cycle_nodes[0] = node;
                            cycle_nodes[1] = neighbor;
                        }
                        break;
                    } else if (neighbor_color == 0) {
                        // White node, visit it
                        if (stack_top < 64) {
                            stack[stack_top++] = neighbor;
                            colors[neighbor] = 1; // Gray
                            all_neighbors_processed = false;
                            break;
                        }
                    }
                }
                
                if (all_neighbors_processed) {
                    colors[node] = 2; // Black
                    stack_top--;
                }
            }
        }
    }
}

// Topological sort using Kahn's algorithm
__global__ void topological_sort_kernel(
    const uint32_t* csr_row_offsets,
    const uint32_t* csr_col_indices,
    uint32_t num_nodes,
    uint32_t* in_degrees,
    uint32_t* topo_order,
    uint32_t* topo_index
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate in-degrees
    for (uint32_t node = tid; node < num_nodes; node += blockDim.x * gridDim.x) {
        in_degrees[node] = 0;
    }
    __syncthreads();
    
    // Count incoming edges
    for (uint32_t node = 0; node < num_nodes; ++node) {
        uint32_t row_start = csr_row_offsets[node];
        uint32_t row_end = csr_row_offsets[node + 1];
        
        for (uint32_t e = row_start + tid; e < row_end; e += blockDim.x * gridDim.x) {
            uint32_t neighbor = csr_col_indices[e];
            atomicAdd(&in_degrees[neighbor], 1);
        }
    }
    __syncthreads();
    
    // Process nodes with in-degree 0
    __shared__ uint32_t queue[256];
    __shared__ uint32_t queue_front, queue_rear;
    
    if (tid == 0) {
        queue_front = 0;
        queue_rear = 0;
        *topo_index = 0;
        
        // Find initial nodes with in-degree 0
        for (uint32_t i = 0; i < num_nodes; ++i) {
            if (in_degrees[i] == 0) {
                queue[queue_rear++] = i;
            }
        }
    }
    __syncthreads();
    
    // Process queue
    while (queue_front < queue_rear) {
        uint32_t node = queue[queue_front++];
        
        // Add to topological order
        uint32_t pos = atomicAdd(topo_index, 1);
        topo_order[pos] = node;
        
        // Decrease in-degree of neighbors
        uint32_t row_start = csr_row_offsets[node];
        uint32_t row_end = csr_row_offsets[node + 1];
        
        for (uint32_t e = row_start; e < row_end; ++e) {
            uint32_t neighbor = csr_col_indices[e];
            uint32_t new_degree = atomicSub(&in_degrees[neighbor], 1) - 1;
            
            if (new_degree == 0) {
                uint32_t pos = atomicAdd(&queue_rear, 1);
                if (pos < 256) {
                    queue[pos] = neighbor;
                }
            }
        }
        __syncthreads();
    }
}

// Host launchers
extern "C" void launch_parallel_bfs(
    const uint32_t* csr_row_offsets,
    const uint32_t* csr_col_indices,
    uint32_t num_nodes,
    uint32_t start_node,
    int32_t* distances,
    uint32_t* predecessors
) {
    bool* d_visited;
    cudaMalloc(&d_visited, num_nodes * sizeof(bool));
    cudaMemset(d_visited, 0, num_nodes * sizeof(bool));
    
    uint32_t threads = 256;
    uint32_t blocks = 1;
    size_t shared_mem = sizeof(BFSSharedMem);
    
    parallel_bfs_kernel<<<blocks, threads, shared_mem>>>(
        csr_row_offsets, csr_col_indices, num_nodes, start_node,
        distances, predecessors, d_visited
    );
    
    cudaFree(d_visited);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in parallel_bfs: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void launch_cycle_detection(
    const uint32_t* csr_row_offsets,
    const uint32_t* csr_col_indices,
    uint32_t num_nodes,
    bool* has_cycle,
    uint32_t* cycle_nodes
) {
    uint8_t* d_colors;
    cudaMalloc(&d_colors, num_nodes * sizeof(uint8_t));
    
    uint32_t threads = 256;
    uint32_t blocks = (num_nodes + threads - 1) / threads;
    
    cycle_detection_kernel<<<blocks, threads>>>(
        csr_row_offsets, csr_col_indices, num_nodes,
        d_colors, has_cycle, cycle_nodes
    );
    
    cudaFree(d_colors);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in cycle_detection: %s\n", cudaGetErrorString(err));
    }
}

} // namespace rustg