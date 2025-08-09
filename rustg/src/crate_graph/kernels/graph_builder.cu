#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cstdint>
#include "../../../include/gpu_types.h"

namespace cg = cooperative_groups;

namespace rustg {

// CSR (Compressed Sparse Row) format for graph representation
struct CSRGraph {
    uint32_t* row_offsets;    // Size: num_nodes + 1
    uint32_t* col_indices;    // Size: num_edges
    uint32_t* values;         // Size: num_edges (edge weights/types)
    uint32_t num_nodes;
    uint32_t num_edges;
};

// Shared memory for graph construction
struct GraphBuilderSharedMem {
    uint32_t edge_counts[256];      // Per-node edge count
    uint32_t edge_offsets[256];     // Prefix sum of counts
    DependencyEdge edges[512];      // Local edge buffer
    uint32_t total_edges;
};

// Count edges per node (out-degree)
__global__ void count_edges_kernel(
    const DependencyEdge* edges,
    uint32_t num_edges,
    uint32_t* edge_counts,
    uint32_t num_nodes
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize counts
    if (tid < num_nodes) {
        edge_counts[tid] = 0;
    }
    __syncthreads();
    
    // Count edges
    if (tid < num_edges) {
        const DependencyEdge& edge = edges[tid];
        atomicAdd(&edge_counts[edge.from_crate], 1);
    }
}

// Build CSR row offsets using prefix sum
__global__ void build_row_offsets_kernel(
    const uint32_t* edge_counts,
    uint32_t num_nodes,
    uint32_t* row_offsets
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid == 0) {
        row_offsets[0] = 0;
        
        // Sequential prefix sum (can be optimized with parallel scan)
        for (uint32_t i = 0; i < num_nodes; ++i) {
            row_offsets[i + 1] = row_offsets[i] + edge_counts[i];
        }
    }
}

// Parallel prefix sum using CUB
__global__ void parallel_prefix_sum_kernel(
    const uint32_t* edge_counts,
    uint32_t num_nodes,
    uint32_t* row_offsets
) {
    extern __shared__ uint32_t shared_data[];
    
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    
    cg::thread_block block = cg::this_thread_block();
    
    // Load data to shared memory
    if (tid < num_nodes) {
        shared_data[tid] = edge_counts[tid];
    } else {
        shared_data[tid] = 0;
    }
    __syncthreads();
    
    // Up-sweep (reduce) phase
    for (uint32_t stride = 1; stride < blockDim.x; stride *= 2) {
        uint32_t index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            shared_data[index] += shared_data[index - stride];
        }
        __syncthreads();
    }
    
    // Down-sweep phase
    if (tid == 0) {
        row_offsets[0] = 0;
        shared_data[blockDim.x - 1] = 0;
    }
    __syncthreads();
    
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        uint32_t index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            uint32_t temp = shared_data[index - stride];
            shared_data[index - stride] = shared_data[index];
            shared_data[index] += temp;
        }
        __syncthreads();
    }
    
    // Write results
    if (tid < num_nodes) {
        row_offsets[tid + 1] = shared_data[tid] + edge_counts[tid];
    }
}

// Fill CSR column indices and values
__global__ void fill_csr_data_kernel(
    const DependencyEdge* edges,
    uint32_t num_edges,
    const uint32_t* row_offsets,
    uint32_t* col_indices,
    uint32_t* values,
    uint32_t* next_idx  // Atomic counters per node
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_edges) {
        const DependencyEdge& edge = edges[tid];
        
        // Get position in CSR arrays
        uint32_t row_start = row_offsets[edge.from_crate];
        uint32_t local_idx = atomicAdd(&next_idx[edge.from_crate], 1);
        uint32_t pos = row_start + local_idx;
        
        // Fill data
        col_indices[pos] = edge.to_crate;
        values[pos] = edge.edge_type;
    }
}

// Optimized CSR construction with warp cooperation
__global__ void build_csr_optimized_kernel(
    const CrateNode* crates,
    uint32_t num_crates,
    const DependencyEdge* edges,
    uint32_t num_edges,
    uint32_t* row_offsets,
    uint32_t* col_indices,
    uint32_t* values
) {
    extern __shared__ char shared_mem_raw[];
    GraphBuilderSharedMem* shared = 
        reinterpret_cast<GraphBuilderSharedMem*>(shared_mem_raw);
    
    const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // Initialize shared memory
    if (tid < 256) {
        shared->edge_counts[tid] = 0;
        shared->edge_offsets[tid] = 0;
    }
    if (tid == 0) {
        shared->total_edges = 0;
    }
    __syncthreads();
    
    // Phase 1: Count edges per node using warps
    if (warp_id < 4) {
        uint32_t edges_per_warp = (num_edges + 3) / 4;
        uint32_t start_edge = warp_id * edges_per_warp;
        uint32_t end_edge = min(start_edge + edges_per_warp, num_edges);
        
        for (uint32_t e = start_edge + lane_id; e < end_edge; e += 32) {
            if (e < num_edges) {
                uint32_t from = edges[e].from_crate;
                if (from < 256) {
                    atomicAdd(&shared->edge_counts[from], 1);
                }
            }
        }
    }
    __syncthreads();
    
    // Phase 2: Compute prefix sum for row offsets
    if (tid == 0) {
        row_offsets[0] = 0;
        for (uint32_t i = 0; i < num_crates; ++i) {
            shared->edge_offsets[i] = (i == 0) ? 0 : 
                shared->edge_offsets[i-1] + shared->edge_counts[i-1];
            row_offsets[i + 1] = shared->edge_offsets[i] + shared->edge_counts[i];
        }
        shared->total_edges = row_offsets[num_crates];
    }
    __syncthreads();
    
    // Phase 3: Fill CSR data in parallel
    for (uint32_t e = tid; e < num_edges; e += blockDim.x) {
        const DependencyEdge& edge = edges[e];
        uint32_t from = edge.from_crate;
        
        if (from < num_crates) {
            // Use atomic to get unique position
            uint32_t pos = atomicAdd(&shared->edge_offsets[from], 1);
            
            // Reset offset if needed
            if (pos >= row_offsets[from + 1]) {
                pos = row_offsets[from]; // Fallback
            }
            
            col_indices[pos] = edge.to_crate;
            values[pos] = edge.edge_type;
        }
    }
}

// Host function to launch graph builder
extern "C" void launch_crate_graph_builder(
    const CrateNode* crates,
    uint32_t num_crates,
    const DependencyEdge* edges,
    uint32_t num_edges,
    uint32_t* csr_row_offsets,
    uint32_t* csr_col_indices,
    uint32_t* csr_values
) {
    // Allocate temporary storage
    uint32_t* d_edge_counts;
    uint32_t* d_next_idx;
    
    cudaMalloc(&d_edge_counts, num_crates * sizeof(uint32_t));
    cudaMalloc(&d_next_idx, num_crates * sizeof(uint32_t));
    
    cudaMemset(d_edge_counts, 0, num_crates * sizeof(uint32_t));
    cudaMemset(d_next_idx, 0, num_crates * sizeof(uint32_t));
    
    // Count edges
    uint32_t threads = 256;
    uint32_t blocks = (num_edges + threads - 1) / threads;
    count_edges_kernel<<<blocks, threads>>>(
        edges, num_edges, d_edge_counts, num_crates
    );
    
    // Build row offsets
    build_row_offsets_kernel<<<1, 1>>>(
        d_edge_counts, num_crates, csr_row_offsets
    );
    
    // Fill CSR data
    fill_csr_data_kernel<<<blocks, threads>>>(
        edges, num_edges, csr_row_offsets,
        csr_col_indices, csr_values, d_next_idx
    );
    
    // Cleanup
    cudaFree(d_edge_counts);
    cudaFree(d_next_idx);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in crate_graph_builder: %s\n",
               cudaGetErrorString(err));
    }
}

// Optimized launcher
extern "C" void launch_crate_graph_builder_optimized(
    const CrateNode* crates,
    uint32_t num_crates,
    const DependencyEdge* edges,
    uint32_t num_edges,
    uint32_t* csr_row_offsets,
    uint32_t* csr_col_indices,
    uint32_t* csr_values
) {
    uint32_t threads = 256;
    uint32_t blocks = 1;
    size_t shared_mem = sizeof(GraphBuilderSharedMem);
    
    build_csr_optimized_kernel<<<blocks, threads, shared_mem>>>(
        crates, num_crates, edges, num_edges,
        csr_row_offsets, csr_col_indices, csr_values
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in build_csr_optimized: %s\n",
               cudaGetErrorString(err));
    }
}

} // namespace rustg