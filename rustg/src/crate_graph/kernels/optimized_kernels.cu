#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include "../../../include/gpu_types.h"

namespace cg = cooperative_groups;

namespace rustg {

// Fused kernel: Graph construction + Symbol table build
__global__ void fused_graph_symbol_kernel(
    const CrateNode* crates,
    uint32_t num_crates,
    const DependencyEdge* edges,
    uint32_t num_edges,
    const Symbol* symbols,
    uint32_t num_symbols,
    uint32_t* csr_row_offsets,
    uint32_t* csr_col_indices,
    uint32_t* hash_table,
    uint32_t table_size,
    uint32_t* stats  // [edge_count, collision_count, symbol_count]
) {
    extern __shared__ char shared_mem[];
    
    // Partition shared memory
    uint32_t* edge_counts = reinterpret_cast<uint32_t*>(shared_mem);
    uint32_t* local_symbols = reinterpret_cast<uint32_t*>(edge_counts + 256);
    
    const uint32_t tid = threadIdx.x;
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // Initialize shared memory
    if (tid < 256) {
        edge_counts[tid] = 0;
    }
    __syncthreads();
    
    // Phase 1: Count edges per node (warps 0-3)
    if (warp_id < 4) {
        uint32_t edges_per_warp = (num_edges + 3) / 4;
        uint32_t start = warp_id * edges_per_warp;
        uint32_t end = min(start + edges_per_warp, num_edges);
        
        for (uint32_t e = start + lane_id; e < end; e += 32) {
            if (e < num_edges) {
                uint32_t from = edges[e].from_crate;
                if (from < 256) {
                    atomicAdd(&edge_counts[from], 1);
                }
            }
        }
    }
    
    // Phase 2: Build symbol hash table (warps 4-7)
    if (warp_id >= 4) {
        uint32_t symbols_per_warp = num_symbols / 4;
        uint32_t start = (warp_id - 4) * symbols_per_warp;
        uint32_t end = min(start + symbols_per_warp, num_symbols);
        
        for (uint32_t s = start + lane_id; s < end; s += 32) {
            if (s < num_symbols) {
                const Symbol& sym = symbols[s];
                uint32_t hash = murmur3_32(sym.name_hash);
                uint32_t pos = hash % table_size;
                
                // Linear probing
                for (uint32_t i = 0; i < 32; ++i) {
                    uint32_t slot = (pos + i) % table_size;
                    uint32_t old = atomicCAS(&hash_table[slot * 2], UINT32_MAX, sym.name_hash);
                    
                    if (old == UINT32_MAX) {
                        hash_table[slot * 2 + 1] = s;
                        break;
                    } else if (old == sym.name_hash) {
                        break; // Duplicate
                    }
                    
                    if (i > 0) {
                        atomicAdd(&stats[1], 1); // Collision count
                    }
                }
            }
        }
    }
    __syncthreads();
    
    // Phase 3: Build CSR row offsets
    if (tid == 0) {
        csr_row_offsets[0] = 0;
        for (uint32_t i = 0; i < num_crates; ++i) {
            csr_row_offsets[i + 1] = csr_row_offsets[i] + edge_counts[i];
        }
        stats[0] = csr_row_offsets[num_crates]; // Total edges
        stats[2] = num_symbols; // Symbol count
    }
    __syncthreads();
    
    // Phase 4: Fill CSR column indices
    for (uint32_t e = gid; e < num_edges; e += blockDim.x * gridDim.x) {
        const DependencyEdge& edge = edges[e];
        uint32_t row_start = csr_row_offsets[edge.from_crate];
        uint32_t pos = atomicAdd(&edge_counts[edge.from_crate + 256], 1);
        csr_col_indices[row_start + pos] = edge.to_crate;
    }
}

// Optimized symbol resolution with prefetching
__global__ void optimized_symbol_resolution_kernel(
    const uint32_t* query_hashes,
    uint32_t num_queries,
    const uint32_t* hash_table,
    uint32_t table_size,
    const Symbol* symbols,
    const ModuleNode* modules,
    uint32_t num_modules,
    const uint8_t* visibility_matrix,
    uint32_t matrix_pitch,
    uint32_t* results,
    bool* found_flags
) {
    extern __shared__ uint32_t shared_cache[];
    
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t local_tid = threadIdx.x;
    const uint32_t warp_id = local_tid / 32;
    const uint32_t lane_id = local_tid % 32;
    
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
    
    // Prefetch common hash table entries to shared memory
    const uint32_t cache_size = 1024;
    if (local_tid < cache_size && local_tid < table_size) {
        shared_cache[local_tid] = hash_table[local_tid * 2];
        shared_cache[local_tid + cache_size] = hash_table[local_tid * 2 + 1];
    }
    __syncthreads();
    
    // Process queries with warp cooperation
    for (uint32_t q = tid / 32; q < num_queries; q += gridDim.x * blockDim.x / 32) {
        uint32_t query = query_hashes[q];
        uint32_t hash = murmur3_32(query);
        uint32_t base_pos = hash % table_size;
        
        bool found = false;
        uint32_t result_idx = UINT32_MAX;
        
        // Check shared cache first
        if (base_pos < cache_size) {
            for (uint32_t i = lane_id; i < 32; i += 32) {
                uint32_t pos = (base_pos + i) % cache_size;
                if (shared_cache[pos] == query) {
                    result_idx = shared_cache[pos + cache_size];
                    found = true;
                    break;
                }
            }
        }
        
        // If not in cache, search main table
        if (!found) {
            for (uint32_t offset = lane_id; offset < table_size; offset += 32) {
                uint32_t pos = (base_pos + offset) % table_size;
                uint32_t key = hash_table[pos * 2];
                
                if (key == query) {
                    result_idx = hash_table[pos * 2 + 1];
                    found = true;
                    break;
                } else if (key == UINT32_MAX) {
                    break; // Empty slot
                }
            }
        }
        
        // Reduce across warp
        uint32_t found_mask = warp.ballot_sync(found);
        if (found_mask != 0) {
            int first_lane = __ffs(found_mask) - 1;
            result_idx = warp.shfl_sync(0xFFFFFFFF, result_idx, first_lane);
            
            if (lane_id == 0) {
                results[q] = result_idx;
                found_flags[q] = true;
                
                // Check visibility if found
                if (result_idx < num_modules) {
                    const Symbol& sym = symbols[result_idx];
                    // Visibility check logic here
                }
            }
        } else if (lane_id == 0) {
            results[q] = UINT32_MAX;
            found_flags[q] = false;
        }
    }
}

// Fused module tree + visibility computation
__global__ void fused_module_visibility_kernel(
    const ModuleNode* modules,
    uint32_t num_modules,
    uint32_t* parent_pointers,
    uint32_t* depth_array,
    uint8_t* visibility_matrix,
    uint32_t matrix_pitch,
    uint32_t* tree_stats
) {
    extern __shared__ uint32_t shared_data[];
    
    uint32_t* local_depths = shared_data;
    uint32_t* local_parents = shared_data + 256;
    
    const uint32_t tid = threadIdx.x;
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load module data to shared memory
    if (tid < 256 && tid < num_modules) {
        const ModuleNode& mod = modules[tid];
        local_parents[tid] = mod.parent_id;
        local_depths[tid] = 0;
    }
    __syncthreads();
    
    // Phase 1: Compute depths in parallel
    if (gid < num_modules) {
        uint32_t depth = 0;
        uint32_t current = gid;
        
        // Use shared memory for fast parent lookups
        while (current != UINT32_MAX && depth < 32) {
            uint32_t parent = (current < 256) ? 
                            local_parents[current] : 
                            parent_pointers[current];
            
            if (parent == UINT32_MAX || parent == current) break;
            current = parent;
            depth++;
        }
        
        depth_array[gid] = depth;
        
        if (tid < 256) {
            local_depths[tid] = depth;
        }
    }
    __syncthreads();
    
    // Phase 2: Compute visibility matrix using depth information
    const uint32_t matrix_size = num_modules * num_modules;
    for (uint32_t idx = gid; idx < matrix_size; idx += blockDim.x * gridDim.x) {
        uint32_t from_module = idx / num_modules;
        uint32_t to_module = idx % num_modules;
        
        uint8_t visibility = 0;
        
        if (from_module == to_module) {
            visibility = 0xFF; // Full visibility to self
        } else {
            // Use cached depths for faster computation
            uint32_t from_depth = (from_module < 256) ? 
                                 local_depths[from_module] : 
                                 depth_array[from_module];
            uint32_t to_depth = (to_module < 256) ? 
                               local_depths[to_module] : 
                               depth_array[to_module];
            
            // Check ancestor relationship
            if (from_depth > to_depth) {
                // Check if 'to' is ancestor of 'from'
                uint32_t current = from_module;
                uint32_t steps = from_depth - to_depth;
                
                for (uint32_t i = 0; i < steps; ++i) {
                    current = parent_pointers[current];
                    if (current == to_module) {
                        visibility = 0x0F; // Ancestor visibility
                        break;
                    }
                }
            }
            
            // Check sibling visibility
            if (visibility == 0 && modules[from_module].crate_id == modules[to_module].crate_id) {
                visibility = 0x07; // Same crate visibility
            }
        }
        
        // Write to matrix
        uint8_t* row = visibility_matrix + from_module * matrix_pitch;
        row[to_module] = visibility;
    }
    
    // Phase 3: Compute tree statistics
    if (tid == 0) {
        uint32_t max_depth = 0;
        for (uint32_t i = 0; i < min(256u, num_modules); ++i) {
            max_depth = max(max_depth, local_depths[i]);
        }
        atomicMax(&tree_stats[0], max_depth);
        tree_stats[1] = num_modules;
    }
}

// Memory-optimized BFS with texture memory
__global__ void texture_optimized_bfs_kernel(
    cudaTextureObject_t row_offsets_tex,
    cudaTextureObject_t col_indices_tex,
    uint32_t num_nodes,
    uint32_t start_node,
    int32_t* distances,
    uint32_t* predecessors
) {
    extern __shared__ uint32_t shared_frontier[];
    
    uint32_t* frontier = shared_frontier;
    uint32_t* next_frontier = shared_frontier + 512;
    
    const uint32_t tid = threadIdx.x;
    const uint32_t num_threads = blockDim.x;
    
    __shared__ uint32_t frontier_size;
    __shared__ uint32_t next_size;
    __shared__ int32_t level;
    
    // Initialize
    if (tid == 0) {
        frontier[0] = start_node;
        frontier_size = 1;
        next_size = 0;
        level = 0;
        distances[start_node] = 0;
        predecessors[start_node] = start_node;
    }
    
    // Initialize all distances
    for (uint32_t i = tid; i < num_nodes; i += num_threads) {
        if (i != start_node) {
            distances[i] = -1;
            predecessors[i] = UINT32_MAX;
        }
    }
    __syncthreads();
    
    // BFS main loop
    while (frontier_size > 0) {
        next_size = 0;
        __syncthreads();
        
        // Process frontier in parallel
        for (uint32_t f = 0; f < frontier_size; f += num_threads) {
            if (f + tid < frontier_size) {
                uint32_t node = frontier[f + tid];
                
                // Use texture memory for coalesced access
                uint32_t row_start = tex1Dfetch<uint32_t>(row_offsets_tex, node);
                uint32_t row_end = tex1Dfetch<uint32_t>(row_offsets_tex, node + 1);
                
                // Process neighbors
                for (uint32_t n = row_start; n < row_end; ++n) {
                    uint32_t neighbor = tex1Dfetch<uint32_t>(col_indices_tex, n);
                    
                    int32_t old_dist = atomicCAS(&distances[neighbor], -1, level + 1);
                    
                    if (old_dist == -1) {
                        predecessors[neighbor] = node;
                        uint32_t pos = atomicAdd(&next_size, 1);
                        if (pos < 512) {
                            next_frontier[pos] = neighbor;
                        }
                    }
                }
            }
        }
        __syncthreads();
        
        // Swap frontiers
        if (tid < next_size && tid < 512) {
            frontier[tid] = next_frontier[tid];
        }
        
        if (tid == 0) {
            frontier_size = min(next_size, 512u);
            level++;
        }
        __syncthreads();
    }
}

// Cooperative groups kernel for large-scale traversal
__global__ void cooperative_graph_traversal_kernel(
    const uint32_t* csr_row_offsets,
    const uint32_t* csr_col_indices,
    uint32_t num_nodes,
    uint32_t* visited,
    uint32_t* component_labels,
    uint32_t* component_sizes
) {
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    
    const uint32_t tid = grid.thread_rank();
    const uint32_t grid_size = grid.size();
    
    __shared__ uint32_t shared_component_id;
    
    if (block.thread_rank() == 0) {
        shared_component_id = 0;
    }
    block.sync();
    
    // Find connected components
    for (uint32_t start = tid; start < num_nodes; start += grid_size) {
        if (atomicCAS(&visited[start], 0, 1) == 0) {
            // New component found
            uint32_t comp_id = atomicAdd(&shared_component_id, 1);
            component_labels[start] = comp_id;
            
            // BFS from this node
            uint32_t queue[64];
            uint32_t queue_size = 1;
            queue[0] = start;
            uint32_t comp_size = 1;
            
            while (queue_size > 0) {
                uint32_t node = queue[--queue_size];
                
                uint32_t row_start = csr_row_offsets[node];
                uint32_t row_end = csr_row_offsets[node + 1];
                
                for (uint32_t e = row_start; e < row_end; ++e) {
                    uint32_t neighbor = csr_col_indices[e];
                    
                    if (atomicCAS(&visited[neighbor], 0, 1) == 0) {
                        component_labels[neighbor] = comp_id;
                        comp_size++;
                        
                        if (queue_size < 64) {
                            queue[queue_size++] = neighbor;
                        }
                    }
                }
            }
            
            atomicAdd(&component_sizes[comp_id], comp_size);
        }
    }
    
    grid.sync(); // Grid-wide synchronization
}

// Host launchers for optimized kernels
extern "C" void launch_fused_graph_symbol(
    const CrateNode* crates, uint32_t num_crates,
    const DependencyEdge* edges, uint32_t num_edges,
    const Symbol* symbols, uint32_t num_symbols,
    uint32_t* csr_row_offsets, uint32_t* csr_col_indices,
    uint32_t* hash_table, uint32_t table_size,
    uint32_t* stats
) {
    uint32_t threads = 256;
    uint32_t blocks = 4;
    size_t shared_mem = 256 * sizeof(uint32_t) * 3; // edge_counts + local_symbols
    
    fused_graph_symbol_kernel<<<blocks, threads, shared_mem>>>(
        crates, num_crates, edges, num_edges,
        symbols, num_symbols,
        csr_row_offsets, csr_col_indices,
        hash_table, table_size, stats
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in fused_graph_symbol: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void launch_optimized_symbol_resolution(
    const uint32_t* query_hashes, uint32_t num_queries,
    const uint32_t* hash_table, uint32_t table_size,
    const Symbol* symbols,
    const ModuleNode* modules, uint32_t num_modules,
    const uint8_t* visibility_matrix, uint32_t matrix_pitch,
    uint32_t* results, bool* found_flags
) {
    uint32_t threads = 256;
    uint32_t blocks = (num_queries * 32 + threads - 1) / threads;
    size_t shared_mem = 2048 * sizeof(uint32_t); // Cache size
    
    optimized_symbol_resolution_kernel<<<blocks, threads, shared_mem>>>(
        query_hashes, num_queries,
        hash_table, table_size,
        symbols, modules, num_modules,
        visibility_matrix, matrix_pitch,
        results, found_flags
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in optimized_symbol_resolution: %s\n", cudaGetErrorString(err));
    }
}

} // namespace rustg