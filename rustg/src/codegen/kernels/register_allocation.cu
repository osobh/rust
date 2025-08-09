#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include "../../../include/gpu_types.h"

namespace cg = cooperative_groups;

namespace rustg {

// Live range representation
struct LiveRange {
    uint32_t var_id;
    uint32_t start_point;
    uint32_t end_point;
    uint32_t register_hint;
    uint32_t spill_weight;
    bool is_spilled;
    bool is_fixed;  // Fixed register assignment
};

// Interference graph for register allocation
struct InterferenceGraph {
    uint32_t* adjacency_matrix;  // N x N bit matrix
    uint32_t num_vars;
    uint32_t* degree;  // Number of neighbors
    uint32_t* colors;  // Register assignments
    uint32_t num_colors;  // Available registers
};

// Register allocation state
struct RegisterAllocation {
    uint32_t* allocation;  // var_id -> register mapping
    uint32_t num_registers;
    uint32_t* spill_slots;
    uint32_t num_spills;
    LiveRange* ranges;
    uint32_t num_ranges;
};

// Shared memory for graph coloring
struct GraphColoringSharedMem {
    uint32_t local_degrees[256];
    uint32_t local_colors[256];
    uint32_t removed_nodes[256];
    uint32_t num_removed;
    uint32_t available_colors[32];  // Bitmask of available colors
    uint32_t simplify_worklist[256];
    uint32_t worklist_size;
};

// Check if two live ranges interfere
__device__ bool ranges_interfere(
    const LiveRange& r1,
    const LiveRange& r2
) {
    // Two ranges interfere if they overlap
    return r1.start_point < r2.end_point && 
           r2.start_point < r1.end_point;
}

// Set bit in adjacency matrix
__device__ void set_interference(
    uint32_t* matrix,
    uint32_t num_vars,
    uint32_t v1,
    uint32_t v2
) {
    if (v1 != v2 && v1 < num_vars && v2 < num_vars) {
        uint32_t idx = v1 * num_vars + v2;
        uint32_t word = idx / 32;
        uint32_t bit = idx % 32;
        atomicOr(&matrix[word], 1u << bit);
        
        // Symmetric
        idx = v2 * num_vars + v1;
        word = idx / 32;
        bit = idx % 32;
        atomicOr(&matrix[word], 1u << bit);
    }
}

// Check if two variables interfere
__device__ bool check_interference(
    const uint32_t* matrix,
    uint32_t num_vars,
    uint32_t v1,
    uint32_t v2
) {
    if (v1 == v2) return false;
    uint32_t idx = v1 * num_vars + v2;
    uint32_t word = idx / 32;
    uint32_t bit = idx % 32;
    return (matrix[word] >> bit) & 1;
}

// Build interference graph kernel
__global__ void build_interference_graph_kernel(
    const LiveRange* ranges,
    uint32_t num_ranges,
    uint32_t* adjacency_matrix,
    uint32_t* degrees,
    uint32_t matrix_words
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
    
    // Clear matrix
    for (uint32_t i = tid; i < matrix_words; i += blockDim.x * gridDim.x) {
        adjacency_matrix[i] = 0;
    }
    
    // Synchronize all threads
    if (tid == 0) {
        __threadfence();
    }
    __syncthreads();
    
    // Build interference graph - each thread handles one range
    for (uint32_t i = tid; i < num_ranges; i += blockDim.x * gridDim.x) {
        const LiveRange& r1 = ranges[i];
        uint32_t local_degree = 0;
        
        // Check interference with all other ranges
        for (uint32_t j = 0; j < num_ranges; ++j) {
            if (i != j) {
                const LiveRange& r2 = ranges[j];
                
                if (ranges_interfere(r1, r2)) {
                    set_interference(adjacency_matrix, num_ranges, i, j);
                    local_degree++;
                }
            }
        }
        
        // Store degree
        degrees[i] = local_degree;
    }
}

// Graph coloring kernel using Chaitin's algorithm
__global__ void graph_coloring_kernel(
    InterferenceGraph* graph,
    RegisterAllocation* allocation,
    uint32_t num_physical_regs
) {
    extern __shared__ char shared_mem_raw[];
    GraphColoringSharedMem* shared = reinterpret_cast<GraphColoringSharedMem*>(shared_mem_raw);
    
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // Initialize shared memory
    if (tid == 0) {
        shared->num_removed = 0;
        shared->worklist_size = 0;
    }
    
    // Load degrees into shared memory
    for (uint32_t i = tid; i < graph->num_vars && i < 256; i += blockDim.x) {
        shared->local_degrees[i] = graph->degree[i];
        shared->local_colors[i] = UINT32_MAX;  // Uncolored
    }
    __syncthreads();
    
    // Simplify phase - remove nodes with degree < K
    bool simplified = true;
    while (simplified) {
        simplified = false;
        
        // Find nodes with degree < num_physical_regs
        for (uint32_t i = tid; i < graph->num_vars && i < 256; i += blockDim.x) {
            if (shared->local_colors[i] == UINT32_MAX &&  // Not colored
                shared->local_degrees[i] < num_physical_regs) {
                
                // Add to simplify worklist
                uint32_t idx = atomicAdd(&shared->worklist_size, 1);
                if (idx < 256) {
                    shared->simplify_worklist[idx] = i;
                    simplified = true;
                }
            }
        }
        __syncthreads();
        
        // Remove nodes from graph
        for (uint32_t w = 0; w < shared->worklist_size; ++w) {
            uint32_t node = shared->simplify_worklist[w];
            
            // Push onto stack
            uint32_t stack_idx = atomicAdd(&shared->num_removed, 1);
            if (stack_idx < 256) {
                shared->removed_nodes[stack_idx] = node;
            }
            
            // Decrease degree of neighbors
            for (uint32_t n = tid; n < graph->num_vars; n += blockDim.x) {
                if (check_interference(graph->adjacency_matrix, 
                                     graph->num_vars, node, n)) {
                    atomicSub(&shared->local_degrees[n], 1);
                }
            }
        }
        
        if (tid == 0) {
            shared->worklist_size = 0;
        }
        __syncthreads();
    }
    
    // Select phase - assign colors to nodes
    for (int32_t i = shared->num_removed - 1; i >= 0; --i) {
        uint32_t node = shared->removed_nodes[i];
        
        // Find available colors
        if (tid == 0) {
            // Initialize all colors as available
            for (uint32_t c = 0; c < num_physical_regs && c < 32; ++c) {
                shared->available_colors[0] |= (1u << c);
            }
            
            // Remove colors used by neighbors
            for (uint32_t n = 0; n < graph->num_vars; ++n) {
                if (check_interference(graph->adjacency_matrix,
                                     graph->num_vars, node, n)) {
                    uint32_t neighbor_color = shared->local_colors[n];
                    if (neighbor_color != UINT32_MAX && neighbor_color < 32) {
                        shared->available_colors[0] &= ~(1u << neighbor_color);
                    }
                }
            }
            
            // Select first available color
            uint32_t selected_color = UINT32_MAX;
            for (uint32_t c = 0; c < num_physical_regs; ++c) {
                if (shared->available_colors[0] & (1u << c)) {
                    selected_color = c;
                    break;
                }
            }
            
            shared->local_colors[node] = selected_color;
            
            // Clear for next iteration
            shared->available_colors[0] = 0;
        }
        __syncthreads();
    }
    
    // Write results to global memory
    for (uint32_t i = tid; i < graph->num_vars && i < 256; i += blockDim.x) {
        graph->colors[i] = shared->local_colors[i];
        allocation->allocation[i] = shared->local_colors[i];
        
        // Mark as spilled if no color assigned
        if (shared->local_colors[i] == UINT32_MAX) {
            allocation->ranges[i].is_spilled = true;
            uint32_t spill_slot = atomicAdd(&allocation->num_spills, 1);
            allocation->spill_slots[i] = spill_slot;
        }
    }
}

// Linear scan register allocation (alternative algorithm)
__global__ void linear_scan_kernel(
    LiveRange* ranges,
    uint32_t num_ranges,
    RegisterAllocation* allocation,
    uint32_t num_registers
) {
    extern __shared__ LiveRange active_ranges[];
    __shared__ uint32_t active_count;
    __shared__ uint32_t free_registers[32];  // Bitmask
    
    const uint32_t tid = threadIdx.x;
    
    // Initialize
    if (tid == 0) {
        active_count = 0;
        free_registers[0] = (1u << num_registers) - 1;  // All registers free
    }
    __syncthreads();
    
    // Sort ranges by start point (simplified - assume pre-sorted)
    
    // Process each range
    for (uint32_t i = 0; i < num_ranges; ++i) {
        LiveRange& current = ranges[i];
        
        // Expire old intervals
        if (tid < active_count) {
            LiveRange& active = active_ranges[tid];
            if (active.end_point <= current.start_point) {
                // Return register to free pool
                uint32_t reg = allocation->allocation[active.var_id];
                atomicOr(&free_registers[0], 1u << reg);
                
                // Remove from active (mark for removal)
                active.var_id = UINT32_MAX;
            }
        }
        __syncthreads();
        
        // Compact active list (remove expired)
        if (tid == 0) {
            uint32_t write_idx = 0;
            for (uint32_t j = 0; j < active_count; ++j) {
                if (active_ranges[j].var_id != UINT32_MAX) {
                    if (write_idx != j) {
                        active_ranges[write_idx] = active_ranges[j];
                    }
                    write_idx++;
                }
            }
            active_count = write_idx;
        }
        __syncthreads();
        
        // Allocate register for current
        if (tid == 0) {
            if (free_registers[0] != 0) {
                // Find first free register
                uint32_t reg = __ffs(free_registers[0]) - 1;
                allocation->allocation[current.var_id] = reg;
                
                // Mark as used
                free_registers[0] &= ~(1u << reg);
                
                // Add to active
                if (active_count < 256) {
                    active_ranges[active_count++] = current;
                }
            } else {
                // Spill - find range with furthest end point
                uint32_t spill_idx = 0;
                uint32_t max_end = active_ranges[0].end_point;
                
                for (uint32_t j = 1; j < active_count; ++j) {
                    if (active_ranges[j].end_point > max_end) {
                        max_end = active_ranges[j].end_point;
                        spill_idx = j;
                    }
                }
                
                if (current.end_point < max_end) {
                    // Spill the other range
                    LiveRange& spilled = active_ranges[spill_idx];
                    uint32_t reg = allocation->allocation[spilled.var_id];
                    
                    // Reassign register
                    allocation->allocation[current.var_id] = reg;
                    allocation->allocation[spilled.var_id] = UINT32_MAX;
                    
                    // Mark as spilled
                    ranges[spilled.var_id].is_spilled = true;
                    uint32_t spill_slot = atomicAdd(&allocation->num_spills, 1);
                    allocation->spill_slots[spilled.var_id] = spill_slot;
                    
                    // Update active list
                    active_ranges[spill_idx] = current;
                } else {
                    // Spill current
                    current.is_spilled = true;
                    allocation->allocation[current.var_id] = UINT32_MAX;
                    uint32_t spill_slot = atomicAdd(&allocation->num_spills, 1);
                    allocation->spill_slots[current.var_id] = spill_slot;
                }
            }
        }
        __syncthreads();
    }
}

// Spill code generation kernel
__global__ void generate_spill_code_kernel(
    const RegisterAllocation* allocation,
    const LiveRange* ranges,
    uint32_t num_ranges,
    Instruction* spill_loads,
    Instruction* spill_stores,
    uint32_t* num_spill_instrs
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (uint32_t i = tid; i < num_ranges; i += blockDim.x * gridDim.x) {
        const LiveRange& range = ranges[i];
        
        if (range.is_spilled) {
            uint32_t spill_slot = allocation->spill_slots[i];
            uint32_t stack_offset = spill_slot * 8;  // 8 bytes per slot
            
            // Generate spill store at definition
            uint32_t store_idx = atomicAdd(num_spill_instrs, 1);
            if (store_idx < 10000) {  // Limit check
                Instruction& store = spill_stores[store_idx];
                store.opcode = OP_STORE;
                store.operands[0] = range.var_id;
                store.operands[1] = stack_offset;  // Stack location
                store.num_operands = 2;
            }
            
            // Generate spill load at use
            uint32_t load_idx = atomicAdd(num_spill_instrs, 1);
            if (load_idx < 10000) {
                Instruction& load = spill_loads[load_idx];
                load.opcode = OP_LOAD;
                load.result_id = range.var_id;
                load.operands[0] = stack_offset;
                load.num_operands = 1;
            }
        }
    }
}

// Coalescing kernel - merge move-related variables
__global__ void coalescing_kernel(
    const Instruction* instructions,
    uint32_t num_instructions,
    LiveRange* ranges,
    uint32_t num_ranges,
    uint32_t* coalesce_map,
    uint32_t* num_coalesced
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Look for move instructions
    for (uint32_t i = tid; i < num_instructions; i += blockDim.x * gridDim.x) {
        const Instruction& inst = instructions[i];
        
        // Check if it's a move (simplified check)
        if (inst.opcode == OP_MOV && inst.num_operands == 1) {
            uint32_t src = inst.operands[0];
            uint32_t dst = inst.result_id;
            
            // Check if they can be coalesced (don't interfere)
            if (src < num_ranges && dst < num_ranges) {
                LiveRange& src_range = ranges[src];
                LiveRange& dst_range = ranges[dst];
                
                if (!ranges_interfere(src_range, dst_range)) {
                    // Coalesce - map src to dst
                    coalesce_map[src] = dst;
                    
                    // Merge live ranges
                    dst_range.start_point = min(dst_range.start_point, 
                                               src_range.start_point);
                    dst_range.end_point = max(dst_range.end_point,
                                             src_range.end_point);
                    
                    atomicAdd(num_coalesced, 1);
                }
            }
        }
    }
}

// Host launchers
extern "C" void launch_build_interference_graph(
    const LiveRange* ranges,
    uint32_t num_ranges,
    uint32_t* adjacency_matrix,
    uint32_t* degrees
) {
    uint32_t matrix_bits = num_ranges * num_ranges;
    uint32_t matrix_words = (matrix_bits + 31) / 32;
    
    uint32_t threads = 256;
    uint32_t blocks = (num_ranges + threads - 1) / threads;
    
    build_interference_graph_kernel<<<blocks, threads>>>(
        ranges, num_ranges,
        adjacency_matrix, degrees,
        matrix_words
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in build_interference_graph: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void launch_graph_coloring(
    InterferenceGraph* graph,
    RegisterAllocation* allocation,
    uint32_t num_physical_regs
) {
    uint32_t threads = 256;
    uint32_t blocks = 1;  // Single block for simplicity
    size_t shared_mem = sizeof(GraphColoringSharedMem);
    
    graph_coloring_kernel<<<blocks, threads, shared_mem>>>(
        graph, allocation, num_physical_regs
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in graph_coloring: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void launch_linear_scan(
    LiveRange* ranges,
    uint32_t num_ranges,
    RegisterAllocation* allocation,
    uint32_t num_registers
) {
    uint32_t threads = 256;
    uint32_t blocks = 1;
    size_t shared_mem = 256 * sizeof(LiveRange);  // Active list
    
    linear_scan_kernel<<<blocks, threads, shared_mem>>>(
        ranges, num_ranges,
        allocation, num_registers
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in linear_scan: %s\n", cudaGetErrorString(err));
    }
}

} // namespace rustg