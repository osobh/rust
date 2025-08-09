#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include "../../../include/gpu_types.h"

namespace cg = cooperative_groups;

namespace rustg {

// Module tree node structure
struct ModuleNode {
    uint32_t module_id;
    uint32_t parent_id;
    uint32_t crate_id;
    uint32_t name_hash;
    uint32_t children_start;    // Index in children array
    uint32_t children_count;
    uint32_t depth;             // Depth in tree
    uint32_t visibility_mask;   // Visibility flags
    uint32_t symbol_start;      // Start index in symbol table
    uint32_t symbol_count;      // Number of symbols
    uint32_t attributes;        // Module attributes (inline, etc.)
    uint32_t file_id;          // Source file reference
};

// Module path component
struct PathComponent {
    uint32_t name_hash;
    uint32_t module_type;  // 0=normal, 1=self, 2=super, 3=crate
    uint32_t index;        // Position in path
};

// Shared memory for tree operations
struct ModuleTreeSharedMem {
    ModuleNode local_nodes[64];
    uint32_t parent_indices[64];
    uint32_t depth_counts[16];     // Count per depth level
    uint32_t max_depth;
    uint32_t total_nodes;
    PathComponent path_buffer[32];
    uint32_t path_length;
};

// Build module tree from flat module list
__global__ void build_module_tree_kernel(
    const ModuleNode* modules,
    uint32_t num_modules,
    uint32_t* parent_pointers,    // Parent index for each module
    uint32_t* children_lists,     // Flattened children arrays
    uint32_t* children_offsets,   // Start index for each module's children
    uint32_t* depth_array,        // Computed depth for each module
    uint32_t* tree_stats          // Statistics: max_depth, total_edges, etc.
) {
    extern __shared__ char shared_mem_raw[];
    ModuleTreeSharedMem* shared = reinterpret_cast<ModuleTreeSharedMem*>(shared_mem_raw);
    
    const uint32_t tid = threadIdx.x;
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // Initialize shared memory
    if (tid < 16) {
        shared->depth_counts[tid] = 0;
    }
    if (tid == 0) {
        shared->max_depth = 0;
        shared->total_nodes = 0;
    }
    __syncthreads();
    
    // Phase 1: Build parent pointers and count children
    if (gid < num_modules) {
        const ModuleNode& module = modules[gid];
        parent_pointers[gid] = module.parent_id;
        
        // Count as child of parent
        if (module.parent_id != UINT32_MAX) {
            atomicAdd(&children_offsets[module.parent_id], 1);
        }
    }
    __syncthreads();
    
    // Phase 2: Compute depth for each module
    if (gid < num_modules) {
        uint32_t depth = 0;
        uint32_t current = gid;
        
        // Traverse up to root
        while (current != UINT32_MAX && depth < 32) {
            uint32_t parent = parent_pointers[current];
            if (parent == UINT32_MAX || parent == current) {
                break; // Root or self-loop
            }
            current = parent;
            depth++;
        }
        
        depth_array[gid] = depth;
        atomicMax(&shared->max_depth, depth);
        atomicAdd(&shared->depth_counts[depth % 16], 1);
    }
    __syncthreads();
    
    // Phase 3: Build children lists
    if (gid < num_modules) {
        uint32_t parent = parent_pointers[gid];
        if (parent != UINT32_MAX && parent < num_modules) {
            uint32_t child_pos = atomicAdd(&children_offsets[parent], 1);
            children_lists[child_pos] = gid;
        }
    }
    
    // Write statistics
    if (tid == 0 && blockIdx.x == 0) {
        tree_stats[0] = shared->max_depth;
        tree_stats[1] = num_modules;
        uint32_t total_edges = 0;
        for (uint32_t i = 0; i < 16; ++i) {
            total_edges += shared->depth_counts[i];
        }
        tree_stats[2] = total_edges;
    }
}

// Parallel module path resolution
__global__ void resolve_module_path_kernel(
    const PathComponent* path,
    uint32_t path_length,
    uint32_t starting_module,
    const ModuleNode* modules,
    uint32_t num_modules,
    const uint32_t* parent_pointers,
    const uint32_t* children_lists,
    const uint32_t* children_offsets,
    uint32_t* resolved_module,
    bool* resolution_success
) {
    extern __shared__ char shared_mem_raw[];
    ModuleTreeSharedMem* shared = reinterpret_cast<ModuleTreeSharedMem*>(shared_mem_raw);
    
    const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
    
    // Copy path to shared memory
    if (tid < path_length) {
        shared->path_buffer[tid] = path[tid];
    }
    if (tid == 0) {
        shared->path_length = path_length;
        *resolution_success = false;
    }
    __syncthreads();
    
    uint32_t current_module = starting_module;
    
    // Process each path component
    for (uint32_t i = 0; i < path_length; ++i) {
        const PathComponent& component = shared->path_buffer[i];
        
        if (component.module_type == 1) { // self
            // Stay in current module
            continue;
        } else if (component.module_type == 2) { // super
            // Go to parent module
            if (current_module < num_modules) {
                current_module = parent_pointers[current_module];
                if (current_module == UINT32_MAX) {
                    // Hit root, path invalid
                    break;
                }
            }
        } else if (component.module_type == 3) { // crate
            // Go to crate root
            current_module = 0; // Assuming crate root is always 0
        } else { // Normal module name
            // Search children for matching name
            if (current_module >= num_modules) break;
            
            uint32_t children_start = children_offsets[current_module];
            uint32_t children_end = (current_module + 1 < num_modules) ? 
                                   children_offsets[current_module + 1] : 
                                   children_start;
            uint32_t children_count = children_end - children_start;
            
            bool found = false;
            
            // Parallel search with warp
            for (uint32_t c = children_start + lane_id; 
                 c < children_end; c += 32) {
                uint32_t child_id = children_lists[c];
                if (child_id < num_modules) {
                    const ModuleNode& child = modules[child_id];
                    if (child.name_hash == component.name_hash) {
                        current_module = child_id;
                        found = true;
                    }
                }
            }
            
            // Check if any lane found it
            uint32_t found_mask = warp.ballot_sync(found);
            if (found_mask == 0) {
                // Module not found
                current_module = UINT32_MAX;
                break;
            }
            
            // Get the found module from the first lane that found it
            int first_lane = __ffs(found_mask) - 1;
            current_module = warp.shfl_sync(0xFFFFFFFF, current_module, first_lane);
        }
    }
    
    // Write result
    if (tid == 0) {
        *resolved_module = current_module;
        *resolution_success = (current_module != UINT32_MAX);
    }
}

// Compute module visibility matrix
__global__ void compute_visibility_matrix_kernel(
    const ModuleNode* modules,
    uint32_t num_modules,
    const uint32_t* parent_pointers,
    const uint32_t* depth_array,
    uint8_t* visibility_matrix,    // NxN matrix of visibility flags
    uint32_t matrix_pitch           // Row pitch in bytes
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t from_module = tid / num_modules;
    const uint32_t to_module = tid % num_modules;
    
    if (from_module < num_modules && to_module < num_modules) {
        const ModuleNode& from = modules[from_module];
        const ModuleNode& to = modules[to_module];
        
        uint8_t visibility = 0;
        
        // Same module - full visibility
        if (from_module == to_module) {
            visibility = 0xFF;
        }
        // Same crate - check visibility rules
        else if (from.crate_id == to.crate_id) {
            // Check if 'to' is ancestor of 'from'
            uint32_t current = from_module;
            bool is_ancestor = false;
            
            while (current != UINT32_MAX) {
                if (current == to_module) {
                    is_ancestor = true;
                    break;
                }
                current = parent_pointers[current];
            }
            
            if (is_ancestor) {
                visibility = 0x0F; // Child can see parent's pub(super)
            } else {
                // Check if modules share common ancestor
                uint32_t from_depth = depth_array[from_module];
                uint32_t to_depth = depth_array[to_module];
                
                // Find common ancestor
                uint32_t from_current = from_module;
                uint32_t to_current = to_module;
                
                // Bring to same depth
                while (from_depth > to_depth) {
                    from_current = parent_pointers[from_current];
                    from_depth--;
                }
                while (to_depth > from_depth) {
                    to_current = parent_pointers[to_current];
                    to_depth--;
                }
                
                // Find common ancestor
                while (from_current != to_current && 
                       from_current != UINT32_MAX && 
                       to_current != UINT32_MAX) {
                    from_current = parent_pointers[from_current];
                    to_current = parent_pointers[to_current];
                }
                
                if (from_current == to_current) {
                    visibility = 0x07; // Sibling visibility
                }
            }
        }
        // Different crates - only public visibility
        else {
            if (to.visibility_mask & 0x01) { // Public flag
                visibility = 0x01;
            }
        }
        
        // Write to matrix
        uint8_t* row = visibility_matrix + from_module * matrix_pitch;
        row[to_module] = visibility;
    }
}

// Find all modules visible from a given module
__global__ void find_visible_modules_kernel(
    uint32_t query_module,
    const ModuleNode* modules,
    uint32_t num_modules,
    const uint8_t* visibility_matrix,
    uint32_t matrix_pitch,
    uint32_t* visible_modules,
    uint32_t* visible_count,
    uint32_t max_results
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ uint32_t local_count;
    
    if (threadIdx.x == 0) {
        local_count = 0;
    }
    __syncthreads();
    
    // Check visibility for each module
    if (tid < num_modules) {
        const uint8_t* row = visibility_matrix + query_module * matrix_pitch;
        uint8_t visibility = row[tid];
        
        if (visibility > 0) {
            uint32_t pos = atomicAdd(&local_count, 1);
            if (pos < max_results) {
                uint32_t global_pos = atomicAdd(visible_count, 1);
                if (global_pos < max_results) {
                    visible_modules[global_pos] = tid;
                }
            }
        }
    }
}

// Parallel module tree traversal
__global__ void traverse_module_tree_kernel(
    uint32_t root_module,
    const ModuleNode* modules,
    uint32_t num_modules,
    const uint32_t* children_lists,
    const uint32_t* children_offsets,
    uint32_t* traversal_order,
    uint32_t* traversal_count,
    uint32_t max_depth
) {
    extern __shared__ uint32_t shared_stack[];
    
    const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
    
    __shared__ uint32_t stack_top;
    __shared__ uint32_t output_pos;
    
    if (tid == 0) {
        stack_top = 0;
        output_pos = 0;
        shared_stack[stack_top++] = root_module;
    }
    __syncthreads();
    
    // Iterative DFS traversal
    while (stack_top > 0) {
        // Pop from stack
        uint32_t current = UINT32_MAX;
        if (tid == 0) {
            current = shared_stack[--stack_top];
        }
        current = __shfl_sync(0xFFFFFFFF, current, 0);
        
        if (current < num_modules) {
            // Add to traversal order
            if (tid == 0) {
                uint32_t pos = atomicAdd(traversal_count, 1);
                traversal_order[pos] = current;
            }
            
            // Get children
            uint32_t children_start = children_offsets[current];
            uint32_t children_end = (current + 1 < num_modules) ?
                                   children_offsets[current + 1] :
                                   children_start;
            
            // Push children to stack (in parallel)
            for (uint32_t c = children_start + lane_id;
                 c < children_end && stack_top < 1024;
                 c += 32) {
                uint32_t child = children_lists[c];
                if (lane_id == 0) {
                    shared_stack[atomicAdd(&stack_top, 1)] = child;
                }
            }
        }
        __syncthreads();
    }
}

// Module dependency analysis
__global__ void analyze_module_dependencies_kernel(
    const ModuleNode* modules,
    uint32_t num_modules,
    const Symbol* symbols,
    uint32_t num_symbols,
    const ImportStatement* imports,
    uint32_t num_imports,
    uint32_t* dependency_matrix,    // Module-to-module dependencies
    uint32_t* dependency_counts      // Number of dependencies per module
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process imports to build dependency matrix
    if (tid < num_imports) {
        const ImportStatement& import = imports[tid];
        
        // Mark dependency from importing module to target module
        uint32_t from_mod = import.from_module;
        uint32_t to_mod = import.to_module;
        
        if (from_mod < num_modules && to_mod < num_modules) {
            uint32_t matrix_idx = from_mod * num_modules + to_mod;
            uint32_t old_val = atomicExch(&dependency_matrix[matrix_idx], 1);
            
            if (old_val == 0) {
                // New dependency
                atomicAdd(&dependency_counts[from_mod], 1);
            }
        }
    }
}

// Host launchers
extern "C" void launch_build_module_tree(
    const ModuleNode* modules,
    uint32_t num_modules,
    uint32_t* parent_pointers,
    uint32_t* children_lists,
    uint32_t* children_offsets,
    uint32_t* depth_array,
    uint32_t* tree_stats
) {
    uint32_t threads = 256;
    uint32_t blocks = (num_modules + threads - 1) / threads;
    size_t shared_mem = sizeof(ModuleTreeSharedMem);
    
    build_module_tree_kernel<<<blocks, threads, shared_mem>>>(
        modules, num_modules,
        parent_pointers, children_lists, children_offsets,
        depth_array, tree_stats
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in build_module_tree: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void launch_resolve_module_path(
    const PathComponent* path,
    uint32_t path_length,
    uint32_t starting_module,
    const ModuleNode* modules,
    uint32_t num_modules,
    const uint32_t* parent_pointers,
    const uint32_t* children_lists,
    const uint32_t* children_offsets,
    uint32_t* resolved_module,
    bool* resolution_success
) {
    size_t shared_mem = sizeof(ModuleTreeSharedMem);
    
    resolve_module_path_kernel<<<1, 256, shared_mem>>>(
        path, path_length, starting_module,
        modules, num_modules,
        parent_pointers, children_lists, children_offsets,
        resolved_module, resolution_success
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in resolve_module_path: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void launch_compute_visibility_matrix(
    const ModuleNode* modules,
    uint32_t num_modules,
    const uint32_t* parent_pointers,
    const uint32_t* depth_array,
    uint8_t* visibility_matrix,
    uint32_t matrix_pitch
) {
    uint32_t total_entries = num_modules * num_modules;
    uint32_t threads = 256;
    uint32_t blocks = (total_entries + threads - 1) / threads;
    
    compute_visibility_matrix_kernel<<<blocks, threads>>>(
        modules, num_modules,
        parent_pointers, depth_array,
        visibility_matrix, matrix_pitch
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in compute_visibility_matrix: %s\n", cudaGetErrorString(err));
    }
}

} // namespace rustg