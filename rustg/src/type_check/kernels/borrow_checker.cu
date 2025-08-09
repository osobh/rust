#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include "../../../include/gpu_types.h"

namespace cg = cooperative_groups;

namespace rustg {

// Lifetime representation
enum LifetimeKind : uint8_t {
    LIFETIME_STATIC,       // 'static
    LIFETIME_ANONYMOUS,    // '_
    LIFETIME_NAMED,        // 'a, 'b, etc.
    LIFETIME_ELIDED,       // Compiler inferred
    LIFETIME_LOCAL         // Local scope
};

struct Lifetime {
    uint32_t lifetime_id;
    LifetimeKind kind;
    uint32_t scope_start;      // Start of lifetime scope
    uint32_t scope_end;        // End of lifetime scope
    uint32_t parent_lifetime;  // Outlives relationship
    uint32_t region_id;        // Region in the program
};

// Borrow representation
enum BorrowKind : uint8_t {
    BORROW_SHARED,         // &T
    BORROW_MUTABLE,        // &mut T
    BORROW_UNIQUE          // Box<T> or unique ownership
};

struct Borrow {
    uint32_t borrow_id;
    uint32_t borrowed_place;   // What is borrowed
    BorrowKind kind;
    uint32_t lifetime_id;      // Lifetime of the borrow
    uint32_t location;         // Where in code
    bool is_active;            // Currently active
};

// Place/location being borrowed
struct Place {
    uint32_t place_id;
    uint32_t var_id;           // Variable being borrowed
    uint32_t* projection;      // Field/index projections
    uint32_t projection_len;
    bool is_move;              // Has been moved
    bool is_initialized;       // Has been initialized
};

// Borrow conflict
struct BorrowConflict {
    uint32_t borrow1_id;
    uint32_t borrow2_id;
    ConflictKind kind;
    uint32_t location;
};

enum ConflictKind : uint8_t {
    CONFLICT_MULTIPLE_MUT,     // Multiple mutable borrows
    CONFLICT_MUT_AND_SHARED,   // Mutable and shared borrow
    CONFLICT_USE_AFTER_MOVE,   // Use after move
    CONFLICT_LIFETIME,         // Lifetime violation
    CONFLICT_UNINITIALIZED     // Use of uninitialized
};

// Shared memory for borrow checking
struct BorrowCheckSharedMem {
    Borrow active_borrows[256];
    uint32_t num_active;
    Lifetime local_lifetimes[128];
    uint32_t conflict_count;
    BorrowConflict conflicts[64];
    uint32_t dataflow_state[256];  // Bit vector for dataflow
};

// Check if two lifetimes overlap
__device__ bool lifetimes_overlap(
    const Lifetime& lt1,
    const Lifetime& lt2
) {
    // Check if scopes overlap
    return !(lt1.scope_end < lt2.scope_start || 
             lt2.scope_end < lt1.scope_start);
}

// Check if lifetime outlives another
__device__ bool outlives(
    const Lifetime& longer,
    const Lifetime& shorter
) {
    if (longer.kind == LIFETIME_STATIC) {
        return true; // 'static outlives everything
    }
    
    return longer.scope_start <= shorter.scope_start &&
           longer.scope_end >= shorter.scope_end;
}

// Check if two borrows conflict
__device__ bool borrows_conflict(
    const Borrow& b1,
    const Borrow& b2,
    const Place* places
) {
    // Check if they borrow the same place
    if (b1.borrowed_place != b2.borrowed_place) {
        // Check for overlapping projections
        const Place& p1 = places[b1.borrowed_place];
        const Place& p2 = places[b2.borrowed_place];
        
        if (p1.var_id != p2.var_id) {
            return false; // Different variables
        }
        
        // Check projection overlap (simplified)
        // Would need to check field/index paths
    }
    
    // Check borrow kinds
    if (b1.kind == BORROW_MUTABLE || b2.kind == BORROW_MUTABLE) {
        return true; // Any mutable borrow conflicts
    }
    
    return false; // Two shared borrows don't conflict
}

// Main borrow checking kernel
__global__ void borrow_check_kernel(
    const Borrow* borrows,
    uint32_t num_borrows,
    const Place* places,
    uint32_t num_places,
    const Lifetime* lifetimes,
    uint32_t num_lifetimes,
    BorrowConflict* conflicts,
    uint32_t* conflict_count,
    uint32_t* error_locations
) {
    extern __shared__ char shared_mem_raw[];
    BorrowCheckSharedMem* shared = 
        reinterpret_cast<BorrowCheckSharedMem*>(shared_mem_raw);
    
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // Initialize shared memory
    if (tid == 0) {
        shared->num_active = 0;
        shared->conflict_count = 0;
    }
    __syncthreads();
    
    // Load active borrows to shared memory
    for (uint32_t b = tid; b < num_borrows && b < 256; b += blockDim.x) {
        if (borrows[b].is_active) {
            uint32_t idx = atomicAdd(&shared->num_active, 1);
            if (idx < 256) {
                shared->active_borrows[idx] = borrows[b];
            }
        }
    }
    __syncthreads();
    
    // Check for conflicts between borrows
    uint32_t num_active = shared->num_active;
    
    for (uint32_t i = gid; i < num_active * num_active; i += blockDim.x * gridDim.x) {
        uint32_t b1_idx = i / num_active;
        uint32_t b2_idx = i % num_active;
        
        if (b1_idx < b2_idx) { // Only check each pair once
            const Borrow& b1 = shared->active_borrows[b1_idx];
            const Borrow& b2 = shared->active_borrows[b2_idx];
            
            // Check if borrows conflict
            if (borrows_conflict(b1, b2, places)) {
                // Check if lifetimes overlap
                const Lifetime& lt1 = lifetimes[b1.lifetime_id];
                const Lifetime& lt2 = lifetimes[b2.lifetime_id];
                
                if (lifetimes_overlap(lt1, lt2)) {
                    // Found a conflict
                    uint32_t conflict_idx = atomicAdd(&shared->conflict_count, 1);
                    
                    if (conflict_idx < 64) {
                        BorrowConflict& conflict = shared->conflicts[conflict_idx];
                        conflict.borrow1_id = b1.borrow_id;
                        conflict.borrow2_id = b2.borrow_id;
                        
                        // Determine conflict type
                        if (b1.kind == BORROW_MUTABLE && b2.kind == BORROW_MUTABLE) {
                            conflict.kind = CONFLICT_MULTIPLE_MUT;
                        } else {
                            conflict.kind = CONFLICT_MUT_AND_SHARED;
                        }
                        
                        conflict.location = max(b1.location, b2.location);
                    }
                }
            }
        }
    }
    __syncthreads();
    
    // Write conflicts to global memory
    if (tid < shared->conflict_count && tid < 64) {
        conflicts[bid * 64 + tid] = shared->conflicts[tid];
    }
    
    if (tid == 0) {
        atomicAdd(conflict_count, shared->conflict_count);
    }
}

// Lifetime inference kernel
__global__ void lifetime_inference_kernel(
    Lifetime* lifetimes,
    uint32_t num_lifetimes,
    const Borrow* borrows,
    uint32_t num_borrows,
    uint32_t* outlives_constraints,  // Pairs of (longer, shorter)
    uint32_t num_constraints,
    bool* changed
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process outlives constraints
    for (uint32_t c = tid; c < num_constraints; c += blockDim.x * gridDim.x) {
        uint32_t longer_id = outlives_constraints[c * 2];
        uint32_t shorter_id = outlives_constraints[c * 2 + 1];
        
        if (longer_id < num_lifetimes && shorter_id < num_lifetimes) {
            Lifetime& longer = lifetimes[longer_id];
            Lifetime& shorter = lifetimes[shorter_id];
            
            // Adjust lifetimes to satisfy constraint
            if (!outlives(longer, shorter)) {
                // Extend longer lifetime
                uint32_t old_start = atomicMin(&longer.scope_start, shorter.scope_start);
                uint32_t old_end = atomicMax(&longer.scope_end, shorter.scope_end);
                
                if (old_start != longer.scope_start || old_end != longer.scope_end) {
                    *changed = true;
                }
            }
        }
    }
}

// Dataflow analysis for initialization and moves
__global__ void dataflow_analysis_kernel(
    Place* places,
    uint32_t num_places,
    const uint32_t* control_flow_graph,
    uint32_t num_blocks,
    uint32_t* dataflow_state,  // Bit vectors
    uint32_t state_size,
    uint32_t max_iterations
) {
    extern __shared__ uint32_t shared_state[];
    
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    
    __shared__ bool converged;
    __shared__ uint32_t iteration;
    
    if (tid == 0) {
        converged = false;
        iteration = 0;
    }
    __syncthreads();
    
    // Initialize dataflow state
    if (tid < state_size) {
        shared_state[tid] = 0; // All uninitialized
    }
    __syncthreads();
    
    // Iterative dataflow analysis
    while (!converged && iteration < max_iterations) {
        bool local_changed = false;
        
        // Process each basic block
        for (uint32_t bb = tid; bb < num_blocks; bb += blockDim.x) {
            uint32_t bb_start = control_flow_graph[bb * 2];
            uint32_t bb_end = control_flow_graph[bb * 2 + 1];
            
            // Compute GEN and KILL sets
            uint32_t gen = 0;
            uint32_t kill = 0;
            
            for (uint32_t inst = bb_start; inst < bb_end; ++inst) {
                // Check for initialization
                if (inst < num_places) {
                    Place& place = places[inst];
                    
                    if (place.is_initialized) {
                        gen |= (1 << place.place_id);
                    }
                    if (place.is_move) {
                        kill |= (1 << place.place_id);
                    }
                }
            }
            
            // Update dataflow state
            uint32_t old_state = shared_state[bb];
            uint32_t new_state = (old_state & ~kill) | gen;
            
            if (new_state != old_state) {
                shared_state[bb] = new_state;
                local_changed = true;
            }
        }
        
        __syncthreads();
        
        // Check convergence
        if (tid == 0) {
            converged = !local_changed;
            iteration++;
        }
        __syncthreads();
    }
    
    // Write results to global memory
    if (tid < state_size) {
        dataflow_state[bid * state_size + tid] = shared_state[tid];
    }
}

// Move checking kernel
__global__ void move_check_kernel(
    Place* places,
    uint32_t num_places,
    const Borrow* borrows,
    uint32_t num_borrows,
    uint32_t* move_errors,
    uint32_t* error_count
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check each place for move violations
    for (uint32_t p = tid; p < num_places; p += blockDim.x * gridDim.x) {
        Place& place = places[p];
        
        if (place.is_move) {
            // Check if place is used after move
            for (uint32_t b = 0; b < num_borrows; ++b) {
                const Borrow& borrow = borrows[b];
                
                if (borrow.borrowed_place == p && 
                    borrow.location > place.place_id) { // After move
                    
                    // Found use after move
                    uint32_t err_idx = atomicAdd(error_count, 1);
                    if (err_idx < 100) {
                        move_errors[err_idx * 2] = p;
                        move_errors[err_idx * 2 + 1] = borrow.location;
                    }
                    break;
                }
            }
        }
        
        if (!place.is_initialized) {
            // Check if uninitialized place is used
            for (uint32_t b = 0; b < num_borrows; ++b) {
                const Borrow& borrow = borrows[b];
                
                if (borrow.borrowed_place == p) {
                    // Found use of uninitialized
                    uint32_t err_idx = atomicAdd(error_count, 1);
                    if (err_idx < 100) {
                        move_errors[err_idx * 2] = p;
                        move_errors[err_idx * 2 + 1] = borrow.location;
                    }
                    break;
                }
            }
        }
    }
}

// Region inference kernel
__global__ void region_inference_kernel(
    Lifetime* lifetimes,
    uint32_t num_lifetimes,
    uint32_t* region_graph,     // Adjacency matrix
    uint32_t graph_size,
    uint32_t* region_map,        // Maps lifetimes to regions
    uint32_t* num_regions
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
    
    // Union-Find for region merging
    __shared__ uint32_t parent[1024];
    __shared__ uint32_t rank[1024];
    
    // Initialize union-find
    if (tid < num_lifetimes) {
        parent[tid] = tid;
        rank[tid] = 0;
    }
    __syncthreads();
    
    // Process region constraints
    for (uint32_t i = warp_id; i < num_lifetimes; i += gridDim.x * blockDim.x / 32) {
        for (uint32_t j = lane_id; j < num_lifetimes; j += 32) {
            uint32_t edge = region_graph[i * graph_size + j];
            
            if (edge > 0) { // Connected in region graph
                // Union the regions
                uint32_t root_i = find_root(parent, i);
                uint32_t root_j = find_root(parent, j);
                
                if (root_i != root_j) {
                    union_types(parent, rank, root_i, root_j);
                }
            }
        }
    }
    __syncthreads();
    
    // Assign region IDs
    if (tid < num_lifetimes) {
        uint32_t root = find_root(parent, tid);
        region_map[tid] = root;
        lifetimes[tid].region_id = root;
    }
    
    // Count unique regions
    if (tid == 0) {
        uint32_t region_count = 0;
        for (uint32_t i = 0; i < num_lifetimes; ++i) {
            if (parent[i] == i) {
                region_count++;
            }
        }
        *num_regions = region_count;
    }
}

// Two-phase borrow checking
__global__ void two_phase_borrow_kernel(
    Borrow* borrows,
    uint32_t num_borrows,
    uint32_t* activation_points,  // When borrows become active
    uint32_t num_points,
    bool* two_phase_needed       // Output: which borrows need two-phase
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check each borrow for two-phase requirements
    for (uint32_t b = tid; b < num_borrows; b += blockDim.x * gridDim.x) {
        Borrow& borrow = borrows[b];
        
        if (borrow.kind == BORROW_MUTABLE) {
            // Check if there's a gap between reservation and activation
            uint32_t activation = activation_points[b];
            
            if (activation > borrow.location) {
                // Two-phase borrow needed
                two_phase_needed[b] = true;
                
                // Borrow starts as reserved, not active
                borrow.is_active = false;
            }
        }
    }
}

// Host launchers
extern "C" void launch_borrow_check(
    const Borrow* borrows,
    uint32_t num_borrows,
    const Place* places,
    uint32_t num_places,
    const Lifetime* lifetimes,
    uint32_t num_lifetimes,
    BorrowConflict* conflicts,
    uint32_t* conflict_count,
    uint32_t* error_locations
) {
    uint32_t threads = 256;
    uint32_t blocks = (num_borrows + threads - 1) / threads;
    size_t shared_mem = sizeof(BorrowCheckSharedMem);
    
    borrow_check_kernel<<<blocks, threads, shared_mem>>>(
        borrows, num_borrows,
        places, num_places,
        lifetimes, num_lifetimes,
        conflicts, conflict_count,
        error_locations
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in borrow_check: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void launch_lifetime_inference(
    Lifetime* lifetimes,
    uint32_t num_lifetimes,
    const Borrow* borrows,
    uint32_t num_borrows,
    uint32_t* outlives_constraints,
    uint32_t num_constraints,
    bool* changed
) {
    uint32_t threads = 256;
    uint32_t blocks = (num_constraints + threads - 1) / threads;
    
    lifetime_inference_kernel<<<blocks, threads>>>(
        lifetimes, num_lifetimes,
        borrows, num_borrows,
        outlives_constraints, num_constraints,
        changed
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in lifetime_inference: %s\n", cudaGetErrorString(err));
    }
}

} // namespace rustg