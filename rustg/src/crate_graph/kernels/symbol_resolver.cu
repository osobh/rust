#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include "../../../include/gpu_types.h"

namespace cg = cooperative_groups;

namespace rustg {

// Symbol resolution context
struct ResolutionContext {
    uint32_t current_crate;
    uint32_t current_module;
    uint32_t* import_map;        // Module import mappings
    uint32_t* use_statements;    // Use statement mappings
    uint32_t num_imports;
    uint32_t search_depth;       // Max module traversal depth
};

// Import statement representation
struct ImportStatement {
    uint32_t from_module;
    uint32_t to_module;
    uint32_t symbol_hash;
    uint32_t alias_hash;
    uint32_t import_type;  // 0=specific, 1=glob, 2=self, 3=super
    uint32_t visibility;
};

// Shared memory for resolution
struct ResolutionSharedMem {
    uint32_t candidate_symbols[64];    // Potential matches
    uint32_t candidate_scores[64];     // Match quality scores
    uint32_t candidate_count;
    uint32_t best_match_idx;
    uint32_t search_path[32];          // Module search path
    uint32_t path_length;
    bool found;
};

// Compute visibility score for symbol access
__device__ uint32_t compute_visibility_score(
    const Symbol& symbol,
    uint32_t requesting_crate,
    uint32_t requesting_module,
    uint32_t* module_hierarchy,
    uint32_t hierarchy_depth
) {
    // Score based on visibility and distance
    uint32_t score = 0;
    
    if (symbol.visibility == 0) { // Public
        score = 1000;
    } else if (symbol.visibility == 1) { // pub(crate)
        if (symbol.crate_id == requesting_crate) {
            score = 800;
        }
    } else if (symbol.visibility == 2) { // pub(super)
        // Check module hierarchy
        bool is_parent = false;
        for (uint32_t i = 0; i < hierarchy_depth; ++i) {
            if (module_hierarchy[i] == symbol.module_id) {
                is_parent = true;
                score = 600 - i * 10; // Closer parents score higher
                break;
            }
        }
    } else if (symbol.visibility == 3) { // pub(in path)
        // Check specific path visibility
        score = 400;
    } else { // Private
        if (symbol.module_id == requesting_module) {
            score = 200;
        }
    }
    
    return score;
}

// Resolve symbol with imports and use statements
__global__ void resolve_symbol_with_context_kernel(
    uint32_t query_hash,
    const ResolutionContext* context,
    const Symbol* symbols,
    uint32_t num_symbols,
    const uint32_t* hash_table,
    uint32_t table_size,
    const ImportStatement* imports,
    uint32_t num_imports,
    Symbol* result,
    bool* found,
    uint32_t* resolution_path
) {
    extern __shared__ char shared_mem_raw[];
    ResolutionSharedMem* shared = reinterpret_cast<ResolutionSharedMem*>(shared_mem_raw);
    
    const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // Initialize shared memory
    if (tid == 0) {
        shared->candidate_count = 0;
        shared->best_match_idx = UINT32_MAX;
        shared->path_length = 0;
        shared->found = false;
    }
    if (tid < 64) {
        shared->candidate_symbols[tid] = UINT32_MAX;
        shared->candidate_scores[tid] = 0;
    }
    __syncthreads();
    
    // Phase 1: Check direct imports
    for (uint32_t i = tid; i < num_imports; i += blockDim.x) {
        const ImportStatement& import = imports[i];
        
        if (import.from_module == context->current_module) {
            bool match = false;
            
            if (import.import_type == 0) { // Specific import
                match = (import.symbol_hash == query_hash || 
                        import.alias_hash == query_hash);
            } else if (import.import_type == 1) { // Glob import
                // Need to check all symbols from target module
                match = true; // Simplified for now
            }
            
            if (match) {
                uint32_t pos = atomicAdd(&shared->candidate_count, 1);
                if (pos < 64) {
                    shared->candidate_symbols[pos] = import.to_module;
                    shared->candidate_scores[pos] = 900; // High priority for direct imports
                }
            }
        }
    }
    __syncthreads();
    
    // Phase 2: Search in current module
    if (warp_id == 0) {
        uint32_t hash_pos = murmur3_32(query_hash) % table_size;
        
        for (uint32_t probe = lane_id; probe < table_size; probe += 32) {
            uint32_t pos = (hash_pos + probe) % table_size;
            uint32_t key = hash_table[pos * 2];
            
            if (key == query_hash) {
                uint32_t sym_idx = hash_table[pos * 2 + 1];
                const Symbol& sym = symbols[sym_idx];
                
                if (sym.module_id == context->current_module ||
                    sym.crate_id == context->current_crate) {
                    
                    uint32_t score = compute_visibility_score(
                        sym, context->current_crate, context->current_module,
                        nullptr, 0
                    );
                    
                    if (score > 0) {
                        uint32_t candidate_pos = atomicAdd(&shared->candidate_count, 1);
                        if (candidate_pos < 64) {
                            shared->candidate_symbols[candidate_pos] = sym_idx;
                            shared->candidate_scores[candidate_pos] = score;
                        }
                    }
                }
            } else if (key == UINT32_MAX) {
                break; // Empty slot
            }
        }
    }
    __syncthreads();
    
    // Phase 3: Search parent modules (super)
    if (warp_id == 1 && context->search_depth > 0) {
        // Build module hierarchy path
        uint32_t current_mod = context->current_module;
        uint32_t depth = 0;
        
        // Traverse up the module tree
        for (uint32_t level = 0; level < context->search_depth; ++level) {
            // Get parent module (simplified - would use module tree)
            uint32_t parent = current_mod / 10; // Placeholder logic
            
            if (parent == 0) break;
            
            // Search for symbol in parent
            uint32_t hash_pos = murmur3_32(query_hash) % table_size;
            
            for (uint32_t probe = lane_id; probe < 32; probe += 32) {
                uint32_t pos = (hash_pos + probe) % table_size;
                uint32_t key = hash_table[pos * 2];
                
                if (key == query_hash) {
                    uint32_t sym_idx = hash_table[pos * 2 + 1];
                    const Symbol& sym = symbols[sym_idx];
                    
                    if (sym.module_id == parent) {
                        uint32_t score = 500 - level * 50; // Decrease score with distance
                        
                        uint32_t candidate_pos = atomicAdd(&shared->candidate_count, 1);
                        if (candidate_pos < 64) {
                            shared->candidate_symbols[candidate_pos] = sym_idx;
                            shared->candidate_scores[candidate_pos] = score;
                        }
                    }
                }
            }
            
            current_mod = parent;
            depth++;
        }
    }
    __syncthreads();
    
    // Phase 4: Select best match
    if (tid < shared->candidate_count) {
        uint32_t score = shared->candidate_scores[tid];
        uint32_t old_best = atomicMax(&shared->candidate_scores[64], score);
        
        if (score > old_best) {
            shared->best_match_idx = shared->candidate_symbols[tid];
            shared->found = true;
        }
    }
    __syncthreads();
    
    // Write result
    if (tid == 0) {
        *found = shared->found;
        if (shared->found && shared->best_match_idx != UINT32_MAX) {
            *result = symbols[shared->best_match_idx];
            
            // Build resolution path
            resolution_path[0] = context->current_module;
            resolution_path[1] = shared->best_match_idx;
            resolution_path[2] = shared->path_length;
        }
    }
}

// Batch symbol resolution for multiple queries
__global__ void batch_resolve_symbols_kernel(
    const uint32_t* query_hashes,
    uint32_t num_queries,
    const ResolutionContext* contexts,
    const Symbol* symbols,
    uint32_t num_symbols,
    const uint32_t* hash_table,
    uint32_t table_size,
    uint32_t* results,
    bool* found_flags
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
    
    // Each warp handles one query
    if (warp_id < num_queries) {
        uint32_t query = query_hashes[warp_id];
        const ResolutionContext& ctx = contexts[warp_id];
        
        uint32_t hash_pos = murmur3_32(query) % table_size;
        bool found = false;
        uint32_t best_match = UINT32_MAX;
        uint32_t best_score = 0;
        
        // Parallel search with warp
        for (uint32_t offset = 0; offset < table_size; offset += 32) {
            uint32_t pos = (hash_pos + offset + lane_id) % table_size;
            uint32_t key = hash_table[pos * 2];
            
            if (key == query) {
                uint32_t sym_idx = hash_table[pos * 2 + 1];
                const Symbol& sym = symbols[sym_idx];
                
                uint32_t score = compute_visibility_score(
                    sym, ctx.current_crate, ctx.current_module,
                    nullptr, 0
                );
                
                if (score > best_score) {
                    best_score = score;
                    best_match = sym_idx;
                    found = true;
                }
            }
            
            // Check for early exit
            bool any_empty = (key == UINT32_MAX);
            uint32_t empty_mask = warp.ballot_sync(any_empty);
            if (empty_mask != 0 && !found) {
                break; // Hit empty slot without finding
            }
            
            // Check if found
            uint32_t found_mask = warp.ballot_sync(found);
            if (found_mask == 0xFFFFFFFF) {
                break; // All lanes found something
            }
        }
        
        // Reduce across warp to find best match
        for (int offset = 16; offset > 0; offset /= 2) {
            uint32_t other_score = warp.shfl_down_sync(0xFFFFFFFF, best_score, offset);
            uint32_t other_match = warp.shfl_down_sync(0xFFFFFFFF, best_match, offset);
            
            if (other_score > best_score) {
                best_score = other_score;
                best_match = other_match;
            }
        }
        
        // Lane 0 writes result
        if (lane_id == 0) {
            results[warp_id] = best_match;
            found_flags[warp_id] = (best_match != UINT32_MAX);
        }
    }
}

// Resolve symbols with caching
__global__ void cached_symbol_resolution_kernel(
    const uint32_t* query_hashes,
    uint32_t num_queries,
    uint32_t* cache,           // LRU cache
    uint32_t cache_size,
    const uint32_t* hash_table,
    uint32_t table_size,
    const Symbol* symbols,
    uint32_t* results
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ uint32_t cache_hits;
    __shared__ uint32_t cache_misses;
    
    if (threadIdx.x == 0) {
        cache_hits = 0;
        cache_misses = 0;
    }
    __syncthreads();
    
    if (tid < num_queries) {
        uint32_t query = query_hashes[tid];
        
        // Check cache first
        uint32_t cache_pos = murmur3_32(query) % cache_size;
        uint32_t cached_key = cache[cache_pos * 2];
        
        if (cached_key == query) {
            // Cache hit
            results[tid] = cache[cache_pos * 2 + 1];
            atomicAdd(&cache_hits, 1);
        } else {
            // Cache miss - perform full lookup
            atomicAdd(&cache_misses, 1);
            
            uint32_t hash_pos = murmur3_32(query) % table_size;
            bool found = false;
            
            for (uint32_t i = 0; i < table_size; ++i) {
                uint32_t pos = (hash_pos + i) % table_size;
                uint32_t key = hash_table[pos * 2];
                
                if (key == query) {
                    uint32_t sym_idx = hash_table[pos * 2 + 1];
                    results[tid] = sym_idx;
                    
                    // Update cache
                    cache[cache_pos * 2] = query;
                    cache[cache_pos * 2 + 1] = sym_idx;
                    found = true;
                    break;
                } else if (key == UINT32_MAX) {
                    results[tid] = UINT32_MAX;
                    break;
                }
            }
            
            if (!found) {
                results[tid] = UINT32_MAX;
            }
        }
    }
}

// Type-aware symbol resolution
__global__ void type_aware_resolution_kernel(
    uint32_t query_hash,
    uint32_t expected_type,     // Function, struct, trait, etc.
    const Symbol* symbols,
    uint32_t num_symbols,
    const uint32_t* hash_table,
    uint32_t table_size,
    Symbol* results,
    uint32_t* result_count,
    uint32_t max_results
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ uint32_t local_count;
    
    if (threadIdx.x == 0) {
        local_count = 0;
    }
    __syncthreads();
    
    // Search for matching symbols
    uint32_t hash_pos = murmur3_32(query_hash) % table_size;
    
    for (uint32_t i = tid; i < table_size; i += blockDim.x * gridDim.x) {
        uint32_t pos = (hash_pos + i) % table_size;
        uint32_t key = hash_table[pos * 2];
        
        if (key == query_hash) {
            uint32_t sym_idx = hash_table[pos * 2 + 1];
            const Symbol& sym = symbols[sym_idx];
            
            // Check if type matches
            if (sym.symbol_type == expected_type || expected_type == UINT32_MAX) {
                uint32_t result_pos = atomicAdd(&local_count, 1);
                
                if (result_pos < max_results) {
                    uint32_t global_pos = atomicAdd(result_count, 1);
                    if (global_pos < max_results) {
                        results[global_pos] = sym;
                    }
                }
            }
        } else if (key == UINT32_MAX) {
            break; // Empty slot
        }
    }
}

// Host launchers
extern "C" void launch_symbol_resolution_with_context(
    uint32_t query_hash,
    const ResolutionContext* context,
    const Symbol* symbols,
    uint32_t num_symbols,
    const uint32_t* hash_table,
    uint32_t table_size,
    const ImportStatement* imports,
    uint32_t num_imports,
    Symbol* result,
    bool* found,
    uint32_t* resolution_path
) {
    size_t shared_mem = sizeof(ResolutionSharedMem);
    
    resolve_symbol_with_context_kernel<<<1, 256, shared_mem>>>(
        query_hash, context, symbols, num_symbols,
        hash_table, table_size, imports, num_imports,
        result, found, resolution_path
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in symbol_resolution_with_context: %s\n", 
               cudaGetErrorString(err));
    }
}

extern "C" void launch_batch_symbol_resolution(
    const uint32_t* query_hashes,
    uint32_t num_queries,
    const ResolutionContext* contexts,
    const Symbol* symbols,
    uint32_t num_symbols,
    const uint32_t* hash_table,
    uint32_t table_size,
    uint32_t* results,
    bool* found_flags
) {
    uint32_t threads = 256;
    uint32_t warps_needed = num_queries;
    uint32_t blocks = (warps_needed * 32 + threads - 1) / threads;
    
    batch_resolve_symbols_kernel<<<blocks, threads>>>(
        query_hashes, num_queries, contexts,
        symbols, num_symbols, hash_table, table_size,
        results, found_flags
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in batch_symbol_resolution: %s\n", 
               cudaGetErrorString(err));
    }
}

} // namespace rustg