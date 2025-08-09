#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include "../../../include/gpu_types.h"

namespace cg = cooperative_groups;

namespace rustg {

// GPU-optimized hash table for symbol storage
// Uses open addressing with linear probing for cache efficiency
struct GPUHashTable {
    uint32_t* keys;           // Symbol name hashes
    uint32_t* values;         // Symbol indices
    uint32_t* metadata;       // Additional symbol data
    uint32_t size;            // Table size (power of 2)
    uint32_t mask;            // Size - 1 for fast modulo
    uint32_t count;           // Number of entries
    float load_factor;        // Max load before resize
};

// Symbol structure for GPU processing
struct Symbol {
    uint32_t name_hash;       // Hash of symbol name
    uint32_t crate_id;        // Owning crate
    uint32_t module_id;       // Owning module
    uint32_t symbol_type;     // Function, struct, enum, etc.
    uint32_t visibility;      // Public, private, crate-local
    uint32_t definition_loc;  // Location in source
    uint32_t attributes;      // Additional flags
    uint32_t generic_params;  // Number of generic parameters
};

// Shared memory for hash table operations
struct HashTableSharedMem {
    uint32_t probe_counts[256];      // Collision statistics
    uint32_t insert_positions[256];  // Insertion locations
    Symbol local_symbols[128];       // Local symbol buffer
    uint32_t num_collisions;
    uint32_t max_probe_distance;
};

// MurmurHash3 for GPU - optimized for 32-bit
__device__ inline uint32_t murmur3_32(uint32_t key, uint32_t seed = 0x9747b28c) {
    key ^= seed;
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key;
}

// Secondary hash for double hashing
__device__ inline uint32_t hash2(uint32_t key) {
    return 1 + (murmur3_32(key, 0xdeadbeef) % 31);
}

// Initialize hash table
__global__ void init_hash_table_kernel(
    uint32_t* keys,
    uint32_t* values,
    uint32_t* metadata,
    uint32_t size
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize entries to empty (0xFFFFFFFF)
    for (uint32_t i = tid; i < size; i += blockDim.x * gridDim.x) {
        keys[i] = UINT32_MAX;
        values[i] = UINT32_MAX;
        metadata[i] = 0;
    }
}

// Insert symbols into hash table with linear probing
__global__ void insert_symbols_kernel(
    const Symbol* symbols,
    uint32_t num_symbols,
    uint32_t* keys,
    uint32_t* values,
    uint32_t* metadata,
    uint32_t table_size,
    uint32_t* collision_count
) {
    extern __shared__ char shared_mem_raw[];
    HashTableSharedMem* shared = reinterpret_cast<HashTableSharedMem*>(shared_mem_raw);
    
    const uint32_t tid = threadIdx.x;
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // Initialize shared memory
    if (tid < 256) {
        shared->probe_counts[tid] = 0;
        shared->insert_positions[tid] = UINT32_MAX;
    }
    if (tid == 0) {
        shared->num_collisions = 0;
        shared->max_probe_distance = 0;
    }
    __syncthreads();
    
    const uint32_t mask = table_size - 1;
    
    // Each thread processes one symbol
    if (gid < num_symbols) {
        const Symbol& sym = symbols[gid];
        uint32_t hash = murmur3_32(sym.name_hash);
        uint32_t pos = hash & mask;
        uint32_t probe_distance = 0;
        
        // Linear probing with early exit
        while (probe_distance < table_size) {
            uint32_t old_key = atomicCAS(&keys[pos], UINT32_MAX, sym.name_hash);
            
            if (old_key == UINT32_MAX) {
                // Successfully inserted
                values[pos] = gid;
                metadata[pos] = (sym.crate_id << 16) | (sym.module_id & 0xFFFF);
                
                // Track statistics
                if (probe_distance > 0) {
                    atomicAdd(&shared->num_collisions, 1);
                    atomicMax(&shared->max_probe_distance, probe_distance);
                }
                break;
            } else if (old_key == sym.name_hash) {
                // Duplicate key - handle based on policy
                // For now, keep first occurrence
                break;
            }
            
            // Collision - continue probing
            probe_distance++;
            pos = (pos + 1) & mask;
        }
        
        shared->probe_counts[tid] = probe_distance;
    }
    __syncthreads();
    
    // Aggregate statistics
    if (tid == 0) {
        atomicAdd(collision_count, shared->num_collisions);
    }
}

// Optimized insertion using double hashing
__global__ void insert_symbols_double_hash_kernel(
    const Symbol* symbols,
    uint32_t num_symbols,
    uint32_t* keys,
    uint32_t* values,
    uint32_t* metadata,
    uint32_t table_size,
    uint32_t* collision_count
) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t mask = table_size - 1;
    
    if (gid < num_symbols) {
        const Symbol& sym = symbols[gid];
        uint32_t h1 = murmur3_32(sym.name_hash);
        uint32_t h2 = hash2(sym.name_hash);
        uint32_t pos = h1 & mask;
        uint32_t probe_count = 0;
        
        // Double hashing probe sequence
        for (uint32_t i = 0; i < table_size; ++i) {
            uint32_t old_key = atomicCAS(&keys[pos], UINT32_MAX, sym.name_hash);
            
            if (old_key == UINT32_MAX || old_key == sym.name_hash) {
                if (old_key == UINT32_MAX) {
                    values[pos] = gid;
                    metadata[pos] = (sym.crate_id << 16) | (sym.module_id & 0xFFFF);
                    
                    if (probe_count > 0) {
                        atomicAdd(collision_count, 1);
                    }
                }
                break;
            }
            
            probe_count++;
            pos = (pos + h2) & mask;
        }
    }
}

// Batch symbol lookup kernel
__global__ void lookup_symbols_kernel(
    const uint32_t* query_hashes,
    uint32_t num_queries,
    const uint32_t* keys,
    const uint32_t* values,
    const uint32_t* metadata,
    uint32_t table_size,
    uint32_t* results,
    bool* found_flags
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t mask = table_size - 1;
    
    if (tid < num_queries) {
        uint32_t query = query_hashes[tid];
        uint32_t hash = murmur3_32(query);
        uint32_t pos = hash & mask;
        bool found = false;
        
        // Linear probe for the key
        for (uint32_t i = 0; i < table_size; ++i) {
            uint32_t key = keys[pos];
            
            if (key == query) {
                results[tid] = values[pos];
                found_flags[tid] = true;
                found = true;
                break;
            } else if (key == UINT32_MAX) {
                // Empty slot - key not found
                break;
            }
            
            pos = (pos + 1) & mask;
        }
        
        if (!found) {
            results[tid] = UINT32_MAX;
            found_flags[tid] = false;
        }
    }
}

// Parallel symbol lookup with warp cooperation
__global__ void lookup_symbols_warp_kernel(
    const uint32_t* query_hashes,
    uint32_t num_queries,
    const uint32_t* keys,
    const uint32_t* values,
    uint32_t table_size,
    uint32_t* results
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    const uint32_t mask = table_size - 1;
    
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
    
    // Each warp handles one query
    if (warp_id < num_queries) {
        uint32_t query = query_hashes[warp_id];
        uint32_t hash = murmur3_32(query);
        uint32_t base_pos = hash & mask;
        
        // Warp searches 32 positions in parallel
        uint32_t pos = (base_pos + lane_id) & mask;
        uint32_t key = keys[pos];
        bool found = (key == query);
        
        // Find first matching lane
        uint32_t found_mask = warp.ballot_sync(found);
        
        if (found_mask != 0) {
            int first_lane = __ffs(found_mask) - 1;
            if (lane_id == first_lane) {
                results[warp_id] = values[pos];
            }
        } else {
            // Continue searching if not found in first 32 positions
            for (uint32_t offset = 32; offset < table_size; offset += 32) {
                pos = (base_pos + offset + lane_id) & mask;
                key = (pos < table_size) ? keys[pos] : UINT32_MAX;
                found = (key == query);
                
                found_mask = warp.ballot_sync(found);
                if (found_mask != 0) {
                    int first_lane = __ffs(found_mask) - 1;
                    if (lane_id == first_lane) {
                        results[warp_id] = values[pos];
                    }
                    break;
                }
                
                // Check for empty slots
                bool empty = (key == UINT32_MAX);
                uint32_t empty_mask = warp.ballot_sync(empty);
                if (empty_mask != 0) {
                    // Key not found
                    if (lane_id == 0) {
                        results[warp_id] = UINT32_MAX;
                    }
                    break;
                }
            }
        }
    }
}

// Build symbol table from crate symbols
__global__ void build_symbol_table_kernel(
    const Symbol* symbols,
    uint32_t num_symbols,
    uint32_t* hash_table,
    uint32_t table_size,
    uint32_t* collision_count
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize collision count
    if (tid == 0) {
        *collision_count = 0;
    }
    __syncthreads();
    
    // Process symbols in parallel
    if (tid < num_symbols) {
        const Symbol& sym = symbols[tid];
        uint32_t pos = murmur3_32(sym.name_hash) % table_size;
        uint32_t probes = 0;
        
        // Quadratic probing for better cache performance
        for (uint32_t i = 0; i < table_size; ++i) {
            uint32_t slot = (pos + i * i) % table_size;
            uint32_t old_val = atomicCAS(&hash_table[slot * 2], UINT32_MAX, sym.name_hash);
            
            if (old_val == UINT32_MAX) {
                // Successfully inserted
                hash_table[slot * 2 + 1] = tid;
                if (probes > 0) {
                    atomicAdd(collision_count, 1);
                }
                break;
            } else if (old_val == sym.name_hash) {
                // Duplicate - handle appropriately
                break;
            }
            probes++;
        }
    }
}

// Resolve symbol with visibility rules
__global__ void resolve_symbol_kernel(
    uint32_t query_hash,
    uint32_t requesting_crate,
    uint32_t requesting_module,
    const uint32_t* hash_table,
    const Symbol* symbols,
    uint32_t table_size,
    Symbol* result,
    bool* found
) {
    const uint32_t tid = threadIdx.x;
    const uint32_t pos = murmur3_32(query_hash) % table_size;
    
    __shared__ bool found_shared;
    __shared__ uint32_t best_match_idx;
    __shared__ uint32_t best_match_score;
    
    if (tid == 0) {
        found_shared = false;
        best_match_idx = UINT32_MAX;
        best_match_score = 0;
    }
    __syncthreads();
    
    // Search for symbol with visibility check
    for (uint32_t i = tid; i < table_size; i += blockDim.x) {
        uint32_t slot = (pos + i) % table_size;
        uint32_t key = hash_table[slot * 2];
        
        if (key == query_hash) {
            uint32_t sym_idx = hash_table[slot * 2 + 1];
            const Symbol& sym = symbols[sym_idx];
            
            // Check visibility rules
            bool visible = false;
            uint32_t score = 0;
            
            if (sym.visibility == 0) { // Public
                visible = true;
                score = 3;
            } else if (sym.visibility == 1 && sym.crate_id == requesting_crate) { // Crate-local
                visible = true;
                score = 2;
            } else if (sym.visibility == 2 && sym.module_id == requesting_module) { // Module-local
                visible = true;
                score = 1;
            }
            
            if (visible) {
                uint32_t old_score = atomicMax(&best_match_score, score);
                if (score > old_score) {
                    best_match_idx = sym_idx;
                    found_shared = true;
                }
            }
        } else if (key == UINT32_MAX) {
            break; // Empty slot, stop searching
        }
    }
    __syncthreads();
    
    // Write result
    if (tid == 0) {
        *found = found_shared;
        if (found_shared && best_match_idx != UINT32_MAX) {
            *result = symbols[best_match_idx];
        }
    }
}

// Host launchers
extern "C" void launch_symbol_table_builder(
    const Symbol* symbols,
    uint32_t num_symbols,
    uint32_t* hash_table,
    uint32_t table_size,
    uint32_t* collision_count
) {
    // Initialize hash table
    uint32_t threads = 256;
    uint32_t blocks = (table_size + threads - 1) / threads;
    
    uint32_t* keys = hash_table;
    uint32_t* values = hash_table + table_size;
    uint32_t* metadata = hash_table + table_size * 2;
    
    init_hash_table_kernel<<<blocks, threads>>>(
        keys, values, metadata, table_size
    );
    
    // Insert symbols
    blocks = (num_symbols + threads - 1) / threads;
    size_t shared_mem = sizeof(HashTableSharedMem);
    
    insert_symbols_kernel<<<blocks, threads, shared_mem>>>(
        symbols, num_symbols,
        keys, values, metadata,
        table_size, collision_count
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in symbol_table_builder: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void launch_symbol_resolver(
    uint32_t query_hash,
    const uint32_t* hash_table,
    const Symbol* symbols,
    uint32_t table_size,
    Symbol* result,
    bool* found
) {
    resolve_symbol_kernel<<<1, 256>>>(
        query_hash, 0, 0,  // Default crate/module for now
        hash_table, symbols, table_size,
        result, found
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in symbol_resolver: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void launch_batch_symbol_lookup(
    const uint32_t* query_hashes,
    uint32_t num_queries,
    const uint32_t* hash_table,
    uint32_t table_size,
    uint32_t* results
) {
    uint32_t threads = 256;
    uint32_t blocks = (num_queries * 32 + threads - 1) / threads;
    
    lookup_symbols_warp_kernel<<<blocks, threads>>>(
        query_hashes, num_queries,
        hash_table, hash_table + table_size,
        table_size, results
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in batch_symbol_lookup: %s\n", cudaGetErrorString(err));
    }
}

} // namespace rustg