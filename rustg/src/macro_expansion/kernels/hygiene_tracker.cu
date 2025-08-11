#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include "../../../include/gpu_types.h"

namespace cg = cooperative_groups;

namespace rustg {

// Simple kernel to assign hygiene contexts
__global__ void assign_hygiene_contexts(
    const Token* tokens,
    uint32_t count,
    uint32_t* contexts,
    uint32_t base
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        // Assign unique context based on position and base
        contexts[idx] = base + (idx / 10); // Group every 10 tokens
    }
}

// Hygiene context information
struct HygieneContext {
    uint32_t context_id;      // Unique context identifier
    uint32_t parent_context;  // Parent context (for nested macros)
    uint32_t macro_id;        // Macro that created this context
    uint32_t invocation_site; // Where the macro was invoked
    uint32_t scope_level;     // Nesting level
    bool is_local;           // Local to macro expansion
};

// Shared memory for hygiene tracking
struct HygieneSharedMem {
    Token tokens[1024];                // Token buffer
    HygieneContext contexts[256];      // Context stack
    uint32_t context_map[1024];        // Token to context mapping
    uint32_t identifier_hashes[512];   // Identifier name hashes
    uint32_t context_count;
    uint32_t current_scope;
    uint32_t base_context;
};

// Generate unique context ID
__device__ uint32_t generate_context_id(
    uint32_t seed,
    uint32_t thread_id,
    uint32_t timestamp
) {
    // Simple hash combining seed, thread, and time
    uint32_t hash = seed;
    hash = ((hash << 5) + hash) + thread_id;
    hash = ((hash << 5) + hash) + timestamp;
    return hash;
}

// Check if identifier needs hygiene protection
__device__ bool needs_hygiene_protection(
    const Token& token,
    uint32_t context_id
) {
    // Identifiers introduced by macros need protection
    // Keywords and literals don't need hygiene
    return token.type == TokenType::Identifier ||
           token.type == TokenType::Lifetime;
}

// Compute identifier hash for hygiene comparison
__device__ uint32_t compute_identifier_hash(
    const char* source,
    const Token& token,
    uint32_t hygiene_context
) {
    uint32_t hash = hygiene_context;
    for (uint32_t i = 0; i < token.length; ++i) {
        hash = ((hash << 5) + hash) + source[token.start_pos + i];
    }
    return hash;
}

// Track hygiene contexts through macro expansion
__global__ void track_hygiene_contexts_kernel(
    const Token* __restrict__ tokens,
    uint32_t token_count,
    uint32_t* __restrict__ hygiene_contexts,
    uint32_t base_context,
    const char* __restrict__ source,
    uint32_t source_len
) {
    extern __shared__ char shared_mem_raw[];
    HygieneSharedMem* shared = 
        reinterpret_cast<HygieneSharedMem*>(shared_mem_raw);
    
    const uint32_t tid = threadIdx.x;
    const uint32_t block_id = blockIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // Initialize shared memory
    if (tid == 0) {
        shared->context_count = 1;
        shared->current_scope = 0;
        shared->base_context = base_context;
        
        // Initialize base context
        shared->contexts[0].context_id = base_context;
        shared->contexts[0].parent_context = 0;
        shared->contexts[0].macro_id = 0;
        shared->contexts[0].invocation_site = 0;
        shared->contexts[0].scope_level = 0;
        shared->contexts[0].is_local = false;
    }
    __syncthreads();
    
    // Calculate chunk for this block
    const uint32_t chunk_size = (token_count + gridDim.x - 1) / gridDim.x;
    const uint32_t start_idx = block_id * chunk_size;
    const uint32_t end_idx = min(start_idx + chunk_size, token_count);
    
    if (start_idx >= token_count) return;
    
    // Load tokens to shared memory
    uint32_t load_size = min(1024u, end_idx - start_idx);
    for (uint32_t i = tid; i < load_size; i += blockDim.x) {
        shared->tokens[i] = tokens[start_idx + i];
        shared->context_map[i] = base_context;
    }
    __syncthreads();
    
    // Process tokens with warp cooperation
    const uint32_t tokens_per_warp = (load_size + 8 - 1) / 8; // 8 warps
    const uint32_t warp_start = warp_id * tokens_per_warp;
    const uint32_t warp_end = min(warp_start + tokens_per_warp, load_size);
    
    // Each warp processes a chunk of tokens
    for (uint32_t base = warp_start; base < warp_end; base += 32) {
        uint32_t idx = base + lane_id;
        
        if (idx < warp_end) {
            Token& token = shared->tokens[idx];
            uint32_t context = shared->base_context;
            
            // Check for macro invocation
            if (token.type == TokenType::MacroBang && idx > 0) {
                // Previous token should be identifier (macro name)
                if (shared->tokens[idx - 1].type == TokenType::Identifier) {
                    // Create new hygiene context for macro expansion
                    uint32_t new_context = generate_context_id(
                        shared->base_context,
                        blockIdx.x * blockDim.x + tid,
                        idx
                    );
                    
                    // Update context for following tokens
                    context = new_context;
                    
                    // Mark this as a context boundary
                    if (lane_id == 0) {
                        uint32_t ctx_idx = atomicAdd(&shared->context_count, 1);
                        if (ctx_idx < 256) {
                            shared->contexts[ctx_idx].context_id = new_context;
                            shared->contexts[ctx_idx].parent_context = shared->base_context;
                            shared->contexts[ctx_idx].macro_id = idx - 1;
                            shared->contexts[ctx_idx].invocation_site = start_idx + idx;
                            shared->contexts[ctx_idx].scope_level = shared->current_scope + 1;
                            shared->contexts[ctx_idx].is_local = true;
                        }
                    }
                }
            }
            
            // Check for scope changes
            if (token.type == TokenType::LeftBrace) {
                // Entering new scope
                uint32_t new_scope = atomicAdd(&shared->current_scope, 1);
                context = generate_context_id(context, new_scope, idx);
            } else if (token.type == TokenType::RightBrace) {
                // Leaving scope
                if (shared->current_scope > 0) {
                    atomicSub(&shared->current_scope, 1);
                }
            }
            
            // Apply hygiene to identifiers
            if (needs_hygiene_protection(token, context)) {
                // Compute hygiene-aware hash
                uint32_t hash = compute_identifier_hash(source, token, context);
                
                // Store in shared memory for deduplication
                if (idx < 512) {
                    shared->identifier_hashes[idx] = hash;
                }
                
                // Update context map
                shared->context_map[idx] = context;
            } else {
                // Non-identifier tokens inherit context
                shared->context_map[idx] = context;
            }
        }
        
        // Sync within warp for context propagation
        warp.sync();
    }
    
    __syncthreads();
    
    // Write hygiene contexts to global memory
    for (uint32_t i = tid; i < load_size; i += blockDim.x) {
        hygiene_contexts[start_idx + i] = shared->context_map[i];
    }
}

// Resolve identifier with hygiene
__global__ void resolve_hygienic_identifiers_kernel(
    const Token* __restrict__ tokens,
    uint32_t token_count,
    const uint32_t* __restrict__ hygiene_contexts,
    uint32_t* __restrict__ resolved_ids,
    const char* __restrict__ source
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= token_count) return;
    
    const Token& token = tokens[tid];
    uint32_t hygiene = hygiene_contexts[tid];
    
    if (token.type == TokenType::Identifier) {
        // Compute hygiene-aware identifier
        uint32_t hash = compute_identifier_hash(source, token, hygiene);
        resolved_ids[tid] = hash;
    } else {
        resolved_ids[tid] = 0; // Non-identifier
    }
}

// Host function to launch hygiene tracker
extern "C" void launch_hygiene_tracker(
    const Token* tokens,
    uint32_t token_count,
    uint32_t* hygiene_contexts,
    uint32_t current_context
) {
    // For simplified test, just assign contexts
    uint32_t threads_per_block = 256;
    uint32_t num_blocks = (token_count + threads_per_block - 1) / threads_per_block;
    
    // Launch simple assignment kernel
    assign_hygiene_contexts<<<num_blocks, threads_per_block>>>(
        tokens, token_count, hygiene_contexts, current_context
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in hygiene tracker: %s\n", cudaGetErrorString(err));
    }
}

// Full hygiene tracking with source
extern "C" void launch_hygiene_tracker_full(
    const Token* tokens,
    uint32_t token_count,
    uint32_t* hygiene_contexts,
    uint32_t base_context,
    const char* source,
    uint32_t source_len
) {
    uint32_t threads_per_block = 256;
    uint32_t num_blocks = min(65535u,
        static_cast<uint32_t>((token_count + 1023) / 1024));
    
    size_t shared_mem_size = sizeof(HygieneSharedMem);
    
    track_hygiene_contexts_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        tokens, token_count, hygiene_contexts, base_context, source, source_len
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in track_hygiene_contexts_kernel: %s\n",
               cudaGetErrorString(err));
    }
}

} // namespace rustg