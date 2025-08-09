#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include "../../../include/gpu_types.h"

namespace cg = cooperative_groups;

namespace rustg {

// Repetition expansion state
struct RepetitionState {
    uint32_t current_iteration;
    uint32_t total_iterations;
    uint32_t pattern_start;
    uint32_t pattern_end;
    uint32_t output_start;
    uint32_t separator_type;
    bool needs_separator;
};

// Shared memory for repetition expansion
struct RepetitionSharedMem {
    Token pattern_buffer[512];      // Repetition pattern
    Token output_buffer[2048];      // Expanded output
    Token binding_values[512];      // Values for each binding
    uint32_t iteration_offsets[64]; // Start of each iteration
    uint32_t binding_offsets[64];   // Offsets for binding values
    uint32_t output_count;
    uint32_t pattern_size;
    uint32_t num_iterations;
};

// Expand $()* repetition pattern
__device__ uint32_t expand_star_repetition(
    const Token* pattern,
    uint32_t pattern_start,
    uint32_t pattern_end,
    const Token* binding_values,
    uint32_t num_values,
    Token* output,
    uint32_t output_start,
    TokenType separator,
    cg::thread_block_tile<32> warp
) {
    const uint32_t lane_id = warp.thread_rank();
    uint32_t output_pos = output_start;
    
    // Process each repetition
    for (uint32_t iter = 0; iter < num_values; ++iter) {
        // Add separator if needed (except first iteration)
        if (iter > 0 && separator != TokenType::Unknown) {
            if (lane_id == 0) {
                output[output_pos].type = separator;
                output[output_pos].start_pos = 0;
                output[output_pos].length = 1;
                output[output_pos].line = 1;
                output[output_pos].column = 1;
            }
            output_pos++;
            warp.sync();
        }
        
        // Expand pattern for this iteration
        uint32_t pattern_idx = pattern_start;
        
        while (pattern_idx < pattern_end) {
            // Each lane processes different positions
            uint32_t local_idx = pattern_idx + lane_id;
            
            if (local_idx < pattern_end) {
                const Token& p_token = pattern[local_idx];
                
                // Check if this is a binding reference
                if (p_token.type == TokenType::Dollar) {
                    // Substitute with binding value for this iteration
                    if (iter < num_values) {
                        output[output_pos + lane_id] = binding_values[iter];
                    }
                } else {
                    // Copy literal token
                    output[output_pos + lane_id] = p_token;
                }
            }
            
            warp.sync();
            
            // Calculate how many tokens were processed
            uint32_t processed = min(32u, pattern_end - pattern_idx);
            output_pos += processed;
            pattern_idx += processed;
        }
    }
    
    return output_pos - output_start;
}

// Expand $()+ repetition pattern (one or more)
__device__ uint32_t expand_plus_repetition(
    const Token* pattern,
    uint32_t pattern_start,
    uint32_t pattern_end,
    const Token* binding_values,
    uint32_t num_values,
    Token* output,
    uint32_t output_start,
    TokenType separator,
    cg::thread_block_tile<32> warp
) {
    // Plus is same as star but requires at least one value
    if (num_values == 0) {
        return 0; // Error: + requires at least one match
    }
    
    return expand_star_repetition(
        pattern, pattern_start, pattern_end,
        binding_values, num_values,
        output, output_start, separator, warp
    );
}

// Process nested repetitions
__device__ void process_nested_repetitions(
    const Token* pattern,
    uint32_t pattern_size,
    Token* output,
    uint32_t& output_size,
    uint32_t depth,
    cg::thread_block_tile<32> warp
) {
    const uint32_t lane_id = warp.thread_rank();
    
    // Find repetition markers in parallel
    uint32_t pos = lane_id;
    
    while (pos < pattern_size) {
        bool found_rep = false;
        uint32_t rep_start = 0;
        uint32_t rep_end = 0;
        
        // Check for $( pattern
        if (pos + 1 < pattern_size &&
            pattern[pos].type == TokenType::Dollar &&
            pattern[pos + 1].type == TokenType::LeftParen) {
            
            found_rep = true;
            rep_start = pos;
            
            // Find matching )* or )+
            uint32_t paren_depth = 1;
            uint32_t search_pos = pos + 2;
            
            while (search_pos < pattern_size && paren_depth > 0) {
                if (pattern[search_pos].type == TokenType::LeftParen) {
                    paren_depth++;
                } else if (pattern[search_pos].type == TokenType::RightParen) {
                    paren_depth--;
                    if (paren_depth == 0) {
                        // Check for * or +
                        if (search_pos + 1 < pattern_size) {
                            TokenType next = pattern[search_pos + 1].type;
                            if (next == TokenType::Star || next == TokenType::Plus) {
                                rep_end = search_pos + 2;
                            }
                        }
                    }
                }
                search_pos++;
            }
        }
        
        // Vote on finding repetition
        uint32_t found_mask = warp.ballot(found_rep);
        
        if (found_mask != 0) {
            // Leader processes the repetition
            uint32_t leader = __ffs(found_mask) - 1;
            
            if (lane_id == leader) {
                // Process this repetition
                // (Simplified: copy pattern for now)
                for (uint32_t i = rep_start; i < rep_end && output_size < 2048; ++i) {
                    output[output_size++] = pattern[i];
                }
            }
        }
        
        pos += 32; // Move to next warp-sized chunk
        warp.sync();
    }
}

// Main repetition expansion kernel
__global__ void expand_repetitions_kernel(
    const Token* __restrict__ pattern_tokens,
    uint32_t pattern_count,
    const Token* __restrict__ binding_values,
    uint32_t* __restrict__ binding_counts,
    uint32_t num_bindings,
    Token* __restrict__ output_tokens,
    uint32_t* __restrict__ output_count,
    uint32_t max_output
) {
    extern __shared__ char shared_mem_raw[];
    RepetitionSharedMem* shared = 
        reinterpret_cast<RepetitionSharedMem*>(shared_mem_raw);
    
    const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // Initialize shared memory
    if (tid == 0) {
        shared->output_count = 0;
        shared->pattern_size = min(512u, pattern_count);
        shared->num_iterations = 0;
    }
    __syncthreads();
    
    // Load pattern to shared memory
    for (uint32_t i = tid; i < shared->pattern_size; i += blockDim.x) {
        shared->pattern_buffer[i] = pattern_tokens[i];
    }
    
    // Load binding values
    uint32_t binding_load = min(512u, num_bindings * 10); // Assume avg 10 tokens per binding
    for (uint32_t i = tid; i < binding_load; i += blockDim.x) {
        shared->binding_values[i] = binding_values[i];
    }
    __syncthreads();
    
    // Each warp handles different repetition patterns
    if (warp_id < 4) {
        uint32_t chunk_size = shared->pattern_size / 4;
        uint32_t chunk_start = warp_id * chunk_size;
        uint32_t chunk_end = min(chunk_start + chunk_size, shared->pattern_size);
        
        // Search for repetition patterns
        for (uint32_t pos = chunk_start + lane_id; pos < chunk_end; pos += 32) {
            if (pos + 2 < shared->pattern_size &&
                shared->pattern_buffer[pos].type == TokenType::Dollar &&
                shared->pattern_buffer[pos + 1].type == TokenType::LeftParen) {
                
                // Found start of repetition
                uint32_t rep_end = pos + 2;
                uint32_t depth = 1;
                
                // Find end of repetition
                while (rep_end < shared->pattern_size && depth > 0) {
                    if (shared->pattern_buffer[rep_end].type == TokenType::LeftParen) {
                        depth++;
                    } else if (shared->pattern_buffer[rep_end].type == TokenType::RightParen) {
                        depth--;
                    }
                    rep_end++;
                }
                
                // Check for * or +
                if (rep_end < shared->pattern_size) {
                    TokenType rep_type = shared->pattern_buffer[rep_end].type;
                    
                    if (rep_type == TokenType::Star || rep_type == TokenType::Plus) {
                        // Expand this repetition
                        uint32_t expanded_size = 0;
                        
                        if (rep_type == TokenType::Star) {
                            expanded_size = expand_star_repetition(
                                shared->pattern_buffer, pos + 2, rep_end - 1,
                                shared->binding_values, 3, // Simplified: 3 iterations
                                shared->output_buffer, shared->output_count,
                                TokenType::Comma, warp
                            );
                        } else {
                            expanded_size = expand_plus_repetition(
                                shared->pattern_buffer, pos + 2, rep_end - 1,
                                shared->binding_values, 3, // Simplified: 3 iterations
                                shared->output_buffer, shared->output_count,
                                TokenType::Comma, warp
                            );
                        }
                        
                        if (lane_id == 0) {
                            atomicAdd(&shared->output_count, expanded_size);
                        }
                    }
                }
            }
        }
    }
    
    __syncthreads();
    
    // Process nested repetitions (warps 4-7)
    if (warp_id >= 4 && warp_id < 8) {
        uint32_t depth = warp_id - 4;
        process_nested_repetitions(
            shared->pattern_buffer, shared->pattern_size,
            shared->output_buffer, shared->output_count,
            depth, warp
        );
    }
    
    __syncthreads();
    
    // Write results to global memory
    if (tid == 0) {
        *output_count = min(shared->output_count, max_output);
    }
    
    // Copy output tokens
    for (uint32_t i = tid; i < shared->output_count && i < max_output; i += blockDim.x) {
        output_tokens[i] = shared->output_buffer[i];
    }
}

// Specialized kernel for vec! macro repetition
__global__ void expand_vec_repetition_kernel(
    const Token* __restrict__ elements,
    uint32_t element_count,
    Token* __restrict__ output,
    uint32_t* __restrict__ output_count,
    uint32_t max_output
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // vec![elem1, elem2, elem3] -> [elem1, elem2, elem3]
    // Already expanded in previous kernel, just format
    
    if (tid == 0) {
        uint32_t out_idx = 0;
        
        // Opening bracket
        output[out_idx].type = TokenType::LeftBracket;
        output[out_idx].start_pos = 0;
        output[out_idx].length = 1;
        out_idx++;
        
        // Copy elements
        for (uint32_t i = 0; i < element_count && out_idx < max_output - 1; ++i) {
            if (i > 0) {
                // Add comma separator
                output[out_idx].type = TokenType::Comma;
                output[out_idx].start_pos = 0;
                output[out_idx].length = 1;
                out_idx++;
            }
            
            output[out_idx] = elements[i];
            out_idx++;
        }
        
        // Closing bracket
        output[out_idx].type = TokenType::RightBracket;
        output[out_idx].start_pos = 0;
        output[out_idx].length = 1;
        out_idx++;
        
        *output_count = out_idx;
    }
}

// Host function to launch repetition expander
extern "C" void launch_repetition_expander(
    const Token* pattern_tokens,
    uint32_t pattern_count,
    const Token* binding_values,
    uint32_t* binding_counts,
    uint32_t num_bindings,
    Token* output_tokens,
    uint32_t* output_count,
    uint32_t max_output
) {
    uint32_t threads_per_block = 256; // 8 warps
    uint32_t num_blocks = 1;
    
    size_t shared_mem_size = sizeof(RepetitionSharedMem);
    
    expand_repetitions_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        pattern_tokens, pattern_count,
        binding_values, binding_counts, num_bindings,
        output_tokens, output_count, max_output
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in expand_repetitions_kernel: %s\n",
               cudaGetErrorString(err));
    }
}

} // namespace rustg