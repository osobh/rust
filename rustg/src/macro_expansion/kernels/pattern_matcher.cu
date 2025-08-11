#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include "../../../include/gpu_types.h"

namespace cg = cooperative_groups;

namespace rustg {

// Macro pattern types
enum class MacroType : uint8_t {
    BuiltIn,      // println!, vec!, assert!, etc.
    Derive,       // #[derive(...)]
    Custom,       // User-defined macros
    Procedural    // Not supported in Phase 2
};

// FragmentType is now defined in gpu_types.h

// Macro pattern node for matching
struct MacroPattern {
    uint32_t pattern_id;
    MacroType macro_type;
    uint32_t token_start;
    uint32_t token_end;
    uint32_t fragment_count;
    FragmentType fragments[16];  // Max 16 fragments per pattern
    uint32_t repetition_start;   // For $(...)* or $(...)+ patterns
    uint32_t repetition_end;
    bool is_variadic;            // Has repetition
};

// Shared memory for pattern matching
struct PatternMatchSharedMem {
    Token tokens[1024];           // Token buffer
    MacroPattern patterns[32];    // Known macro patterns
    uint8_t match_flags[1024];    // Per-token match flags
    uint32_t match_positions[256]; // Match start positions
    uint32_t pattern_count;
    uint32_t match_count;
};

// Built-in macro patterns (simplified)
__constant__ MacroPattern builtin_patterns[] = {
    // println! macro
    {0, MacroType::BuiltIn, 0, 0, 1, {FragmentType::Expr}, 0, 0, false},
    // vec! macro
    {1, MacroType::BuiltIn, 0, 0, 1, {FragmentType::Expr}, 0, 0, true},
    // assert! macro
    {2, MacroType::BuiltIn, 0, 0, 1, {FragmentType::Expr}, 0, 0, false},
    // format! macro
    {3, MacroType::BuiltIn, 0, 0, 2, {FragmentType::Literal, FragmentType::Expr}, 0, 0, true},
    // dbg! macro
    {4, MacroType::BuiltIn, 0, 0, 1, {FragmentType::Expr}, 0, 0, false},
    // todo! macro
    {5, MacroType::BuiltIn, 0, 0, 0, {}, 0, 0, false},
    // unimplemented! macro
    {6, MacroType::BuiltIn, 0, 0, 0, {}, 0, 0, false},
    // panic! macro
    {7, MacroType::BuiltIn, 0, 0, 1, {FragmentType::Literal}, 0, 0, false}
};

// Check if token matches a macro invocation pattern
__device__ bool is_macro_invocation(
    const Token* tokens,
    uint32_t pos,
    uint32_t token_count
) {
    // Check for identifier followed by !
    if (pos + 1 >= token_count) return false;
    
    return tokens[pos].type == TokenType::Identifier &&
           tokens[pos + 1].type == TokenType::MacroBang;
}

// Get macro name hash for quick lookup
__device__ uint32_t get_macro_name_hash(
    const char* source,
    const Token& token
) {
    uint32_t hash = 5381;
    for (uint32_t i = 0; i < token.length; ++i) {
        hash = ((hash << 5) + hash) + source[token.start_pos + i];
    }
    return hash;
}

// Match a specific built-in macro
__device__ MacroType identify_builtin_macro(
    const char* source,
    const Token& token
) {
    // Common macro name hashes (precomputed)
    const uint32_t PRINTLN_HASH = 0x7c9bbd73;
    const uint32_t VEC_HASH = 0xb8860b73;
    const uint32_t ASSERT_HASH = 0x3c5f0e4d;
    const uint32_t FORMAT_HASH = 0x8c1fc4ed;
    const uint32_t DBG_HASH = 0xb8820193;
    
    uint32_t hash = get_macro_name_hash(source, token);
    
    switch (hash) {
        case PRINTLN_HASH: return MacroType::BuiltIn;
        case VEC_HASH: return MacroType::BuiltIn;
        case ASSERT_HASH: return MacroType::BuiltIn;
        case FORMAT_HASH: return MacroType::BuiltIn;
        case DBG_HASH: return MacroType::BuiltIn;
        default: return MacroType::Custom;
    }
}

// Match macro arguments pattern
__device__ bool match_macro_arguments(
    const Token* tokens,
    uint32_t start_pos,
    uint32_t end_pos,
    const MacroPattern& pattern
) {
    // Find opening parenthesis or bracket
    uint32_t pos = start_pos + 2; // Skip ident and !
    if (pos >= end_pos) return false;
    
    TokenType open_delim = tokens[pos].type;
    if (open_delim != TokenType::LeftParen &&
        open_delim != TokenType::LeftBracket &&
        open_delim != TokenType::LeftBrace) {
        return false;
    }
    
    // Simple argument matching (can be enhanced)
    // For now, just check if arguments exist
    pos++;
    
    uint32_t arg_count = 0;
    uint32_t depth = 1;
    
    while (pos < end_pos && depth > 0) {
        TokenType type = tokens[pos].type;
        
        if (type == TokenType::LeftParen || 
            type == TokenType::LeftBracket ||
            type == TokenType::LeftBrace) {
            depth++;
        } else if (type == TokenType::RightParen ||
                   type == TokenType::RightBracket ||
                   type == TokenType::RightBrace) {
            depth--;
        } else if (type == TokenType::Comma && depth == 1) {
            arg_count++;
        }
        
        pos++;
    }
    
    // Check if we have at least one argument if pattern requires it
    if (pattern.fragment_count > 0 && arg_count == 0) {
        return pos > start_pos + 3; // At least some tokens between delimiters
    }
    
    return true;
}

// Main pattern matching kernel
__global__ void match_macro_patterns_kernel(
    const Token* __restrict__ tokens,
    uint32_t token_count,
    const char* __restrict__ source,
    uint32_t source_len,
    uint8_t* __restrict__ pattern_matches,
    uint32_t* __restrict__ match_positions,
    uint32_t* __restrict__ match_count,
    uint32_t max_matches
) {
    extern __shared__ char shared_mem_raw[];
    PatternMatchSharedMem* shared = 
        reinterpret_cast<PatternMatchSharedMem*>(shared_mem_raw);
    
    const uint32_t tid = threadIdx.x;
    const uint32_t block_id = blockIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // Initialize shared memory
    if (tid == 0) {
        shared->match_count = 0;
        shared->pattern_count = 8; // Number of built-in patterns
    }
    __syncthreads();
    
    // Load built-in patterns to shared memory
    if (tid < 8) {
        shared->patterns[tid] = builtin_patterns[tid];
    }
    __syncthreads();
    
    // Calculate chunk for this block
    const uint32_t chunk_size = (token_count + gridDim.x - 1) / gridDim.x;
    const uint32_t start_idx = block_id * chunk_size;
    const uint32_t end_idx = min(start_idx + chunk_size, token_count);
    
    if (start_idx >= token_count) return;
    
    // Load tokens to shared memory cooperatively
    uint32_t load_size = min(1024u, end_idx - start_idx);
    for (uint32_t i = tid; i < load_size; i += blockDim.x) {
        shared->tokens[i] = tokens[start_idx + i];
        shared->match_flags[i] = 0;
    }
    __syncthreads();
    
    // Pattern matching with warp cooperation
    const uint32_t tokens_per_thread = (load_size + blockDim.x - 1) / blockDim.x;
    const uint32_t thread_start = tid * tokens_per_thread;
    const uint32_t thread_end = min(thread_start + tokens_per_thread, load_size);
    
    for (uint32_t i = thread_start; i < thread_end; ++i) {
        // Check for macro invocation pattern
        if (is_macro_invocation(shared->tokens, i, load_size)) {
            // Identify macro type
            MacroType macro_type = identify_builtin_macro(source, shared->tokens[i]);
            
            if (macro_type == MacroType::BuiltIn) {
                // Find matching pattern
                for (uint32_t p = 0; p < shared->pattern_count; ++p) {
                    if (match_macro_arguments(shared->tokens, i, load_size,
                                            shared->patterns[p])) {
                        // Mark as matched
                        shared->match_flags[i] = p + 1; // Pattern ID + 1
                        
                        // Add to match positions atomically
                        uint32_t match_idx = atomicAdd(&shared->match_count, 1);
                        if (match_idx < 256) {
                            shared->match_positions[match_idx] = start_idx + i;
                        }
                        break;
                    }
                }
            }
        }
    }
    
    __syncthreads();
    
    // Write results to global memory (single thread)
    if (tid == 0 && shared->match_count > 0) {
        uint32_t global_offset = atomicAdd(match_count, shared->match_count);
        
        if (global_offset + shared->match_count <= max_matches) {
            // Copy match positions
            for (uint32_t i = 0; i < shared->match_count && i < 256; ++i) {
                match_positions[global_offset + i] = shared->match_positions[i];
            }
            
            // Copy pattern matches
            for (uint32_t i = 0; i < load_size; ++i) {
                if (shared->match_flags[i] > 0) {
                    pattern_matches[start_idx + i] = shared->match_flags[i];
                }
            }
        }
    }
}

// Host function to launch pattern matcher
extern "C" void launch_macro_pattern_matcher(
    const Token* tokens,
    uint32_t token_count,
    const char* source,
    uint32_t source_len,
    uint8_t* pattern_matches,
    uint32_t* match_positions,
    uint32_t* match_count,
    uint32_t max_matches
) {
    uint32_t threads_per_block = 256;
    uint32_t num_blocks = min(65535u,
        static_cast<uint32_t>((token_count + 1023) / 1024));
    
    size_t shared_mem_size = sizeof(PatternMatchSharedMem);
    
    match_macro_patterns_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        tokens, token_count, source, source_len,
        pattern_matches, match_positions, match_count, max_matches
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in match_macro_patterns_kernel: %s\n",
               cudaGetErrorString(err));
    }
}

} // namespace rustg