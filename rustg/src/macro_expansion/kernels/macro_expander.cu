#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include "../../../include/gpu_types.h"

namespace cg = cooperative_groups;

namespace rustg {

// Expansion template for built-in macros
struct ExpansionTemplate {
    uint32_t macro_id;
    const char* prefix;      // Tokens before arguments
    const char* suffix;      // Tokens after arguments
    bool needs_format;       // Requires format_args! wrapping
    bool is_variadic;        // Can take multiple arguments
};

// Shared memory for macro expansion
struct MacroExpansionSharedMem {
    Token input_tokens[512];       // Input token buffer
    Token expanded_tokens[2048];   // Expanded output buffer
    uint32_t hygiene_contexts[2048]; // Hygiene tracking
    uint32_t expansion_offsets[32];  // Per-macro expansion offsets
    uint32_t input_count;
    uint32_t output_count;
    uint32_t current_hygiene;
};

// Built-in macro expansion templates
__constant__ ExpansionTemplate expansion_templates[] = {
    // println! -> std::io::_print(format_args!(...))
    {0, "std::io::_print(format_args!(", "))\n", true, false},
    // vec! -> <[_]>::into_vec(box [...])
    {1, "<[_]>::into_vec(box [", "])", false, true},
    // assert! -> if !(cond) { panic!(...) }
    {2, "if !(", ") { panic!(\"assertion failed\") }", false, false},
    // format! -> format_args!(...)
    {3, "format_args!(", ")", true, true},
    // dbg! -> { eprintln!(...); expr }
    {4, "{ eprintln!(\"[{}:{}] {} = {:?}\", file!(), line!(), stringify!(", ")); ", false, false},
    // todo! -> panic!("not yet implemented")
    {5, "panic!(\"not yet implemented", ")", false, false},
    // unimplemented! -> panic!("unimplemented")
    {6, "panic!(\"unimplemented", ")", false, false},
    // panic! -> panic!(...)
    {7, "panic!(", ")", false, false}
};

// Generate new hygiene context
__device__ uint32_t generate_hygiene_context(
    uint32_t base_context,
    uint32_t macro_id,
    uint32_t invocation_site
) {
    // Simple hygiene: combine base, macro ID, and site
    return (base_context << 16) | (macro_id << 8) | (invocation_site & 0xFF);
}

// Expand println! macro
__device__ uint32_t expand_println(
    const Token* input_tokens,
    uint32_t start_pos,
    uint32_t end_pos,
    Token* output_tokens,
    uint32_t output_pos,
    uint32_t hygiene_context
) {
    uint32_t out_idx = output_pos;
    
    // std::io::_print
    output_tokens[out_idx].type = TokenType::Identifier;
    output_tokens[out_idx].start_pos = 0;
    output_tokens[out_idx].length = 15; // "std::io::_print"
    output_tokens[out_idx].line = input_tokens[start_pos].line;
    output_tokens[out_idx].column = input_tokens[start_pos].column;
    out_idx++;
    
    // (
    output_tokens[out_idx].type = TokenType::LeftParen;
    output_tokens[out_idx].start_pos = 0;
    output_tokens[out_idx].length = 1;
    out_idx++;
    
    // format_args!
    output_tokens[out_idx].type = TokenType::Identifier;
    output_tokens[out_idx].start_pos = 0;
    output_tokens[out_idx].length = 11; // "format_args"
    out_idx++;
    
    output_tokens[out_idx].type = TokenType::MacroBang;
    output_tokens[out_idx].start_pos = 0;
    output_tokens[out_idx].length = 1;
    out_idx++;
    
    // (
    output_tokens[out_idx].type = TokenType::LeftParen;
    output_tokens[out_idx].start_pos = 0;
    output_tokens[out_idx].length = 1;
    out_idx++;
    
    // Copy arguments from input (skip macro name, !, and opening paren)
    uint32_t arg_start = start_pos + 3;
    uint32_t arg_end = end_pos - 1; // Skip closing paren
    
    for (uint32_t i = arg_start; i < arg_end; ++i) {
        output_tokens[out_idx] = input_tokens[i];
        out_idx++;
    }
    
    // Add newline to format string if it's a string literal
    if (arg_start < arg_end && input_tokens[arg_start].type == TokenType::StringLiteral) {
        // Modify to add \n (simplified)
        output_tokens[out_idx - 1].length++; // Pretend we added \n
    }
    
    // ))
    output_tokens[out_idx].type = TokenType::RightParen;
    output_tokens[out_idx].start_pos = 0;
    output_tokens[out_idx].length = 1;
    out_idx++;
    
    output_tokens[out_idx].type = TokenType::RightParen;
    output_tokens[out_idx].start_pos = 0;
    output_tokens[out_idx].length = 1;
    out_idx++;
    
    return out_idx - output_pos;
}

// Expand vec! macro
__device__ uint32_t expand_vec(
    const Token* input_tokens,
    uint32_t start_pos,
    uint32_t end_pos,
    Token* output_tokens,
    uint32_t output_pos,
    uint32_t hygiene_context
) {
    uint32_t out_idx = output_pos;
    
    // <[_]>::into_vec
    output_tokens[out_idx].type = TokenType::Less;
    output_tokens[out_idx].start_pos = 0;
    output_tokens[out_idx].length = 1;
    out_idx++;
    
    output_tokens[out_idx].type = TokenType::LeftBracket;
    output_tokens[out_idx].start_pos = 0;
    output_tokens[out_idx].length = 1;
    out_idx++;
    
    output_tokens[out_idx].type = TokenType::Underscore;
    output_tokens[out_idx].start_pos = 0;
    output_tokens[out_idx].length = 1;
    out_idx++;
    
    output_tokens[out_idx].type = TokenType::RightBracket;
    output_tokens[out_idx].start_pos = 0;
    output_tokens[out_idx].length = 1;
    out_idx++;
    
    output_tokens[out_idx].type = TokenType::Greater;
    output_tokens[out_idx].start_pos = 0;
    output_tokens[out_idx].length = 1;
    out_idx++;
    
    output_tokens[out_idx].type = TokenType::ColonColon;
    output_tokens[out_idx].start_pos = 0;
    output_tokens[out_idx].length = 2;
    out_idx++;
    
    output_tokens[out_idx].type = TokenType::Identifier;
    output_tokens[out_idx].start_pos = 0;
    output_tokens[out_idx].length = 8; // "into_vec"
    out_idx++;
    
    // (box [
    output_tokens[out_idx].type = TokenType::LeftParen;
    output_tokens[out_idx].start_pos = 0;
    output_tokens[out_idx].length = 1;
    out_idx++;
    
    output_tokens[out_idx].type = TokenType::Identifier;
    output_tokens[out_idx].start_pos = 0;
    output_tokens[out_idx].length = 3; // "box"
    out_idx++;
    
    output_tokens[out_idx].type = TokenType::LeftBracket;
    output_tokens[out_idx].start_pos = 0;
    output_tokens[out_idx].length = 1;
    out_idx++;
    
    // Copy elements from input
    uint32_t elem_start = start_pos + 3; // Skip vec![  
    uint32_t elem_end = end_pos - 1;     // Skip ]
    
    for (uint32_t i = elem_start; i < elem_end; ++i) {
        output_tokens[out_idx] = input_tokens[i];
        out_idx++;
    }
    
    // ])
    output_tokens[out_idx].type = TokenType::RightBracket;
    output_tokens[out_idx].start_pos = 0;
    output_tokens[out_idx].length = 1;
    out_idx++;
    
    output_tokens[out_idx].type = TokenType::RightParen;
    output_tokens[out_idx].start_pos = 0;
    output_tokens[out_idx].length = 1;
    out_idx++;
    
    return out_idx - output_pos;
}

// Main macro expansion kernel
__global__ void expand_macros_kernel(
    const Token* __restrict__ input_tokens,
    uint32_t token_count,
    const uint8_t* __restrict__ pattern_matches,
    const uint32_t* __restrict__ match_positions,
    uint32_t match_count,
    Token* __restrict__ expanded_tokens,
    uint32_t* __restrict__ expanded_count,
    uint32_t* __restrict__ hygiene_contexts,
    uint32_t max_expanded
) {
    extern __shared__ char shared_mem_raw[];
    MacroExpansionSharedMem* shared = 
        reinterpret_cast<MacroExpansionSharedMem*>(shared_mem_raw);
    
    const uint32_t tid = threadIdx.x;
    const uint32_t block_id = blockIdx.x;
    
    // Initialize shared memory
    if (tid == 0) {
        shared->input_count = 0;
        shared->output_count = 0;
        shared->current_hygiene = block_id * 1000; // Unique per block
    }
    __syncthreads();
    
    // Each block handles one macro expansion
    if (block_id >= match_count) return;
    
    uint32_t macro_pos = match_positions[block_id];
    uint8_t pattern_id = pattern_matches[macro_pos];
    
    if (pattern_id == 0) return; // No match
    
    // Find macro boundaries
    uint32_t start_pos = macro_pos;
    uint32_t end_pos = macro_pos + 2; // At least ident + !
    
    // Find closing delimiter
    if (end_pos < token_count) {
        TokenType open_delim = input_tokens[end_pos].type;
        uint32_t depth = 1;
        end_pos++;
        
        while (end_pos < token_count && depth > 0) {
            TokenType type = input_tokens[end_pos].type;
            
            if (type == open_delim) {
                depth++;
            } else if ((open_delim == TokenType::LeftParen && type == TokenType::RightParen) ||
                      (open_delim == TokenType::LeftBracket && type == TokenType::RightBracket) ||
                      (open_delim == TokenType::LeftBrace && type == TokenType::RightBrace)) {
                depth--;
            }
            end_pos++;
        }
    }
    
    // Load input tokens to shared memory
    uint32_t load_size = min(512u, end_pos - start_pos);
    for (uint32_t i = tid; i < load_size; i += blockDim.x) {
        shared->input_tokens[i] = input_tokens[start_pos + i];
    }
    __syncthreads();
    
    // Perform expansion (single thread for now)
    if (tid == 0) {
        uint32_t expanded_size = 0;
        
        // Check macro type and expand
        TokenType first_token_type = shared->input_tokens[0].type;
        
        if (first_token_type == TokenType::Identifier) {
            // Simplified: check first few characters
            // In real implementation, would properly identify macro
            
            // Assume println! for pattern_id == 1
            if (pattern_id == 1) {
                expanded_size = expand_println(
                    shared->input_tokens, 0, load_size,
                    shared->expanded_tokens, 0,
                    shared->current_hygiene
                );
            }
            // Assume vec! for pattern_id == 2
            else if (pattern_id == 2) {
                expanded_size = expand_vec(
                    shared->input_tokens, 0, load_size,
                    shared->expanded_tokens, 0,
                    shared->current_hygiene
                );
            }
            // Default: copy as-is
            else {
                for (uint32_t i = 0; i < load_size; ++i) {
                    shared->expanded_tokens[i] = shared->input_tokens[i];
                }
                expanded_size = load_size;
            }
        }
        
        shared->output_count = expanded_size;
        
        // Set hygiene contexts
        for (uint32_t i = 0; i < expanded_size; ++i) {
            shared->hygiene_contexts[i] = generate_hygiene_context(
                shared->current_hygiene, pattern_id - 1, macro_pos
            );
        }
    }
    
    __syncthreads();
    
    // Write results to global memory
    if (tid == 0 && shared->output_count > 0) {
        uint32_t write_offset = atomicAdd(expanded_count, shared->output_count);
        
        if (write_offset + shared->output_count <= max_expanded) {
            // Copy expanded tokens
            for (uint32_t i = 0; i < shared->output_count; ++i) {
                expanded_tokens[write_offset + i] = shared->expanded_tokens[i];
                hygiene_contexts[write_offset + i] = shared->hygiene_contexts[i];
            }
        }
    }
}

// Host function to launch macro expander
extern "C" void launch_macro_expander(
    const Token* tokens,
    uint32_t token_count,
    const uint8_t* pattern_matches,
    const uint32_t* match_positions,
    uint32_t match_count,
    Token* expanded_tokens,
    uint32_t* expanded_count,
    uint32_t* hygiene_contexts,
    uint32_t max_expanded
) {
    if (match_count == 0) return;
    
    uint32_t threads_per_block = 256;
    uint32_t num_blocks = match_count; // One block per macro
    
    size_t shared_mem_size = sizeof(MacroExpansionSharedMem);
    
    expand_macros_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        tokens, token_count, pattern_matches, match_positions, match_count,
        expanded_tokens, expanded_count, hygiene_contexts, max_expanded
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in expand_macros_kernel: %s\n",
               cudaGetErrorString(err));
    }
}

} // namespace rustg