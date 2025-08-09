#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include "../../../include/gpu_types.h"

namespace cg = cooperative_groups;

namespace rustg {

// Forward declarations
extern __device__ CharClass classify_char(unsigned char ch);

// Advanced Rust syntax states
enum class SyntaxState : uint8_t {
    Normal,
    Lifetime,
    Generic,
    Pattern,
    TypeAnnotation,
    MacroInvocation,
    Complete
};

// Pattern matching types
enum class PatternType : uint8_t {
    Literal,
    Identifier,
    Wildcard,
    Struct,
    Tuple,
    Enum,
    Range,
    Reference,
    Slice
};

// Shared memory for advanced syntax processing
struct AdvancedSyntaxSharedMem {
    char source_buffer[8192];           // Source code chunk
    uint32_t lifetime_positions[256];   // Lifetime annotation positions
    uint32_t generic_brackets[256];     // Generic bracket pairs
    uint32_t pattern_boundaries[256];   // Pattern boundaries
    uint32_t lifetime_count;
    uint32_t generic_count;
    uint32_t pattern_count;
    uint8_t bracket_stack[128];         // Stack for bracket matching
    uint32_t stack_top;
};

// Process lifetime annotations ('a, 'static, etc.)
__device__ uint32_t process_lifetime(
    const char* source,
    uint32_t pos,
    uint32_t max_pos,
    Token& token
) {
    if (pos >= max_pos || source[pos] != '\'') {
        return pos;
    }
    
    uint32_t start = pos;
    pos++; // Skip apostrophe
    
    // Check for 'static
    if (pos + 5 < max_pos && 
        source[pos] == 's' && source[pos+1] == 't' &&
        source[pos+2] == 'a' && source[pos+3] == 't' &&
        source[pos+4] == 'i' && source[pos+5] == 'c') {
        token.type = TokenType::Lifetime;
        token.start_pos = start;
        token.length = 7; // 'static
        return pos + 6;
    }
    
    // Check for regular lifetime 'a, 'foo, etc.
    if (pos < max_pos && (source[pos] == '_' || 
        (source[pos] >= 'a' && source[pos] <= 'z'))) {
        
        while (pos < max_pos && 
               (source[pos] == '_' || 
                (source[pos] >= 'a' && source[pos] <= 'z') ||
                (source[pos] >= 'A' && source[pos] <= 'Z') ||
                (source[pos] >= '0' && source[pos] <= '9'))) {
            pos++;
        }
        
        token.type = TokenType::Lifetime;
        token.start_pos = start;
        token.length = pos - start;
        return pos;
    }
    
    return start; // Not a lifetime
}

// Process generic parameters <T, U, const N: usize>
__device__ uint32_t process_generics(
    const char* source,
    uint32_t pos,
    uint32_t max_pos,
    Token* tokens,
    uint32_t& token_index,
    uint32_t max_tokens
) {
    if (pos >= max_pos || source[pos] != '<') {
        return pos;
    }
    
    uint32_t start = pos;
    pos++; // Skip '<'
    
    // Count angle brackets to handle nested generics
    uint32_t angle_count = 1;
    uint32_t param_start = pos;
    
    while (pos < max_pos && angle_count > 0 && token_index < max_tokens) {
        char ch = source[pos];
        
        if (ch == '<') {
            angle_count++;
        } else if (ch == '>') {
            angle_count--;
            if (angle_count == 0) {
                // Found matching closing bracket
                Token& token = tokens[token_index++];
                token.type = TokenType::Generic;
                token.start_pos = start;
                token.length = pos - start + 1;
                return pos + 1;
            }
        } else if (ch == ',' && angle_count == 1) {
            // Generic parameter separator at top level
            if (pos > param_start) {
                Token& token = tokens[token_index++];
                token.type = TokenType::Identifier; // Generic param name
                token.start_pos = param_start;
                token.length = pos - param_start;
            }
            param_start = pos + 1;
        } else if (ch == ':' && angle_count == 1) {
            // Trait bound or const generic
            if (pos > param_start) {
                Token& token = tokens[token_index++];
                token.type = TokenType::Identifier;
                token.start_pos = param_start;
                token.length = pos - param_start;
            }
            param_start = pos + 1;
        }
        
        pos++;
    }
    
    return start; // Unclosed generic
}

// Process pattern matching constructs
__device__ uint32_t process_pattern(
    const char* source,
    uint32_t pos,
    uint32_t max_pos,
    Token& token,
    PatternType& pattern_type
) {
    if (pos >= max_pos) return pos;
    
    uint32_t start = pos;
    char ch = source[pos];
    
    // Wildcard pattern '_'
    if (ch == '_' && (pos + 1 >= max_pos || 
        !(source[pos+1] >= 'a' && source[pos+1] <= 'z'))) {
        token.type = TokenType::Underscore;
        token.start_pos = start;
        token.length = 1;
        pattern_type = PatternType::Wildcard;
        return pos + 1;
    }
    
    // Range patterns 0..10, 'a'..='z'
    if (pos + 2 < max_pos && source[pos+1] == '.' && source[pos+2] == '.') {
        if (pos + 3 < max_pos && source[pos+3] == '=') {
            token.type = TokenType::DotDotEq;
            token.length = 3;
            pattern_type = PatternType::Range;
            return pos + 3;
        } else {
            token.type = TokenType::DotDot;
            token.length = 2;
            pattern_type = PatternType::Range;
            return pos + 2;
        }
    }
    
    // Reference patterns &x, &mut x
    if (ch == '&') {
        pos++;
        if (pos + 2 < max_pos && 
            source[pos] == 'm' && source[pos+1] == 'u' && 
            source[pos+2] == 't') {
            token.type = TokenType::RefMut;
            token.start_pos = start;
            token.length = 4;
            pattern_type = PatternType::Reference;
            return pos + 3;
        } else {
            token.type = TokenType::Ampersand;
            token.start_pos = start;
            token.length = 1;
            pattern_type = PatternType::Reference;
            return pos;
        }
    }
    
    // Tuple patterns (x, y, z)
    if (ch == '(') {
        uint32_t paren_count = 1;
        pos++;
        
        while (pos < max_pos && paren_count > 0) {
            if (source[pos] == '(') paren_count++;
            else if (source[pos] == ')') paren_count--;
            pos++;
        }
        
        token.type = TokenType::TuplePattern;
        token.start_pos = start;
        token.length = pos - start;
        pattern_type = PatternType::Tuple;
        return pos;
    }
    
    // Slice patterns [x, y, ..]
    if (ch == '[') {
        uint32_t bracket_count = 1;
        pos++;
        
        while (pos < max_pos && bracket_count > 0) {
            if (source[pos] == '[') bracket_count++;
            else if (source[pos] == ']') bracket_count--;
            pos++;
        }
        
        token.type = TokenType::SlicePattern;
        token.start_pos = start;
        token.length = pos - start;
        pattern_type = PatternType::Slice;
        return pos;
    }
    
    return start; // Not a pattern
}

// Main kernel for advanced Rust syntax
__global__ void process_advanced_syntax_kernel(
    const char* __restrict__ source,
    size_t source_len,
    Token* __restrict__ tokens,
    uint32_t* __restrict__ token_count,
    uint32_t max_tokens
) {
    extern __shared__ char shared_mem_raw[];
    AdvancedSyntaxSharedMem* shared = 
        reinterpret_cast<AdvancedSyntaxSharedMem*>(shared_mem_raw);
    
    const uint32_t tid = threadIdx.x;
    const uint32_t block_id = blockIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // Initialize shared memory
    if (tid == 0) {
        shared->lifetime_count = 0;
        shared->generic_count = 0;
        shared->pattern_count = 0;
        shared->stack_top = 0;
    }
    __syncthreads();
    
    // Calculate chunk for this block
    const uint32_t chunk_size = (source_len + gridDim.x - 1) / gridDim.x;
    const uint32_t start_pos = block_id * chunk_size;
    const uint32_t end_pos = min(start_pos + chunk_size, 
                                  static_cast<uint32_t>(source_len));
    
    if (start_pos >= source_len) return;
    
    // Load source chunk cooperatively
    uint32_t load_size = min(8192u, end_pos - start_pos);
    for (uint32_t i = tid; i < load_size; i += blockDim.x) {
        shared->source_buffer[i] = source[start_pos + i];
    }
    __syncthreads();
    
    // Process advanced syntax (8 chars per thread)
    const uint32_t chars_per_thread = 8;
    const uint32_t thread_start = tid * chars_per_thread;
    const uint32_t thread_end = min(thread_start + chars_per_thread, load_size);
    
    Token local_tokens[16];
    uint32_t local_token_count = 0;
    
    for (uint32_t pos = thread_start; pos < thread_end && local_token_count < 16;) {
        char ch = shared->source_buffer[pos];
        
        // Check for lifetime annotations
        if (ch == '\'' && pos + 1 < load_size) {
            char next = shared->source_buffer[pos + 1];
            // Not a char literal if followed by identifier char
            if ((next >= 'a' && next <= 'z') || next == '_' || next == 's') {
                Token& token = local_tokens[local_token_count];
                uint32_t new_pos = process_lifetime(
                    shared->source_buffer, pos, load_size, token);
                
                if (new_pos > pos) {
                    token.line = 1; // Simplified
                    token.column = pos + 1;
                    local_token_count++;
                    pos = new_pos;
                    continue;
                }
            }
        }
        
        // Check for generics
        if (ch == '<' && pos > 0) {
            char prev = shared->source_buffer[pos - 1];
            // Likely generic if preceded by identifier or ::>
            if ((prev >= 'A' && prev <= 'Z') || 
                (prev >= 'a' && prev <= 'z') || 
                prev == '>') {
                
                uint32_t new_pos = process_generics(
                    shared->source_buffer, pos, load_size,
                    local_tokens, local_token_count, 16);
                
                if (new_pos > pos) {
                    pos = new_pos;
                    continue;
                }
            }
        }
        
        // Check for pattern matching constructs
        if (ch == '_' || ch == '&' || ch == '(' || ch == '[' || 
            (ch == '.' && pos + 1 < load_size && shared->source_buffer[pos+1] == '.')) {
            
            Token& token = local_tokens[local_token_count];
            PatternType pattern_type;
            uint32_t new_pos = process_pattern(
                shared->source_buffer, pos, load_size, token, pattern_type);
            
            if (new_pos > pos) {
                token.line = 1;
                token.column = pos + 1;
                local_token_count++;
                pos = new_pos;
                continue;
            }
        }
        
        // Check for macro invocations (ident!)
        if (ch == '!' && pos > 0) {
            char prev = shared->source_buffer[pos - 1];
            if ((prev >= 'a' && prev <= 'z') || 
                (prev >= 'A' && prev <= 'Z') || 
                prev == '_') {
                
                Token& token = local_tokens[local_token_count++];
                token.type = TokenType::MacroBang;
                token.start_pos = start_pos + pos;
                token.length = 1;
                token.line = 1;
                token.column = pos + 1;
                pos++;
                continue;
            }
        }
        
        pos++;
    }
    
    // Cooperatively write tokens to global memory
    if (local_token_count > 0) {
        // Use warp-level reduction to find total tokens
        uint32_t warp_total = 0;
        for (int offset = 16; offset > 0; offset /= 2) {
            uint32_t other_count = warp.shfl_down(local_token_count, offset);
            if (lane_id + offset < 32) {
                warp_total += other_count;
            }
        }
        
        // Lane 0 reserves space for entire warp
        uint32_t write_offset = 0;
        if (lane_id == 0 && warp_total > 0) {
            write_offset = atomicAdd(token_count, warp_total);
        }
        
        // Broadcast offset to all lanes
        write_offset = warp.shfl(write_offset, 0);
        
        // Calculate per-thread offset
        uint32_t thread_offset = write_offset;
        for (int i = 0; i < lane_id; ++i) {
            thread_offset += warp.shfl(local_token_count, i);
        }
        
        // Write tokens
        if (thread_offset + local_token_count <= max_tokens) {
            for (uint32_t i = 0; i < local_token_count; ++i) {
                tokens[thread_offset + i] = local_tokens[i];
            }
        }
    }
}

// Host function to launch advanced syntax processor
extern "C" void launch_advanced_syntax_processor(
    const char* source,
    size_t source_len,
    Token* tokens,
    uint32_t* token_count,
    uint32_t max_tokens
) {
    uint32_t threads_per_block = 256;
    uint32_t num_blocks = min(65535u, 
        static_cast<uint32_t>((source_len + 8192 - 1) / 8192));
    
    size_t shared_mem_size = sizeof(AdvancedSyntaxSharedMem);
    
    process_advanced_syntax_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        source, source_len, tokens, token_count, max_tokens);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in process_advanced_syntax_kernel: %s\n", 
               cudaGetErrorString(err));
    }
}

} // namespace rustg