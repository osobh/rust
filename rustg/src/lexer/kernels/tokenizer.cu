#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include "../../../include/gpu_types.h"
#include "../../../include/char_classifier.h"

namespace cg = cooperative_groups;

namespace rustg {

// Shared memory for warp cooperation
extern __shared__ char shared_mem[];

// Device function to check if character starts an identifier
__device__ inline bool is_identifier_start(CharClass cls) {
    return cls == CharClass::Letter;
}

// Device function to check if character continues an identifier
__device__ inline bool is_identifier_cont(CharClass cls) {
    return cls == CharClass::Letter || cls == CharClass::Digit;
}

// Structure for token boundary detection
struct TokenBoundary {
    uint32_t start;
    uint32_t end;
    TokenType type;
    bool valid;
};

// Warp-level token boundary detection
__device__ TokenBoundary detect_token_boundary_warp(
    const char* source,
    uint32_t pos,
    uint32_t source_len,
    cg::coalesced_group& warp
) {
    TokenBoundary result = {pos, pos, TokenType::Invalid, false};
    
    if (pos >= source_len) return result;
    
    unsigned char ch = static_cast<unsigned char>(source[pos]);
    CharClass cls = classify_char(ch);
    
    // Handle different token types
    switch (cls) {
        case CharClass::Whitespace:
            result.type = TokenType::Whitespace;
            result.end = pos + 1;
            // Extend whitespace using warp cooperation
            while (result.end < source_len && warp.any(true)) {
                if (warp.thread_rank() == 0 && result.end < source_len) {
                    CharClass next_cls = classify_char(source[result.end]);
                    if (next_cls == CharClass::Whitespace) {
                        result.end++;
                    } else {
                        break;
                    }
                }
                result.end = warp.shfl(result.end, 0);
            }
            result.valid = true;
            break;
            
        case CharClass::Newline:
            result.type = TokenType::Newline;
            result.end = pos + 1;
            result.valid = true;
            break;
            
        case CharClass::Letter:
            // Identifier or keyword
            result.type = TokenType::Identifier;
            result.end = pos + 1;
            // Extend identifier
            while (result.end < source_len) {
                CharClass next_cls = classify_char(source[result.end]);
                if (is_identifier_cont(next_cls)) {
                    result.end++;
                } else {
                    break;
                }
            }
            // Check for keywords (simplified for now)
            uint32_t len = result.end - result.start;
            if (len == 2 && source[pos] == 'f' && source[pos+1] == 'n') {
                result.type = TokenType::KeywordFn;
            } else if (len == 3 && source[pos] == 'l' && source[pos+1] == 'e' && source[pos+2] == 't') {
                result.type = TokenType::KeywordLet;
            }
            result.valid = true;
            break;
            
        case CharClass::Digit:
            result.type = TokenType::IntegerLiteral;
            result.end = pos + 1;
            // Extend number
            while (result.end < source_len) {
                CharClass next_cls = classify_char(source[result.end]);
                if (next_cls == CharClass::Digit) {
                    result.end++;
                } else if (source[result.end] == '.' && result.end + 1 < source_len) {
                    // Check for float
                    CharClass after_dot = classify_char(source[result.end + 1]);
                    if (after_dot == CharClass::Digit) {
                        result.type = TokenType::FloatLiteral;
                        result.end += 2;
                        // Continue reading digits
                        while (result.end < source_len && 
                               classify_char(source[result.end]) == CharClass::Digit) {
                            result.end++;
                        }
                    }
                    break;
                } else {
                    break;
                }
            }
            result.valid = true;
            break;
            
        case CharClass::Operator:
            // Single character operators for now
            switch (ch) {
                case '+': result.type = TokenType::Plus; break;
                case '-': result.type = TokenType::Minus; break;
                case '*': result.type = TokenType::Star; break;
                case '/': result.type = TokenType::Slash; break;
                case '%': result.type = TokenType::Percent; break;
                case '&': result.type = TokenType::Ampersand; break;
                case '|': result.type = TokenType::Pipe; break;
                case '^': result.type = TokenType::Caret; break;
                case '~': result.type = TokenType::Tilde; break;
                case '!': result.type = TokenType::Bang; break;
                case '=': result.type = TokenType::Equal; break;
                case '<': result.type = TokenType::Less; break;
                case '>': result.type = TokenType::Greater; break;
                default: result.type = TokenType::Invalid; break;
            }
            result.end = pos + 1;
            result.valid = (result.type != TokenType::Invalid);
            break;
            
        case CharClass::Delimiter:
            switch (ch) {
                case '(': result.type = TokenType::LeftParen; break;
                case ')': result.type = TokenType::RightParen; break;
                case '{': result.type = TokenType::LeftBrace; break;
                case '}': result.type = TokenType::RightBrace; break;
                case '[': result.type = TokenType::LeftBracket; break;
                case ']': result.type = TokenType::RightBracket; break;
                case ';': result.type = TokenType::Semicolon; break;
                case ',': result.type = TokenType::Comma; break;
                case '.': result.type = TokenType::Dot; break;
                case ':': result.type = TokenType::Colon; break;
                default: result.type = TokenType::Invalid; break;
            }
            result.end = pos + 1;
            // Check for multi-character tokens
            if (ch == ':' && pos + 1 < source_len && source[pos + 1] == ':') {
                result.type = TokenType::DoubleColon;
                result.end = pos + 2;
            }
            result.valid = (result.type != TokenType::Invalid);
            break;
            
        case CharClass::Quote:
            // String or character literal (simplified)
            if (ch == '"') {
                result.type = TokenType::StringLiteral;
                result.end = pos + 1;
                // Find closing quote
                while (result.end < source_len && source[result.end] != '"') {
                    if (source[result.end] == '\\' && result.end + 1 < source_len) {
                        result.end += 2; // Skip escape sequence
                    } else {
                        result.end++;
                    }
                }
                if (result.end < source_len) {
                    result.end++; // Include closing quote
                }
            } else if (ch == '\'') {
                result.type = TokenType::CharLiteral;
                result.end = pos + 1;
                // Simple char literal handling
                if (result.end < source_len && source[result.end] == '\\') {
                    result.end += 2; // Escape sequence
                } else if (result.end < source_len) {
                    result.end++; // Character
                }
                if (result.end < source_len && source[result.end] == '\'') {
                    result.end++; // Closing quote
                }
            }
            result.valid = true;
            break;
            
        default:
            result.valid = false;
            break;
    }
    
    return result;
}

// Main tokenizer kernel
__global__ void tokenize_kernel(
    const char* source,
    size_t source_len,
    Token* tokens,
    uint32_t* token_count,
    uint32_t max_tokens
) {
    // Get thread and warp information
    cg::thread_block block = cg::this_thread_block();
    cg::coalesced_group warp = cg::coalesced_threads();
    
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_threads = gridDim.x * blockDim.x;
    
    // Each thread processes a chunk of the source
    uint32_t chunk_size = (source_len + total_threads - 1) / total_threads;
    uint32_t start_pos = tid * chunk_size;
    
    if (start_pos >= source_len) return;
    
    uint32_t end_pos = min(start_pos + chunk_size, static_cast<uint32_t>(source_len));
    uint32_t pos = start_pos;
    
    // Local token buffer
    const uint32_t local_buffer_size = 32;
    Token local_tokens[local_buffer_size];
    uint32_t local_count = 0;
    
    // Track line and column
    uint32_t line = 1;
    uint32_t column = 1;
    
    // Tokenize the chunk
    while (pos < end_pos && local_count < local_buffer_size) {
        TokenBoundary boundary = detect_token_boundary_warp(source, pos, source_len, warp);
        
        if (boundary.valid) {
            // Skip whitespace tokens (optional)
            if (boundary.type != TokenType::Whitespace) {
                Token& token = local_tokens[local_count++];
                token.type = boundary.type;
                token.start_pos = boundary.start;
                token.length = boundary.end - boundary.start;
                token.line = line;
                token.column = column;
            }
            
            // Update line and column
            for (uint32_t i = boundary.start; i < boundary.end; ++i) {
                if (source[i] == '\n') {
                    line++;
                    column = 1;
                } else {
                    column++;
                }
            }
            
            pos = boundary.end;
        } else {
            // Skip invalid character
            pos++;
            column++;
        }
    }
    
    // Write local tokens to global memory atomically
    if (local_count > 0) {
        uint32_t global_offset = atomicAdd(token_count, local_count);
        
        if (global_offset + local_count <= max_tokens) {
            for (uint32_t i = 0; i < local_count; ++i) {
                tokens[global_offset + i] = local_tokens[i];
            }
        }
    }
}

// Host function to launch the tokenizer kernel
extern "C" void launch_tokenizer_kernel(
    const char* source,
    size_t source_len,
    Token* tokens,
    uint32_t* token_count,
    uint32_t max_tokens
) {
    // Initialize character classification table
    static bool initialized = false;
    if (!initialized) {
        initialize_char_class_table();
        initialized = true;
    }
    
    // Configure kernel launch parameters
    uint32_t threads_per_block = 256;
    uint32_t num_blocks = min(65535u, 
        static_cast<uint32_t>((source_len + threads_per_block - 1) / threads_per_block));
    
    // Calculate shared memory size
    size_t shared_mem_size = threads_per_block * 64; // 64 bytes per thread
    
    // Launch kernel
    tokenize_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        source, source_len, tokens, token_count, max_tokens
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in tokenize_kernel: %s\n", cudaGetErrorString(err));
    }
}

} // namespace rustg