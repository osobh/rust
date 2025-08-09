#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include "../../../include/gpu_types.h"

namespace cg = cooperative_groups;

namespace rustg {

// Forward declarations
extern __device__ CharClass classify_char(unsigned char ch);

// String literal states for finite state machine
enum class StringState : uint8_t {
    Normal,
    Escape,
    UnicodeEscape,
    RawString,
    ByteString,
    Complete,
    Error
};

// Comment states
enum class CommentState : uint8_t {
    None,
    LineComment,
    BlockComment,
    BlockCommentStar,  // Seen '*' in block comment, might be end
    DocComment,
    Complete
};

// Shared memory for string/comment processing
struct StringCommentSharedMem {
    char buffer[8192];              // Source buffer
    uint8_t escape_buffer[1024];    // Escape sequence buffer
    uint32_t string_boundaries[256]; // String start/end positions
    uint32_t comment_boundaries[256]; // Comment boundaries
    uint32_t string_count;
    uint32_t comment_count;
};

// Process escape sequences in strings (cooperative)
__device__ uint32_t process_escape_sequence(
    const char* source,
    uint32_t pos,
    uint32_t max_pos,
    char* output,
    uint32_t& output_pos
) {
    if (pos >= max_pos || source[pos] != '\\') {
        return pos;
    }
    
    pos++; // Skip backslash
    if (pos >= max_pos) {
        return pos;
    }
    
    char ch = source[pos];
    switch (ch) {
        case 'n':  output[output_pos++] = '\n'; break;
        case 't':  output[output_pos++] = '\t'; break;
        case 'r':  output[output_pos++] = '\r'; break;
        case '\\': output[output_pos++] = '\\'; break;
        case '\'': output[output_pos++] = '\''; break;
        case '"':  output[output_pos++] = '"';  break;
        case '0':  output[output_pos++] = '\0'; break;
        
        case 'x': {
            // Hex escape \xNN
            pos++;
            if (pos + 1 < max_pos) {
                char high = source[pos];
                char low = source[pos + 1];
                uint8_t value = 0;
                
                // Convert hex digits
                if (high >= '0' && high <= '9') value = (high - '0') << 4;
                else if (high >= 'a' && high <= 'f') value = (high - 'a' + 10) << 4;
                else if (high >= 'A' && high <= 'F') value = (high - 'A' + 10) << 4;
                
                if (low >= '0' && low <= '9') value |= (low - '0');
                else if (low >= 'a' && low <= 'f') value |= (low - 'a' + 10);
                else if (low >= 'A' && low <= 'F') value |= (low - 'A' + 10);
                
                output[output_pos++] = static_cast<char>(value);
                pos += 2;
            }
            break;
        }
        
        case 'u': {
            // Unicode escape \u{NNNN}
            pos++;
            if (pos < max_pos && source[pos] == '{') {
                pos++; // Skip '{'
                uint32_t unicode_value = 0;
                while (pos < max_pos && source[pos] != '}') {
                    char digit = source[pos];
                    if (digit >= '0' && digit <= '9') {
                        unicode_value = unicode_value * 16 + (digit - '0');
                    } else if (digit >= 'a' && digit <= 'f') {
                        unicode_value = unicode_value * 16 + (digit - 'a' + 10);
                    } else if (digit >= 'A' && digit <= 'F') {
                        unicode_value = unicode_value * 16 + (digit - 'A' + 10);
                    }
                    pos++;
                }
                if (pos < max_pos && source[pos] == '}') {
                    pos++; // Skip '}'
                }
                
                // Simplified: just store low byte for now
                output[output_pos++] = static_cast<char>(unicode_value & 0xFF);
            }
            break;
        }
        
        default:
            // Unknown escape, keep as-is
            output[output_pos++] = ch;
            break;
    }
    
    return pos + 1;
}

// Kernel for string literal processing
__global__ void process_string_literals_kernel(
    const char* __restrict__ source,
    size_t source_len,
    Token* __restrict__ tokens,
    uint32_t* __restrict__ token_count,
    uint32_t max_tokens
) {
    extern __shared__ char shared_mem_raw[];
    StringCommentSharedMem* shared = reinterpret_cast<StringCommentSharedMem*>(shared_mem_raw);
    
    const uint32_t tid = threadIdx.x;
    const uint32_t block_id = blockIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    
    // Initialize shared memory
    if (tid == 0) {
        shared->string_count = 0;
        shared->comment_count = 0;
    }
    __syncthreads();
    
    // Calculate chunk for this block
    const uint32_t chunk_size = (source_len + gridDim.x - 1) / gridDim.x;
    const uint32_t start_pos = block_id * chunk_size;
    const uint32_t end_pos = min(start_pos + chunk_size, static_cast<uint32_t>(source_len));
    
    if (start_pos >= source_len) return;
    
    // Load source chunk into shared memory
    uint32_t load_size = min(8192u, end_pos - start_pos);
    for (uint32_t i = tid; i < load_size; i += blockDim.x) {
        shared->buffer[i] = source[start_pos + i];
    }
    __syncthreads();
    
    // Process strings and comments (simplified: one thread for now)
    if (tid == 0) {
        uint32_t pos = 0;
        uint32_t local_token_count = 0;
        Token local_tokens[128];
        
        while (pos < load_size && local_token_count < 128) {
            char ch = shared->buffer[pos];
            char next_ch = (pos + 1 < load_size) ? shared->buffer[pos + 1] : '\0';
            
            // Check for string literals
            if (ch == '"') {
                Token& token = local_tokens[local_token_count++];
                token.type = TokenType::StringLiteral;
                token.start_pos = start_pos + pos;
                token.line = 1; // Simplified
                token.column = pos + 1;
                
                pos++; // Skip opening quote
                uint32_t string_start = pos;
                StringState state = StringState::Normal;
                
                while (pos < load_size && state != StringState::Complete) {
                    ch = shared->buffer[pos];
                    
                    switch (state) {
                        case StringState::Normal:
                            if (ch == '"') {
                                state = StringState::Complete;
                            } else if (ch == '\\') {
                                state = StringState::Escape;
                            }
                            break;
                            
                        case StringState::Escape:
                            // Skip escape sequence
                            state = StringState::Normal;
                            break;
                    }
                    
                    if (state != StringState::Complete) {
                        pos++;
                    }
                }
                
                if (state == StringState::Complete) {
                    pos++; // Skip closing quote
                }
                
                token.length = (start_pos + pos) - token.start_pos;
            }
            // Check for char literals
            else if (ch == '\'') {
                Token& token = local_tokens[local_token_count++];
                token.type = TokenType::CharLiteral;
                token.start_pos = start_pos + pos;
                token.line = 1;
                token.column = pos + 1;
                
                pos++; // Skip opening quote
                
                if (pos < load_size && shared->buffer[pos] == '\\') {
                    pos += 2; // Skip escape sequence
                } else {
                    pos++; // Skip character
                }
                
                if (pos < load_size && shared->buffer[pos] == '\'') {
                    pos++; // Skip closing quote
                }
                
                token.length = (start_pos + pos) - token.start_pos;
            }
            // Check for line comments
            else if (ch == '/' && next_ch == '/') {
                Token& token = local_tokens[local_token_count++];
                
                // Check if it's a doc comment
                if (pos + 2 < load_size && shared->buffer[pos + 2] == '/') {
                    token.type = TokenType::BlockComment; // Using as doc comment
                } else {
                    token.type = TokenType::LineComment;
                }
                
                token.start_pos = start_pos + pos;
                token.line = 1;
                token.column = pos + 1;
                
                // Skip to end of line
                while (pos < load_size && shared->buffer[pos] != '\n') {
                    pos++;
                }
                
                token.length = (start_pos + pos) - token.start_pos;
            }
            // Check for block comments
            else if (ch == '/' && next_ch == '*') {
                Token& token = local_tokens[local_token_count++];
                token.type = TokenType::BlockComment;
                token.start_pos = start_pos + pos;
                token.line = 1;
                token.column = pos + 1;
                
                pos += 2; // Skip /*
                
                // Find end of block comment
                while (pos + 1 < load_size) {
                    if (shared->buffer[pos] == '*' && shared->buffer[pos + 1] == '/') {
                        pos += 2; // Skip */
                        break;
                    }
                    pos++;
                }
                
                token.length = (start_pos + pos) - token.start_pos;
            }
            // Raw string literals (r#"..."#)
            else if (ch == 'r' && next_ch == '#') {
                Token& token = local_tokens[local_token_count++];
                token.type = TokenType::StringLiteral;
                token.start_pos = start_pos + pos;
                token.line = 1;
                token.column = pos + 1;
                
                pos += 2; // Skip r#
                
                // Count number of # symbols
                uint32_t hash_count = 1;
                while (pos < load_size && shared->buffer[pos] == '#') {
                    hash_count++;
                    pos++;
                }
                
                if (pos < load_size && shared->buffer[pos] == '"') {
                    pos++; // Skip opening quote
                    
                    // Find closing quote with same number of hashes
                    bool found_end = false;
                    while (pos < load_size && !found_end) {
                        if (shared->buffer[pos] == '"') {
                            uint32_t end_hash_count = 0;
                            uint32_t check_pos = pos + 1;
                            
                            while (check_pos < load_size && 
                                   shared->buffer[check_pos] == '#' && 
                                   end_hash_count < hash_count) {
                                end_hash_count++;
                                check_pos++;
                            }
                            
                            if (end_hash_count == hash_count) {
                                pos = check_pos;
                                found_end = true;
                            } else {
                                pos++;
                            }
                        } else {
                            pos++;
                        }
                    }
                }
                
                token.length = (start_pos + pos) - token.start_pos;
            }
            // Byte string literals (b"...")
            else if (ch == 'b' && next_ch == '"') {
                Token& token = local_tokens[local_token_count++];
                token.type = TokenType::StringLiteral;
                token.start_pos = start_pos + pos;
                token.line = 1;
                token.column = pos + 1;
                
                pos += 2; // Skip b"
                
                // Process like regular string
                while (pos < load_size && shared->buffer[pos] != '"') {
                    if (shared->buffer[pos] == '\\' && pos + 1 < load_size) {
                        pos += 2; // Skip escape sequence
                    } else {
                        pos++;
                    }
                }
                
                if (pos < load_size && shared->buffer[pos] == '"') {
                    pos++; // Skip closing quote
                }
                
                token.length = (start_pos + pos) - token.start_pos;
            }
            else {
                // Not a string or comment, skip
                pos++;
            }
        }
        
        // Write tokens to global memory
        if (local_token_count > 0) {
            uint32_t write_offset = atomicAdd(token_count, local_token_count);
            
            if (write_offset + local_token_count <= max_tokens) {
                for (uint32_t i = 0; i < local_token_count; ++i) {
                    tokens[write_offset + i] = local_tokens[i];
                }
            }
        }
    }
}

// Host function to launch string/comment handler
extern "C" void launch_string_comment_handler(
    const char* source,
    size_t source_len,
    Token* tokens,
    uint32_t* token_count,
    uint32_t max_tokens
) {
    uint32_t threads_per_block = 256;
    uint32_t num_blocks = min(65535u, 
        static_cast<uint32_t>((source_len + 8192 - 1) / 8192));
    
    size_t shared_mem_size = sizeof(StringCommentSharedMem);
    
    process_string_literals_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        source, source_len, tokens, token_count, max_tokens
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in process_string_literals_kernel: %s\n", cudaGetErrorString(err));
    }
}

} // namespace rustg