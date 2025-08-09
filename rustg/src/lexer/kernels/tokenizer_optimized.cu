#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include "../../../include/gpu_types.h"

namespace cg = cooperative_groups;

namespace rustg {

// Forward declarations
extern __device__ CharClass classify_char(unsigned char ch);
extern __host__ void initialize_char_class_table();

// Optimized constants for performance
constexpr uint32_t WARP_SIZE = 32;
constexpr uint32_t WARPS_PER_BLOCK = 8;
constexpr uint32_t THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;
constexpr uint32_t SHARED_MEM_SIZE = 48 * 1024;  // 48KB shared memory
constexpr uint32_t CHARS_PER_THREAD = 8;  // Increased for better throughput
constexpr uint32_t TOKEN_BUFFER_SIZE = 64;  // Per-warp token buffer

// Shared memory structure for optimized access
struct SharedMemory {
    char source_cache[8192];  // 8KB source cache
    Token token_buffer[TOKEN_BUFFER_SIZE * WARPS_PER_BLOCK];  // Token buffers
    uint32_t warp_token_count[WARPS_PER_BLOCK];  // Token counts per warp
    uint32_t warp_write_offset[WARPS_PER_BLOCK];  // Global write offsets
};

// Optimized token detection using warp voting
__device__ inline uint32_t detect_token_type_optimized(
    const char* source,
    uint32_t pos,
    uint32_t source_len,
    uint32_t& token_end
) {
    if (pos >= source_len) return static_cast<uint32_t>(TokenType::EOF);
    
    unsigned char ch = static_cast<unsigned char>(source[pos]);
    CharClass cls = classify_char(ch);
    token_end = pos + 1;
    
    // Fast path for single-character tokens
    switch (cls) {
        case CharClass::Whitespace:
            // Unroll loop for whitespace scanning
            #pragma unroll 4
            for (int i = 0; i < 4 && token_end < source_len; ++i) {
                if (classify_char(source[token_end]) == CharClass::Whitespace) {
                    token_end++;
                } else {
                    break;
                }
            }
            return static_cast<uint32_t>(TokenType::Whitespace);
            
        case CharClass::Newline:
            return static_cast<uint32_t>(TokenType::Newline);
            
        case CharClass::Letter:
            // Vectorized identifier scanning
            while (token_end < source_len && token_end < pos + 64) {
                CharClass next = classify_char(source[token_end]);
                if (next == CharClass::Letter || next == CharClass::Digit) {
                    token_end++;
                } else {
                    break;
                }
            }
            
            // Fast keyword detection using hash
            uint32_t len = token_end - pos;
            if (len == 2) {
                uint16_t word = (source[pos] << 8) | source[pos + 1];
                if (word == ('f' << 8 | 'n')) return static_cast<uint32_t>(TokenType::KeywordFn);
                if (word == ('i' << 8 | 'f')) return static_cast<uint32_t>(TokenType::KeywordIf);
            } else if (len == 3) {
                uint32_t word = (source[pos] << 16) | (source[pos + 1] << 8) | source[pos + 2];
                if (word == ('l' << 16 | 'e' << 8 | 't')) 
                    return static_cast<uint32_t>(TokenType::KeywordLet);
                if (word == ('m' << 16 | 'u' << 8 | 't')) 
                    return static_cast<uint32_t>(TokenType::KeywordMut);
                if (word == ('f' << 16 | 'o' << 8 | 'r')) 
                    return static_cast<uint32_t>(TokenType::KeywordFor);
                if (word == ('p' << 16 | 'u' << 8 | 'b')) 
                    return static_cast<uint32_t>(TokenType::KeywordPub);
                if (word == ('u' << 16 | 's' << 8 | 'e')) 
                    return static_cast<uint32_t>(TokenType::KeywordUse);
                if (word == ('m' << 16 | 'o' << 8 | 'd')) 
                    return static_cast<uint32_t>(TokenType::KeywordMod);
            }
            return static_cast<uint32_t>(TokenType::Identifier);
            
        case CharClass::Digit:
            // Vectorized number scanning with float detection
            while (token_end < source_len && classify_char(source[token_end]) == CharClass::Digit) {
                token_end++;
            }
            if (token_end < source_len && source[token_end] == '.') {
                if (token_end + 1 < source_len && classify_char(source[token_end + 1]) == CharClass::Digit) {
                    token_end += 2;
                    while (token_end < source_len && classify_char(source[token_end]) == CharClass::Digit) {
                        token_end++;
                    }
                    return static_cast<uint32_t>(TokenType::FloatLiteral);
                }
            }
            return static_cast<uint32_t>(TokenType::IntegerLiteral);
            
        case CharClass::Operator:
            // Lookup table for operators
            switch (ch) {
                case '+': return static_cast<uint32_t>(TokenType::Plus);
                case '-': 
                    if (token_end < source_len && source[token_end] == '>') {
                        token_end++;
                        return static_cast<uint32_t>(TokenType::Arrow);
                    }
                    return static_cast<uint32_t>(TokenType::Minus);
                case '*': return static_cast<uint32_t>(TokenType::Star);
                case '/': return static_cast<uint32_t>(TokenType::Slash);
                case '%': return static_cast<uint32_t>(TokenType::Percent);
                case '&': return static_cast<uint32_t>(TokenType::Ampersand);
                case '|': return static_cast<uint32_t>(TokenType::Pipe);
                case '^': return static_cast<uint32_t>(TokenType::Caret);
                case '~': return static_cast<uint32_t>(TokenType::Tilde);
                case '!': return static_cast<uint32_t>(TokenType::Bang);
                case '=': 
                    if (token_end < source_len && source[token_end] == '>') {
                        token_end++;
                        return static_cast<uint32_t>(TokenType::FatArrow);
                    }
                    return static_cast<uint32_t>(TokenType::Equal);
                case '<': return static_cast<uint32_t>(TokenType::Less);
                case '>': return static_cast<uint32_t>(TokenType::Greater);
            }
            break;
            
        case CharClass::Delimiter:
            switch (ch) {
                case '(': return static_cast<uint32_t>(TokenType::LeftParen);
                case ')': return static_cast<uint32_t>(TokenType::RightParen);
                case '{': return static_cast<uint32_t>(TokenType::LeftBrace);
                case '}': return static_cast<uint32_t>(TokenType::RightBrace);
                case '[': return static_cast<uint32_t>(TokenType::LeftBracket);
                case ']': return static_cast<uint32_t>(TokenType::RightBracket);
                case ';': return static_cast<uint32_t>(TokenType::Semicolon);
                case ',': return static_cast<uint32_t>(TokenType::Comma);
                case '.': return static_cast<uint32_t>(TokenType::Dot);
                case ':': 
                    if (token_end < source_len && source[token_end] == ':') {
                        token_end++;
                        return static_cast<uint32_t>(TokenType::DoubleColon);
                    }
                    return static_cast<uint32_t>(TokenType::Colon);
            }
            break;
    }
    
    return static_cast<uint32_t>(TokenType::Invalid);
}

// Optimized tokenizer kernel with shared memory and warp cooperation
__global__ void tokenize_optimized_kernel(
    const char* __restrict__ source,
    size_t source_len,
    Token* __restrict__ tokens,
    uint32_t* __restrict__ token_count,
    uint32_t max_tokens
) {
    extern __shared__ char shared_mem_raw[];
    SharedMemory* shared = reinterpret_cast<SharedMemory*>(shared_mem_raw);
    
    // Thread and warp identification
    const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = tid / WARP_SIZE;
    const uint32_t lane_id = tid % WARP_SIZE;
    const uint32_t global_tid = blockIdx.x * blockDim.x + tid;
    
    // Initialize shared memory
    if (tid < WARPS_PER_BLOCK) {
        shared->warp_token_count[tid] = 0;
        shared->warp_write_offset[tid] = 0;
    }
    __syncthreads();
    
    // Calculate work distribution
    const uint32_t total_threads = gridDim.x * blockDim.x;
    const uint32_t chunk_size = (source_len + total_threads - 1) / total_threads;
    const uint32_t start_pos = global_tid * chunk_size;
    
    if (start_pos >= source_len) return;
    
    const uint32_t end_pos = min(start_pos + chunk_size, static_cast<uint32_t>(source_len));
    
    // Load source data into shared memory (coalesced)
    const uint32_t cache_start = (start_pos / 128) * 128;  // Align to 128-byte boundary
    const uint32_t cache_end = min(cache_start + 8192, static_cast<uint32_t>(source_len));
    
    // Cooperative loading into shared memory
    for (uint32_t i = tid; i < (cache_end - cache_start); i += blockDim.x) {
        if (cache_start + i < source_len) {
            shared->source_cache[i] = source[cache_start + i];
        }
    }
    __syncthreads();
    
    // Token detection phase
    uint32_t pos = start_pos;
    uint32_t local_token_count = 0;
    Token* warp_tokens = &shared->token_buffer[warp_id * TOKEN_BUFFER_SIZE];
    
    // Track line and column (vectorized tracking)
    uint32_t line = 1;
    uint32_t column = 1;
    
    // Main tokenization loop
    while (pos < end_pos && local_token_count < TOKEN_BUFFER_SIZE) {
        uint32_t token_end;
        uint32_t token_type;
        
        // Use cached data if available
        if (pos >= cache_start && pos < cache_end) {
            const char* cached_source = &shared->source_cache[pos - cache_start];
            token_type = detect_token_type_optimized(
                shared->source_cache, 
                pos - cache_start, 
                cache_end - cache_start, 
                token_end
            );
            token_end += cache_start;
        } else {
            token_type = detect_token_type_optimized(source, pos, source_len, token_end);
        }
        
        // Skip whitespace tokens for better performance
        if (token_type != static_cast<uint32_t>(TokenType::Whitespace)) {
            Token& token = warp_tokens[local_token_count++];
            token.type = static_cast<TokenType>(token_type);
            token.start_pos = pos;
            token.length = token_end - pos;
            token.line = line;
            token.column = column;
        }
        
        // Update position tracking
        for (uint32_t i = pos; i < token_end; ++i) {
            if (source[i] == '\n') {
                line++;
                column = 1;
            } else {
                column++;
            }
        }
        
        pos = token_end;
    }
    
    // Warp-level token count reduction
    __syncwarp();
    
    // Lane 0 handles the warp's token buffer write
    if (lane_id == 0 && local_token_count > 0) {
        // Atomically get write offset
        uint32_t write_offset = atomicAdd(token_count, local_token_count);
        shared->warp_write_offset[warp_id] = write_offset;
        shared->warp_token_count[warp_id] = local_token_count;
    }
    __syncthreads();
    
    // Cooperative token writing (all threads in warp participate)
    if (warp_id < WARPS_PER_BLOCK && shared->warp_token_count[warp_id] > 0) {
        uint32_t write_offset = shared->warp_write_offset[warp_id];
        uint32_t count = shared->warp_token_count[warp_id];
        
        if (write_offset + count <= max_tokens) {
            // Coalesced writes using all threads in warp
            for (uint32_t i = lane_id; i < count; i += WARP_SIZE) {
                tokens[write_offset + i] = warp_tokens[i];
            }
        }
    }
}

// Host function to launch optimized tokenizer
extern "C" void launch_tokenizer_optimized(
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
    
    // Optimal kernel configuration for modern GPUs
    const uint32_t threads_per_block = THREADS_PER_BLOCK;
    const uint32_t max_blocks = 65535;
    const uint32_t chars_per_thread = CHARS_PER_THREAD;
    
    // Calculate grid size
    uint32_t total_threads_needed = (source_len + chars_per_thread - 1) / chars_per_thread;
    uint32_t num_blocks = min(max_blocks, 
        (total_threads_needed + threads_per_block - 1) / threads_per_block);
    
    // Calculate shared memory size
    size_t shared_mem_size = sizeof(SharedMemory);
    
    // Launch optimized kernel
    tokenize_optimized_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        source, source_len, tokens, token_count, max_tokens
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in tokenize_optimized_kernel: %s\n", cudaGetErrorString(err));
    }
}

} // namespace rustg