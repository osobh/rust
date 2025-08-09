#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include "../../../include/gpu_types.h"

namespace cg = cooperative_groups;

namespace rustg {

// Error types for macro expansion
enum class MacroErrorType : uint8_t {
    None = 0,
    InvalidPattern,
    UnmatchedDelimiter,
    UnknownFragment,
    RecursionLimit,
    ExpansionOverflow,
    InvalidBinding,
    MissingArgument,
    ExtraArgument,
    InvalidRepetition,
    HygieneViolation
};

// Error information structure
struct MacroError {
    MacroErrorType type;
    uint32_t token_pos;      // Position where error occurred
    uint32_t line;
    uint32_t column;
    uint32_t macro_id;       // Which macro caused error
    uint32_t context_hash;   // Additional context
    char message[128];       // Error message
};

// Shared memory for error handling
struct ErrorHandlerSharedMem {
    MacroError errors[256];       // Error buffer
    uint32_t error_count;
    uint32_t max_errors;
    Token problem_tokens[128];    // Tokens causing errors
    uint32_t recursion_depth;
    uint32_t expansion_size;
    bool overflow_detected;
};

// Generate error message
__device__ void generate_error_message(
    MacroErrorType type,
    char* message,
    uint32_t max_len
) {
    const char* msg = nullptr;
    
    switch (type) {
        case MacroErrorType::InvalidPattern:
            msg = "Invalid macro pattern";
            break;
        case MacroErrorType::UnmatchedDelimiter:
            msg = "Unmatched delimiter in macro";
            break;
        case MacroErrorType::UnknownFragment:
            msg = "Unknown fragment specifier";
            break;
        case MacroErrorType::RecursionLimit:
            msg = "Macro recursion limit exceeded";
            break;
        case MacroErrorType::ExpansionOverflow:
            msg = "Macro expansion too large";
            break;
        case MacroErrorType::InvalidBinding:
            msg = "Invalid binding in macro pattern";
            break;
        case MacroErrorType::MissingArgument:
            msg = "Missing required macro argument";
            break;
        case MacroErrorType::ExtraArgument:
            msg = "Extra argument to macro";
            break;
        case MacroErrorType::InvalidRepetition:
            msg = "Invalid repetition pattern";
            break;
        case MacroErrorType::HygieneViolation:
            msg = "Hygiene violation detected";
            break;
        default:
            msg = "Unknown macro error";
            break;
    }
    
    // Copy message
    uint32_t i = 0;
    while (msg[i] != '\0' && i < max_len - 1) {
        message[i] = msg[i];
        i++;
    }
    message[i] = '\0';
}

// Validate macro pattern
__device__ bool validate_macro_pattern(
    const Token* pattern,
    uint32_t pattern_size,
    MacroError& error
) {
    uint32_t paren_depth = 0;
    uint32_t bracket_depth = 0;
    uint32_t brace_depth = 0;
    
    for (uint32_t i = 0; i < pattern_size; ++i) {
        const Token& token = pattern[i];
        
        // Check delimiter matching
        switch (token.type) {
            case TokenType::LeftParen:
                paren_depth++;
                break;
            case TokenType::RightParen:
                if (paren_depth == 0) {
                    error.type = MacroErrorType::UnmatchedDelimiter;
                    error.token_pos = i;
                    return false;
                }
                paren_depth--;
                break;
            case TokenType::LeftBracket:
                bracket_depth++;
                break;
            case TokenType::RightBracket:
                if (bracket_depth == 0) {
                    error.type = MacroErrorType::UnmatchedDelimiter;
                    error.token_pos = i;
                    return false;
                }
                bracket_depth--;
                break;
            case TokenType::LeftBrace:
                brace_depth++;
                break;
            case TokenType::RightBrace:
                if (brace_depth == 0) {
                    error.type = MacroErrorType::UnmatchedDelimiter;
                    error.token_pos = i;
                    return false;
                }
                brace_depth--;
                break;
        }
        
        // Check for invalid patterns
        if (token.type == TokenType::Dollar && i + 1 < pattern_size) {
            const Token& next = pattern[i + 1];
            if (next.type != TokenType::Identifier &&
                next.type != TokenType::LeftParen) {
                error.type = MacroErrorType::InvalidPattern;
                error.token_pos = i;
                return false;
            }
        }
    }
    
    // Check all delimiters are closed
    if (paren_depth != 0 || bracket_depth != 0 || brace_depth != 0) {
        error.type = MacroErrorType::UnmatchedDelimiter;
        error.token_pos = pattern_size - 1;
        return false;
    }
    
    return true;
}

// Check recursion depth
__device__ bool check_recursion_depth(
    uint32_t current_depth,
    uint32_t max_depth,
    MacroError& error
) {
    if (current_depth >= max_depth) {
        error.type = MacroErrorType::RecursionLimit;
        generate_error_message(error.type, error.message, 128);
        return false;
    }
    return true;
}

// Check expansion size
__device__ bool check_expansion_size(
    uint32_t current_size,
    uint32_t max_size,
    MacroError& error
) {
    if (current_size >= max_size) {
        error.type = MacroErrorType::ExpansionOverflow;
        generate_error_message(error.type, error.message, 128);
        return false;
    }
    return true;
}

// Main error detection kernel
__global__ void detect_macro_errors_kernel(
    const Token* __restrict__ tokens,
    uint32_t token_count,
    const uint8_t* __restrict__ pattern_matches,
    uint32_t match_count,
    MacroError* __restrict__ errors,
    uint32_t* __restrict__ error_count,
    uint32_t max_errors
) {
    extern __shared__ char shared_mem_raw[];
    ErrorHandlerSharedMem* shared = 
        reinterpret_cast<ErrorHandlerSharedMem*>(shared_mem_raw);
    
    const uint32_t tid = threadIdx.x;
    const uint32_t block_id = blockIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // Initialize shared memory
    if (tid == 0) {
        shared->error_count = 0;
        shared->max_errors = min(256u, max_errors);
        shared->recursion_depth = 0;
        shared->expansion_size = 0;
        shared->overflow_detected = false;
    }
    __syncthreads();
    
    // Each warp checks different aspects
    if (warp_id == 0) {
        // Warp 0: Check pattern validity
        for (uint32_t i = lane_id; i < token_count; i += 32) {
            if (pattern_matches[i] > 0) {
                // Found a macro, validate its pattern
                MacroError local_error = {};
                
                // Find macro extent
                uint32_t end_pos = i + 2; // At least name + !
                uint32_t depth = 0;
                
                if (end_pos < token_count) {
                    TokenType delim = tokens[end_pos].type;
                    if (delim == TokenType::LeftParen ||
                        delim == TokenType::LeftBracket ||
                        delim == TokenType::LeftBrace) {
                        depth = 1;
                        end_pos++;
                        
                        while (end_pos < token_count && depth > 0) {
                            TokenType t = tokens[end_pos].type;
                            if (t == delim) depth++;
                            else if ((delim == TokenType::LeftParen && t == TokenType::RightParen) ||
                                    (delim == TokenType::LeftBracket && t == TokenType::RightBracket) ||
                                    (delim == TokenType::LeftBrace && t == TokenType::RightBrace)) {
                                depth--;
                            }
                            end_pos++;
                        }
                        
                        if (depth != 0) {
                            // Unmatched delimiter
                            local_error.type = MacroErrorType::UnmatchedDelimiter;
                            local_error.token_pos = i;
                            local_error.line = tokens[i].line;
                            local_error.column = tokens[i].column;
                            generate_error_message(local_error.type, local_error.message, 128);
                            
                            // Add to shared errors
                            uint32_t err_idx = atomicAdd(&shared->error_count, 1);
                            if (err_idx < shared->max_errors) {
                                shared->errors[err_idx] = local_error;
                            }
                        }
                    }
                }
            }
        }
    } else if (warp_id == 1) {
        // Warp 1: Check for recursion depth
        uint32_t depth_counter = 0;
        for (uint32_t i = lane_id; i < match_count; i += 32) {
            depth_counter++;
            if (depth_counter > 100) { // Max recursion depth
                if (lane_id == 0) {
                    MacroError rec_error = {};
                    rec_error.type = MacroErrorType::RecursionLimit;
                    rec_error.token_pos = i;
                    generate_error_message(rec_error.type, rec_error.message, 128);
                    
                    uint32_t err_idx = atomicAdd(&shared->error_count, 1);
                    if (err_idx < shared->max_errors) {
                        shared->errors[err_idx] = rec_error;
                    }
                }
                break;
            }
        }
    } else if (warp_id == 2) {
        // Warp 2: Check expansion size
        uint32_t total_size = 0;
        for (uint32_t i = lane_id; i < token_count; i += 32) {
            if (pattern_matches[i] > 0) {
                total_size += 100; // Estimate expansion size
            }
        }
        
        // Reduce within warp
        for (int offset = 16; offset > 0; offset /= 2) {
            total_size += warp.shfl_down(total_size, offset);
        }
        
        if (lane_id == 0 && total_size > 100000) { // Max expansion size
            MacroError size_error = {};
            size_error.type = MacroErrorType::ExpansionOverflow;
            generate_error_message(size_error.type, size_error.message, 128);
            
            uint32_t err_idx = atomicAdd(&shared->error_count, 1);
            if (err_idx < shared->max_errors) {
                shared->errors[err_idx] = size_error;
            }
        }
    }
    
    __syncthreads();
    
    // Write errors to global memory
    if (tid == 0) {
        *error_count = min(shared->error_count, max_errors);
    }
    
    // Copy errors
    for (uint32_t i = tid; i < shared->error_count && i < max_errors; i += blockDim.x) {
        errors[i] = shared->errors[i];
    }
}

// Error recovery kernel
__global__ void recover_from_errors_kernel(
    Token* __restrict__ tokens,
    uint32_t token_count,
    const MacroError* __restrict__ errors,
    uint32_t error_count,
    uint8_t* __restrict__ skip_mask
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= token_count) return;
    
    // Check if this token is part of an error
    bool should_skip = false;
    
    for (uint32_t i = 0; i < error_count; ++i) {
        const MacroError& error = errors[i];
        
        // Skip tokens around error position
        if (tid >= error.token_pos && tid < error.token_pos + 10) {
            should_skip = true;
            break;
        }
    }
    
    skip_mask[tid] = should_skip ? 1 : 0;
}

// Host function to launch error detection
extern "C" void launch_macro_error_detection(
    const Token* tokens,
    uint32_t token_count,
    const uint8_t* pattern_matches,
    uint32_t match_count,
    MacroError* errors,
    uint32_t* error_count,
    uint32_t max_errors
) {
    uint32_t threads_per_block = 96; // 3 warps
    uint32_t num_blocks = 1;
    
    size_t shared_mem_size = sizeof(ErrorHandlerSharedMem);
    
    detect_macro_errors_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        tokens, token_count, pattern_matches, match_count,
        errors, error_count, max_errors
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in detect_macro_errors_kernel: %s\n",
               cudaGetErrorString(err));
    }
}

// Host function for error recovery
extern "C" void launch_error_recovery(
    Token* tokens,
    uint32_t token_count,
    const MacroError* errors,
    uint32_t error_count,
    uint8_t* skip_mask
) {
    uint32_t threads_per_block = 256;
    uint32_t num_blocks = (token_count + threads_per_block - 1) / threads_per_block;
    
    recover_from_errors_kernel<<<num_blocks, threads_per_block>>>(
        tokens, token_count, errors, error_count, skip_mask
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in recover_from_errors_kernel: %s\n",
               cudaGetErrorString(err));
    }
}

} // namespace rustg