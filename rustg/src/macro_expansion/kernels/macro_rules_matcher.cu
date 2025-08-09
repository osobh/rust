#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include "../../../include/gpu_types.h"

namespace cg = cooperative_groups;

namespace rustg {

// macro_rules! pattern components
struct MacroRulesPattern {
    uint32_t pattern_id;
    uint32_t num_arms;           // Number of match arms
    uint32_t arm_offsets[16];    // Offset to each arm in pattern buffer
    bool has_repetition;         // Contains $()* or $()+
    bool is_recursive;           // Can expand to itself
};

// Match arm for macro_rules!
struct MacroArm {
    uint32_t pattern_start;      // Start of pattern tokens
    uint32_t pattern_end;        // End of pattern tokens
    uint32_t template_start;     // Start of expansion template
    uint32_t template_end;       // End of expansion template
    uint32_t num_bindings;       // Number of $name bindings
    uint32_t binding_types[32];  // Fragment types for each binding
};

// Binding for captured macro arguments
struct MacroBinding {
    uint32_t name_hash;          // Hash of binding name
    uint32_t token_start;        // Start of captured tokens
    uint32_t token_end;          // End of captured tokens
    FragmentType fragment_type;  // Type of fragment (expr, ty, etc.)
    bool is_repetition;          // Part of repetition group
    uint32_t repetition_count;   // Number of repetitions
};

// Repetition group in pattern
struct RepetitionGroup {
    uint32_t start_pos;          // Start of repetition pattern
    uint32_t end_pos;            // End of repetition pattern
    uint32_t separator_token;    // Token type for separator (comma, etc.)
    bool is_plus;                // + (one or more) vs * (zero or more)
    uint32_t min_count;          // Minimum repetitions (0 or 1)
    uint32_t max_count;          // Maximum repetitions (UINT32_MAX for unlimited)
};

// Shared memory for macro_rules! matching
struct MacroRulesSharedMem {
    Token pattern_tokens[512];      // Pattern to match against
    Token input_tokens[512];        // Input tokens to match
    MacroBinding bindings[64];      // Captured bindings
    RepetitionGroup repetitions[16]; // Repetition groups
    uint32_t binding_count;
    uint32_t repetition_count;
    uint32_t match_score;            // Quality of match (for best arm selection)
    bool match_success;
};

// Fragment matcher for specific types
__device__ bool match_fragment(
    const Token* tokens,
    uint32_t& pos,
    uint32_t end,
    FragmentType fragment,
    uint32_t& consumed
) {
    if (pos >= end) return false;
    
    consumed = 0;
    uint32_t start = pos;
    
    switch (fragment) {
        case FragmentType::Expr: {
            // Match expression (simplified: until semicolon or comma)
            uint32_t depth = 0;
            while (pos < end) {
                TokenType type = tokens[pos].type;
                
                if (type == TokenType::LeftParen || 
                    type == TokenType::LeftBracket ||
                    type == TokenType::LeftBrace) {
                    depth++;
                } else if (type == TokenType::RightParen ||
                           type == TokenType::RightBracket ||
                           type == TokenType::RightBrace) {
                    if (depth == 0) break;
                    depth--;
                } else if (depth == 0 && 
                          (type == TokenType::Semicolon || 
                           type == TokenType::Comma)) {
                    break;
                }
                pos++;
            }
            consumed = pos - start;
            return consumed > 0;
        }
        
        case FragmentType::Ident: {
            // Match single identifier
            if (tokens[pos].type == TokenType::Identifier) {
                consumed = 1;
                pos++;
                return true;
            }
            return false;
        }
        
        case FragmentType::Type: {
            // Match type (simplified: identifier with optional generics)
            if (tokens[pos].type == TokenType::Identifier) {
                pos++;
                consumed = 1;
                
                // Check for generics
                if (pos < end && tokens[pos].type == TokenType::Less) {
                    uint32_t angle_depth = 1;
                    pos++;
                    
                    while (pos < end && angle_depth > 0) {
                        if (tokens[pos].type == TokenType::Less) {
                            angle_depth++;
                        } else if (tokens[pos].type == TokenType::Greater) {
                            angle_depth--;
                        }
                        pos++;
                        consumed++;
                    }
                }
                return true;
            }
            return false;
        }
        
        case FragmentType::Path: {
            // Match path (mod::Type::method)
            if (tokens[pos].type != TokenType::Identifier) return false;
            
            pos++;
            consumed = 1;
            
            while (pos + 1 < end && 
                   tokens[pos].type == TokenType::ColonColon &&
                   tokens[pos + 1].type == TokenType::Identifier) {
                pos += 2;
                consumed += 2;
            }
            return true;
        }
        
        case FragmentType::Pattern: {
            // Match pattern (simplified)
            if (tokens[pos].type == TokenType::Identifier ||
                tokens[pos].type == TokenType::Underscore ||
                tokens[pos].type == TokenType::IntegerLiteral) {
                pos++;
                consumed = 1;
                return true;
            }
            return false;
        }
        
        case FragmentType::Literal: {
            // Match literal
            TokenType type = tokens[pos].type;
            if (type == TokenType::IntegerLiteral ||
                type == TokenType::FloatLiteral ||
                type == TokenType::StringLiteral ||
                type == TokenType::CharLiteral ||
                type == TokenType::BoolLiteral) {
                pos++;
                consumed = 1;
                return true;
            }
            return false;
        }
        
        case FragmentType::TokenTree: {
            // Match any single token tree
            if (tokens[pos].type == TokenType::LeftParen ||
                tokens[pos].type == TokenType::LeftBracket ||
                tokens[pos].type == TokenType::LeftBrace) {
                
                uint32_t depth = 1;
                TokenType open = tokens[pos].type;
                TokenType close = (open == TokenType::LeftParen) ? TokenType::RightParen :
                                 (open == TokenType::LeftBracket) ? TokenType::RightBracket :
                                 TokenType::RightBrace;
                pos++;
                
                while (pos < end && depth > 0) {
                    if (tokens[pos].type == open) depth++;
                    else if (tokens[pos].type == close) depth--;
                    pos++;
                }
                
                consumed = pos - start;
                return depth == 0;
            } else {
                // Single token
                pos++;
                consumed = 1;
                return true;
            }
        }
        
        default:
            return false;
    }
}

// Match repetition pattern $(...)* or $()+
__device__ bool match_repetition(
    const Token* pattern,
    uint32_t pattern_start,
    uint32_t pattern_end,
    const Token* input,
    uint32_t& input_pos,
    uint32_t input_end,
    RepetitionGroup& group,
    MacroBinding* bindings,
    uint32_t& binding_count,
    uint32_t max_bindings
) {
    uint32_t match_count = 0;
    uint32_t start_pos = input_pos;
    
    // Try to match pattern repeatedly
    while (input_pos < input_end && match_count < group.max_count) {
        uint32_t prev_pos = input_pos;
        uint32_t pattern_pos = pattern_start;
        bool arm_matched = true;
        
        // Try to match one instance of the pattern
        while (pattern_pos < pattern_end && arm_matched) {
            // Handle $ bindings
            if (pattern[pattern_pos].type == TokenType::Dollar &&
                pattern_pos + 1 < pattern_end) {
                
                pattern_pos++; // Skip $
                
                // Get binding name and type
                if (pattern[pattern_pos].type == TokenType::Identifier &&
                    pattern_pos + 1 < pattern_end &&
                    pattern[pattern_pos + 1].type == TokenType::Colon) {
                    
                    uint32_t name_hash = 0; // Compute hash from token
                    pattern_pos += 2; // Skip name and colon
                    
                    // Get fragment type
                    FragmentType frag_type = FragmentType::Expr; // Parse from pattern
                    pattern_pos++;
                    
                    uint32_t consumed = 0;
                    if (match_fragment(input, input_pos, input_end, frag_type, consumed)) {
                        // Record binding
                        if (binding_count < max_bindings) {
                            bindings[binding_count].name_hash = name_hash;
                            bindings[binding_count].token_start = input_pos - consumed;
                            bindings[binding_count].token_end = input_pos;
                            bindings[binding_count].fragment_type = frag_type;
                            bindings[binding_count].is_repetition = true;
                            bindings[binding_count].repetition_count = match_count;
                            binding_count++;
                        }
                    } else {
                        arm_matched = false;
                    }
                }
            } else {
                // Match literal token
                if (input_pos < input_end && 
                    pattern[pattern_pos].type == input[input_pos].type) {
                    pattern_pos++;
                    input_pos++;
                } else {
                    arm_matched = false;
                }
            }
        }
        
        if (arm_matched) {
            match_count++;
            
            // Check for separator
            if (group.separator_token != 0 && input_pos < input_end) {
                if (input[input_pos].type == static_cast<TokenType>(group.separator_token)) {
                    input_pos++; // Consume separator
                } else if (match_count >= group.min_count) {
                    // No separator, but we have enough matches
                    break;
                }
            }
        } else {
            // Restore position if match failed
            input_pos = prev_pos;
            break;
        }
    }
    
    // Check if we matched minimum required
    return match_count >= group.min_count;
}

// Main macro_rules! pattern matching kernel
__global__ void match_macro_rules_kernel(
    const Token* __restrict__ pattern_tokens,
    uint32_t pattern_count,
    const Token* __restrict__ input_tokens,
    uint32_t input_count,
    MacroBinding* __restrict__ output_bindings,
    uint32_t* __restrict__ binding_count,
    uint32_t* __restrict__ match_result,
    uint32_t max_bindings
) {
    extern __shared__ char shared_mem_raw[];
    MacroRulesSharedMem* shared = 
        reinterpret_cast<MacroRulesSharedMem*>(shared_mem_raw);
    
    const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // Initialize shared memory
    if (tid == 0) {
        shared->binding_count = 0;
        shared->repetition_count = 0;
        shared->match_score = 0;
        shared->match_success = false;
    }
    __syncthreads();
    
    // Load pattern and input to shared memory
    uint32_t pattern_load = min(512u, pattern_count);
    uint32_t input_load = min(512u, input_count);
    
    for (uint32_t i = tid; i < pattern_load; i += blockDim.x) {
        shared->pattern_tokens[i] = pattern_tokens[i];
    }
    
    for (uint32_t i = tid; i < input_load; i += blockDim.x) {
        shared->input_tokens[i] = input_tokens[i];
    }
    __syncthreads();
    
    // Pattern matching with warp cooperation
    if (warp_id == 0) {
        // Warp 0 handles main pattern matching
        uint32_t pattern_pos = lane_id;
        uint32_t input_pos = lane_id;
        
        // Simplified: try to match pattern against input
        bool local_match = false;
        
        if (pattern_pos < pattern_load && input_pos < input_load) {
            // Check for $ binding
            if (shared->pattern_tokens[pattern_pos].type == TokenType::Dollar) {
                // This is a binding position
                local_match = true;
            } else if (shared->pattern_tokens[pattern_pos].type == 
                      shared->input_tokens[input_pos].type) {
                // Literal match
                local_match = true;
            }
        }
        
        // Vote on matches
        uint32_t match_mask = warp.ballot(local_match);
        
        if (lane_id == 0) {
            // Count matches
            uint32_t match_count = __popc(match_mask);
            shared->match_score = match_count;
            shared->match_success = (match_count > pattern_load / 2);
        }
    }
    
    __syncthreads();
    
    // Process bindings (single thread for simplicity)
    if (tid == 0 && shared->match_success) {
        uint32_t pattern_idx = 0;
        uint32_t input_idx = 0;
        
        while (pattern_idx < pattern_load && input_idx < input_load) {
            Token& p_token = shared->pattern_tokens[pattern_idx];
            
            // Check for binding
            if (p_token.type == TokenType::Dollar && pattern_idx + 3 < pattern_load) {
                // Parse binding: $name:type
                pattern_idx++; // Skip $
                
                Token& name_token = shared->pattern_tokens[pattern_idx];
                if (name_token.type == TokenType::Identifier) {
                    pattern_idx++; // Skip name
                    
                    if (shared->pattern_tokens[pattern_idx].type == TokenType::Colon) {
                        pattern_idx++; // Skip :
                        
                        // Get fragment type (simplified)
                        FragmentType frag = FragmentType::Expr;
                        pattern_idx++;
                        
                        // Capture binding
                        uint32_t consumed = 0;
                        if (match_fragment(shared->input_tokens, input_idx, 
                                         input_load, frag, consumed)) {
                            
                            if (shared->binding_count < 64) {
                                MacroBinding& binding = shared->bindings[shared->binding_count];
                                binding.name_hash = 0; // Compute from name_token
                                binding.token_start = input_idx - consumed;
                                binding.token_end = input_idx;
                                binding.fragment_type = frag;
                                binding.is_repetition = false;
                                shared->binding_count++;
                            }
                        }
                    }
                }
            } else {
                // Match literal
                if (p_token.type == shared->input_tokens[input_idx].type) {
                    pattern_idx++;
                    input_idx++;
                } else {
                    // Mismatch
                    shared->match_success = false;
                    break;
                }
            }
        }
    }
    
    __syncthreads();
    
    // Write results to global memory
    if (tid == 0) {
        *match_result = shared->match_success ? 1 : 0;
        *binding_count = shared->binding_count;
        
        // Copy bindings
        for (uint32_t i = 0; i < shared->binding_count && i < max_bindings; ++i) {
            output_bindings[i] = shared->bindings[i];
        }
    }
}

// Host function to launch macro_rules! matcher
extern "C" void launch_macro_rules_matcher(
    const Token* pattern_tokens,
    uint32_t pattern_count,
    const Token* input_tokens,
    uint32_t input_count,
    MacroBinding* output_bindings,
    uint32_t* binding_count,
    uint32_t* match_result,
    uint32_t max_bindings
) {
    uint32_t threads_per_block = 256;
    uint32_t num_blocks = 1; // Single block for now
    
    size_t shared_mem_size = sizeof(MacroRulesSharedMem);
    
    match_macro_rules_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        pattern_tokens, pattern_count,
        input_tokens, input_count,
        output_bindings, binding_count,
        match_result, max_bindings
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in match_macro_rules_kernel: %s\n",
               cudaGetErrorString(err));
    }
}

} // namespace rustg