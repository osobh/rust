#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include "../../../include/gpu_types.h"

namespace cg = cooperative_groups;

namespace rustg {

// ASTNode structure is now defined in gpu_types.h
// Using ASTNode instead of ASTNode for consistency

// Operator precedence levels (higher = tighter binding)
__device__ inline uint32_t get_operator_precedence(TokenType op) {
    switch (op) {
        case TokenType::Star:
        case TokenType::Slash:
        case TokenType::Percent:
            return 150;
            
        case TokenType::Plus:
        case TokenType::Minus:
            return 140;
            
        case TokenType::Less:
        case TokenType::Greater:
            return 130;
            
        case TokenType::Equal:
            return 120;
            
        case TokenType::Ampersand:
            return 110;
            
        case TokenType::Pipe:
            return 100;
            
        default:
            return 0;
    }
}

// Check if token is a binary operator
__device__ inline bool is_binary_operator(TokenType type) {
    return type == TokenType::Plus || type == TokenType::Minus ||
           type == TokenType::Star || type == TokenType::Slash ||
           type == TokenType::Percent || type == TokenType::Equal ||
           type == TokenType::Less || type == TokenType::Greater ||
           type == TokenType::Ampersand || type == TokenType::Pipe;
}

// Check if token starts an expression
__device__ inline bool is_expression_start(TokenType type) {
    return type == TokenType::Identifier || 
           type == TokenType::IntegerLiteral ||
           type == TokenType::FloatLiteral ||
           type == TokenType::StringLiteral ||
           type == TokenType::CharLiteral ||
           type == TokenType::BoolLiteral ||
           type == TokenType::LeftParen;
}

// Shared memory for AST construction
struct SharedASTMemory {
    Token token_cache[256];          // Token cache
    ASTNode node_buffer[128];     // Local AST node buffer
    uint32_t stack[64];               // Parser stack per warp
    uint32_t warp_node_count[8];     // Node count per warp
    uint32_t warp_write_offset[8];   // Global write offset per warp
};

// Simplified Pratt parser for expressions (GPU-adapted)
__device__ uint32_t parse_expression_gpu(
    const Token* tokens,
    uint32_t& pos,
    uint32_t token_count,
    ASTNode* nodes,
    uint32_t& node_index,
    uint32_t max_nodes,
    uint32_t min_precedence = 0
) {
    if (pos >= token_count || node_index >= max_nodes) {
        return 0xFFFFFFFF;
    }
    
    // Parse primary expression
    uint32_t left_node = 0xFFFFFFFF;
    Token current = tokens[pos];
    
    if (current.type == TokenType::IntegerLiteral ||
        current.type == TokenType::FloatLiteral ||
        current.type == TokenType::Identifier) {
        // Create literal/identifier node
        left_node = node_index++;
        nodes[left_node].type = ASTNodeType::Literal;
        nodes[left_node].token_index = pos;
        nodes[left_node].first_child = 0xFFFFFFFF;
        nodes[left_node].next_sibling = 0xFFFFFFFF;
        pos++;
    } else if (current.type == TokenType::LeftParen) {
        // Parse parenthesized expression
        pos++; // Skip '('
        left_node = parse_expression_gpu(tokens, pos, token_count, nodes, node_index, max_nodes, 0);
        if (pos < token_count && tokens[pos].type == TokenType::RightParen) {
            pos++; // Skip ')'
        }
    }
    
    // Parse binary operations with precedence
    while (pos < token_count && node_index < max_nodes) {
        Token op_token = tokens[pos];
        
        if (!is_binary_operator(op_token.type)) {
            break;
        }
        
        uint32_t precedence = get_operator_precedence(op_token.type);
        if (precedence < min_precedence) {
            break;
        }
        
        pos++; // Consume operator
        
        // Parse right side with higher precedence
        uint32_t right_node = parse_expression_gpu(
            tokens, pos, token_count, nodes, node_index, max_nodes,
            precedence + 1
        );
        
        if (right_node == 0xFFFFFFFF) {
            break;
        }
        
        // Create binary operation node
        uint32_t op_node = node_index++;
        nodes[op_node].type = ASTNodeType::BinaryOp;
        nodes[op_node].token_index = pos - 1; // Operator token
        nodes[op_node].first_child = left_node;
        nodes[op_node].next_sibling = 0xFFFFFFFF;
        nodes[op_node].data = precedence;
        
        // Link children
        if (left_node != 0xFFFFFFFF) {
            nodes[left_node].parent_index = op_node;
            nodes[left_node].next_sibling = right_node;
        }
        if (right_node != 0xFFFFFFFF) {
            nodes[right_node].parent_index = op_node;
            nodes[right_node].next_sibling = 0xFFFFFFFF;
        }
        
        left_node = op_node;
    }
    
    return left_node;
}

// Main AST construction kernel
__global__ void ast_construction_kernel(
    const Token* __restrict__ tokens,
    uint32_t token_count,
    ASTNode* __restrict__ ast_nodes,
    uint32_t* __restrict__ ast_node_count,
    uint32_t* __restrict__ parent_indices,
    uint32_t max_nodes
) {
    extern __shared__ char shared_mem_raw[];
    SharedASTMemory* shared = reinterpret_cast<SharedASTMemory*>(shared_mem_raw);
    
    const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    
    // Initialize shared memory
    if (tid < 8) {
        shared->warp_node_count[tid] = 0;
        shared->warp_write_offset[tid] = 0;
    }
    __syncthreads();
    
    // Load tokens into shared memory (coalesced)
    uint32_t tokens_to_load = min(256u, token_count);
    for (uint32_t i = tid; i < tokens_to_load; i += blockDim.x) {
        shared->token_cache[i] = tokens[i];
    }
    __syncthreads();
    
    // Simple AST construction (one thread for now, can be parallelized)
    if (tid == 0) {
        ASTNode local_nodes[512];  // Local node buffer
        uint32_t node_count = 0;
        uint32_t pos = 0;
        
        // Create root Program node
        local_nodes[node_count].type = ASTNodeType::Program;
        local_nodes[node_count].token_index = 0;
        local_nodes[node_count].parent_index = 0xFFFFFFFF;
        local_nodes[node_count].first_child = 0xFFFFFFFF;
        local_nodes[node_count].next_sibling = 0xFFFFFFFF;
        uint32_t program_node = node_count++;
        
        uint32_t last_child = 0xFFFFFFFF;
        
        // Parse top-level items
        while (pos < token_count && node_count < 500) {
            Token current = (pos < tokens_to_load) ? 
                          shared->token_cache[pos] : tokens[pos];
            
            if (current.type == TokenType::EndOfFile) {
                break;
            }
            
            uint32_t item_node = 0xFFFFFFFF;
            
            // Parse function
            if (current.type == TokenType::KeywordFn) {
                item_node = node_count++;
                local_nodes[item_node].type = ASTNodeType::Function;
                local_nodes[item_node].token_index = pos;
                local_nodes[item_node].parent_index = program_node;
                local_nodes[item_node].first_child = 0xFFFFFFFF;
                local_nodes[item_node].next_sibling = 0xFFFFFFFF;
                
                pos++; // Skip 'fn'
                
                // Parse function name
                if (pos < token_count && tokens[pos].type == TokenType::Identifier) {
                    uint32_t name_node = node_count++;
                    local_nodes[name_node].type = ASTNodeType::Identifier;
                    local_nodes[name_node].token_index = pos;
                    local_nodes[name_node].parent_index = item_node;
                    local_nodes[name_node].first_child = 0xFFFFFFFF;
                    local_nodes[name_node].next_sibling = 0xFFFFFFFF;
                    
                    local_nodes[item_node].first_child = name_node;
                    pos++;
                }
                
                // Skip to function body
                while (pos < token_count && tokens[pos].type != TokenType::LeftBrace) {
                    pos++;
                }
                
                if (pos < token_count && tokens[pos].type == TokenType::LeftBrace) {
                    pos++; // Skip '{'
                    
                    // Parse function body
                    uint32_t block_node = node_count++;
                    local_nodes[block_node].type = ASTNodeType::Block;
                    local_nodes[block_node].token_index = pos - 1;
                    local_nodes[block_node].parent_index = item_node;
                    local_nodes[block_node].first_child = 0xFFFFFFFF;
                    local_nodes[block_node].next_sibling = 0xFFFFFFFF;
                    
                    // Link block to function
                    if (local_nodes[item_node].first_child != 0xFFFFFFFF) {
                        local_nodes[local_nodes[item_node].first_child].next_sibling = block_node;
                    } else {
                        local_nodes[item_node].first_child = block_node;
                    }
                    
                    // Skip to closing brace
                    uint32_t brace_count = 1;
                    while (pos < token_count && brace_count > 0) {
                        if (tokens[pos].type == TokenType::LeftBrace) brace_count++;
                        if (tokens[pos].type == TokenType::RightBrace) brace_count--;
                        pos++;
                    }
                }
            }
            // Parse expression
            else if (is_expression_start(current.type)) {
                item_node = parse_expression_gpu(
                    tokens, pos, token_count,
                    local_nodes, node_count, 500
                );
                
                if (item_node != 0xFFFFFFFF) {
                    local_nodes[item_node].parent_index = program_node;
                }
                
                // Skip to next statement
                while (pos < token_count && 
                       tokens[pos].type != TokenType::Semicolon &&
                       tokens[pos].type != TokenType::EndOfFile) {
                    pos++;
                }
                if (pos < token_count && tokens[pos].type == TokenType::Semicolon) {
                    pos++;
                }
            }
            else {
                // Skip unknown token
                pos++;
                continue;
            }
            
            // Link item to program
            if (item_node != 0xFFFFFFFF) {
                if (local_nodes[program_node].first_child == 0xFFFFFFFF) {
                    local_nodes[program_node].first_child = item_node;
                } else if (last_child != 0xFFFFFFFF) {
                    local_nodes[last_child].next_sibling = item_node;
                }
                last_child = item_node;
            }
        }
        
        // Write nodes to global memory
        uint32_t write_offset = atomicAdd(ast_node_count, node_count);
        
        if (write_offset + node_count <= max_nodes) {
            for (uint32_t i = 0; i < node_count; ++i) {
                ast_nodes[write_offset + i].type = local_nodes[i].type;
                ast_nodes[write_offset + i].token_index = local_nodes[i].token_index;
                // Initialize child links
                ast_nodes[write_offset + i].first_child = 0xFFFFFFFF;
                ast_nodes[write_offset + i].next_sibling = 0xFFFFFFFF;
                
                parent_indices[write_offset + i] = 
                    (local_nodes[i].parent_index == 0xFFFFFFFF) ? 
                    0xFFFFFFFF : write_offset + local_nodes[i].parent_index;
            }
        }
    }
}

// AST validation kernel
__global__ void ast_validator_kernel(
    const ASTNode* __restrict__ ast_nodes,
    uint32_t node_count,
    const uint32_t* __restrict__ parent_indices,
    bool* __restrict__ is_valid
) {
    __shared__ bool shared_valid;
    
    if (threadIdx.x == 0) {
        shared_valid = true;
    }
    __syncthreads();
    
    // Each thread validates a subset of nodes
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    
    for (uint32_t i = tid; i < node_count; i += stride) {
        // Check node type is valid
        if (static_cast<uint32_t>(ast_nodes[i].type) > 
            static_cast<uint32_t>(ASTNodeType::TraitDef)) {
            atomicAnd(reinterpret_cast<int*>(&shared_valid), 0);
        }
        
        // Check parent index is valid
        if (parent_indices[i] != 0xFFFFFFFF && parent_indices[i] >= node_count) {
            atomicAnd(reinterpret_cast<int*>(&shared_valid), 0);
        }
        
        // Root node should have no parent
        if (i == 0 && parent_indices[i] != 0xFFFFFFFF) {
            atomicAnd(reinterpret_cast<int*>(&shared_valid), 0);
        }
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        *is_valid = shared_valid;
    }
}

// Host functions to launch kernels
extern "C" void launch_ast_construction_kernel(
    const Token* tokens,
    uint32_t token_count,
    ASTNode* ast_nodes,
    uint32_t* ast_node_count,
    uint32_t* parent_indices,
    uint32_t max_nodes
) {
    // Configure kernel launch
    uint32_t threads_per_block = 256;
    uint32_t num_blocks = 1; // Simple version uses single block
    
    size_t shared_mem_size = sizeof(SharedASTMemory);
    
    ast_construction_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        tokens, token_count,
        ast_nodes, ast_node_count, parent_indices,
        max_nodes
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in ast_construction_kernel: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void launch_ast_validator_kernel(
    const ASTNode* ast_nodes,
    uint32_t node_count,
    const uint32_t* parent_indices,
    bool* is_valid
) {
    uint32_t threads_per_block = 256;
    uint32_t num_blocks = (node_count + threads_per_block - 1) / threads_per_block;
    
    ast_validator_kernel<<<num_blocks, threads_per_block>>>(
        ast_nodes, node_count, parent_indices, is_valid
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in ast_validator_kernel: %s\n", cudaGetErrorString(err));
    }
}

} // namespace rustg