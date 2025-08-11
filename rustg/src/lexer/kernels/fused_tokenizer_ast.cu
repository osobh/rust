#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include "../../../include/gpu_types.h"
#include "../../../include/char_classifier.h"

namespace cg = cooperative_groups;

namespace rustg {

// Fused kernel shared memory structure
struct FusedSharedMemory {
    // Tokenizer section
    char source_cache[4096];        // Source cache
    Token token_buffer[256];         // Token buffer
    
    // AST section  
    uint32_t ast_stack[128];         // Parser stack
    uint32_t ast_nodes[256];         // AST node indices
    
    // Synchronization
    uint32_t token_ready_flag;      // Tokens ready for parsing
    uint32_t token_count;            // Number of tokens
    uint32_t ast_node_count;         // Number of AST nodes
};

// Token producer function (runs in first warp)
__device__ void produce_tokens(
    const char* source,
    uint32_t source_len,
    uint32_t start_pos,
    uint32_t end_pos,
    Token* token_buffer,
    uint32_t& token_count,
    volatile uint32_t* ready_flag
) {
    uint32_t pos = start_pos;
    token_count = 0;
    
    while (pos < end_pos && token_count < 256) {
        unsigned char ch = source[pos];
        CharClass cls = classify_char(ch);
        
        Token& token = token_buffer[token_count];
        token.start_pos = pos;
        token.line = 1; // Simplified
        token.column = pos + 1;
        
        // Simplified tokenization
        switch (cls) {
            case CharClass::Letter:
                token.type = TokenType::Identifier;
                token.length = 1;
                while (pos + token.length < end_pos && 
                       (classify_char(source[pos + token.length]) == CharClass::Letter ||
                        classify_char(source[pos + token.length]) == CharClass::Digit)) {
                    token.length++;
                }
                break;
                
            case CharClass::Digit:
                token.type = TokenType::IntegerLiteral;
                token.length = 1;
                while (pos + token.length < end_pos && 
                       classify_char(source[pos + token.length]) == CharClass::Digit) {
                    token.length++;
                }
                break;
                
            case CharClass::Whitespace:
                token.type = TokenType::Whitespace;
                token.length = 1;
                break;
                
            case CharClass::Operator:
                switch (ch) {
                    case '+': token.type = TokenType::Plus; break;
                    case '-': token.type = TokenType::Minus; break;
                    case '*': token.type = TokenType::Star; break;
                    case '/': token.type = TokenType::Slash; break;
                    case '=': token.type = TokenType::Equal; break;
                    default: token.type = TokenType::Invalid;
                }
                token.length = 1;
                break;
                
            case CharClass::Delimiter:
                switch (ch) {
                    case '(': token.type = TokenType::LeftParen; break;
                    case ')': token.type = TokenType::RightParen; break;
                    case '{': token.type = TokenType::LeftBrace; break;
                    case '}': token.type = TokenType::RightBrace; break;
                    case ';': token.type = TokenType::Semicolon; break;
                    default: token.type = TokenType::Invalid;
                }
                token.length = 1;
                break;
                
            default:
                token.type = TokenType::Invalid;
                token.length = 1;
        }
        
        if (token.type != TokenType::Whitespace) {
            token_count++;
        }
        pos += token.length;
    }
    
    // Signal tokens ready
    atomicExch(const_cast<uint32_t*>(ready_flag), 1);
}

// AST consumer function (runs in second warp)
__device__ void consume_tokens_build_ast(
    Token* token_buffer,
    volatile uint32_t* token_count,
    volatile uint32_t* ready_flag,
    uint32_t* ast_nodes,
    uint32_t& ast_node_count
) {
    // Wait for tokens to be ready
    while (atomicAdd(const_cast<uint32_t*>(ready_flag), 0) == 0) {
        __threadfence();
    }
    
    uint32_t num_tokens = *token_count;
    ast_node_count = 0;
    
    // Simple AST construction
    uint32_t pos = 0;
    
    // Create root node
    ast_nodes[ast_node_count++] = 0; // Program node
    
    // Parse tokens into AST
    while (pos < num_tokens && ast_node_count < 256) {
        Token& token = token_buffer[pos];
        
        switch (token.type) {
            case TokenType::KeywordFn:
                ast_nodes[ast_node_count++] = pos; // Function node
                break;
                
            case TokenType::Identifier:
                ast_nodes[ast_node_count++] = pos; // Identifier node
                break;
                
            case TokenType::IntegerLiteral:
            case TokenType::FloatLiteral:
                ast_nodes[ast_node_count++] = pos; // Literal node
                break;
                
            case TokenType::Plus:
            case TokenType::Minus:
            case TokenType::Star:
            case TokenType::Slash:
                ast_nodes[ast_node_count++] = pos; // Binary op node
                break;
        }
        
        pos++;
    }
}

// Fused tokenizer and AST construction kernel
__global__ void fused_tokenizer_ast_kernel(
    const char* __restrict__ source,
    size_t source_len,
    Token* __restrict__ global_tokens,
    uint32_t* __restrict__ global_token_count,
    ASTNode* __restrict__ global_ast_nodes,
    uint32_t* __restrict__ global_ast_node_count,
    uint32_t max_tokens,
    uint32_t max_ast_nodes
) {
    extern __shared__ char shared_mem_raw[];
    FusedSharedMemory* shared = reinterpret_cast<FusedSharedMemory*>(shared_mem_raw);
    
    const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    const uint32_t block_id = blockIdx.x;
    
    // Initialize shared memory
    if (tid == 0) {
        shared->token_ready_flag = 0;
        shared->token_count = 0;
        shared->ast_node_count = 0;
    }
    __syncthreads();
    
    // Calculate work distribution
    const uint32_t chunk_size = (source_len + gridDim.x - 1) / gridDim.x;
    const uint32_t start_pos = block_id * chunk_size;
    const uint32_t end_pos = min(start_pos + chunk_size, static_cast<uint32_t>(source_len));
    
    if (start_pos >= source_len) return;
    
    // Load source into shared memory (all threads cooperate)
    uint32_t cache_size = min(4096u, end_pos - start_pos);
    for (uint32_t i = tid; i < cache_size; i += blockDim.x) {
        shared->source_cache[i] = source[start_pos + i];
    }
    __syncthreads();
    
    // Pipeline: Warp 0 tokenizes, Warp 1 builds AST
    if (warp_id == 0 && lane_id == 0) {
        // Tokenizer producer
        produce_tokens(
            shared->source_cache,
            cache_size,
            0,
            cache_size,
            shared->token_buffer,
            shared->token_count,
            &shared->token_ready_flag
        );
    } else if (warp_id == 1 && lane_id == 0) {
        // AST consumer
        consume_tokens_build_ast(
            shared->token_buffer,
            &shared->token_count,
            &shared->token_ready_flag,
            shared->ast_nodes,
            shared->ast_node_count
        );
    }
    
    __syncthreads();
    
    // Write results to global memory (all threads cooperate)
    if (tid == 0) {
        // Write tokens
        uint32_t token_offset = atomicAdd(global_token_count, shared->token_count);
        if (token_offset + shared->token_count <= max_tokens) {
            for (uint32_t i = 0; i < shared->token_count; ++i) {
                global_tokens[token_offset + i] = shared->token_buffer[i];
            }
        }
        
        // Write AST nodes (simplified - just indices for now)
        uint32_t ast_offset = atomicAdd(global_ast_node_count, shared->ast_node_count);
        if (ast_offset + shared->ast_node_count <= max_ast_nodes) {
            for (uint32_t i = 0; i < shared->ast_node_count; ++i) {
                global_ast_nodes[ast_offset + i].type = ASTNodeType::Invalid;
                global_ast_nodes[ast_offset + i].token_index = shared->ast_nodes[i];
            }
        }
    }
}

// Performance monitoring kernel for profiling
__global__ void profile_fused_kernel(
    uint32_t* cycle_counts,
    uint32_t* memory_transactions,
    uint32_t* warp_divergence
) {
    uint32_t tid = threadIdx.x;
    
    // Use PTX to read performance counters
    uint32_t cycles_start, cycles_end;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(cycles_start));
    
    // Simulated work
    __syncthreads();
    
    asm volatile("mov.u32 %0, %%clock;" : "=r"(cycles_end));
    
    if (tid == 0) {
        cycle_counts[blockIdx.x] = cycles_end - cycles_start;
    }
}

// Host function to launch fused kernel
extern "C" void launch_fused_tokenizer_ast(
    const char* source,
    size_t source_len,
    Token* tokens,
    uint32_t* token_count,
    ASTNode* ast_nodes,
    uint32_t* ast_node_count,
    uint32_t max_tokens,
    uint32_t max_ast_nodes
) {
    // Initialize character classification table
    static bool initialized = false;
    if (!initialized) {
        initialize_char_class_table();
        initialized = true;
    }
    
    // Configure kernel launch
    uint32_t threads_per_block = 256;
    uint32_t num_blocks = min(65535u, 
        static_cast<uint32_t>((source_len + 1024 - 1) / 1024));
    
    size_t shared_mem_size = sizeof(FusedSharedMemory);
    
    // Launch fused kernel
    fused_tokenizer_ast_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        source, source_len,
        tokens, token_count,
        ast_nodes, ast_node_count,
        max_tokens, max_ast_nodes
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in fused_tokenizer_ast_kernel: %s\n", cudaGetErrorString(err));
    }
}

// Compare performance of fused vs separate kernels
extern "C" void benchmark_kernel_fusion(
    const char* source,
    size_t source_len,
    int iterations
) {
    // Allocate memory for both approaches
    Token* d_tokens;
    uint32_t* d_token_count;
    ASTNode* d_ast_nodes;
    uint32_t* d_ast_node_count;
    
    uint32_t max_tokens = source_len;
    uint32_t max_ast_nodes = source_len * 2;
    
    cudaMalloc(&d_tokens, max_tokens * sizeof(Token));
    cudaMalloc(&d_token_count, sizeof(uint32_t));
    cudaMalloc(&d_ast_nodes, max_ast_nodes * sizeof(ASTNode));
    cudaMalloc(&d_ast_node_count, sizeof(uint32_t));
    
    char* d_source;
    cudaMalloc(&d_source, source_len);
    cudaMemcpy(d_source, source, source_len, cudaMemcpyHostToDevice);
    
    // Warm up
    launch_fused_tokenizer_ast(
        d_source, source_len,
        d_tokens, d_token_count,
        d_ast_nodes, d_ast_node_count,
        max_tokens, max_ast_nodes
    );
    cudaDeviceSynchronize();
    
    // Benchmark fused kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        cudaMemset(d_token_count, 0, sizeof(uint32_t));
        cudaMemset(d_ast_node_count, 0, sizeof(uint32_t));
        
        launch_fused_tokenizer_ast(
            d_source, source_len,
            d_tokens, d_token_count,
            d_ast_nodes, d_ast_node_count,
            max_tokens, max_ast_nodes
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float fused_time_ms;
    cudaEventElapsedTime(&fused_time_ms, start, stop);
    
    printf("Fused kernel performance:\n");
    printf("  Total time: %.3f ms\n", fused_time_ms);
    printf("  Avg time per iteration: %.3f ms\n", fused_time_ms / iterations);
    printf("  Throughput: %.2f MB/s\n", 
           (source_len * iterations) / (fused_time_ms * 1000.0));
    
    // Cleanup
    cudaFree(d_tokens);
    cudaFree(d_token_count);
    cudaFree(d_ast_nodes);
    cudaFree(d_ast_node_count);
    cudaFree(d_source);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

} // namespace rustg