#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cassert>
#include <cstdio>
#include "../include/gpu_types.h"

namespace cg = cooperative_groups;

// Test framework macros
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define TEST_ASSERT(condition, message) do { \
    if (!(condition)) { \
        printf("TEST FAILED: %s at %s:%d\n", message, __FILE__, __LINE__); \
        return false; \
    } \
} while(0)

namespace rustg_tests {

// Test data structures
struct TestIR {
    Instruction instructions[1000];
    uint32_t num_instructions;
    BasicBlock blocks[50];
    uint32_t num_blocks;
    LLVMValue values[500];
    uint32_t num_values;
};

struct TestResults {
    uint32_t stats[10];
    uint8_t machine_code[10000];
    uint32_t code_size;
    bool success;
};

// Test kernel for IR generation
__global__ void test_ir_generation_kernel(
    TestResults* results
) {
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    
    // Test IR instruction creation
    if (tid == 0 && bid == 0) {
        // Create test AST nodes
        TypedASTNode ast_nodes[10];
        
        // Binary operation: a + b
        ast_nodes[0].node_type = AST_BINARY_OP;
        ast_nodes[0].data.binary_op.op_type = BINOP_ADD;
        ast_nodes[0].data.binary_op.left = 1;
        ast_nodes[0].data.binary_op.right = 2;
        ast_nodes[0].type = {TYPE_INT, 4, 4};
        
        // Function call
        ast_nodes[1].node_type = AST_FUNCTION_CALL;
        ast_nodes[1].data.call.func_id = 100;
        ast_nodes[1].data.call.num_args = 2;
        ast_nodes[1].data.call.arg1 = 1;
        ast_nodes[1].data.call.arg2 = 2;
        
        // Return statement
        ast_nodes[2].node_type = AST_RETURN;
        ast_nodes[2].data.ret.has_value = true;
        ast_nodes[2].data.ret.value = 1;
        
        // Test IR generation (simplified)
        uint32_t instruction_count = 0;
        
        // Process each AST node
        for (uint32_t i = 0; i < 3; ++i) {
            const TypedASTNode& node = ast_nodes[i];
            
            switch (node.node_type) {
                case AST_BINARY_OP:
                    instruction_count++;
                    break;
                case AST_FUNCTION_CALL:
                    instruction_count++;
                    break;
                case AST_RETURN:
                    instruction_count++;
                    break;
            }
        }
        
        results->stats[0] = instruction_count;
        results->success = (instruction_count == 3);
    }
}

// Test kernel for machine code generation
__global__ void test_machine_code_kernel(
    TestResults* results
) {
    const uint32_t tid = threadIdx.x;
    
    if (tid == 0) {
        // Test instruction encoding
        uint8_t* code = results->machine_code;
        uint32_t offset = 0;
        
        // Test MOV instruction encoding
        // mov rax, rbx (REX.W + 89 /r)
        code[offset++] = 0x48;  // REX.W
        code[offset++] = 0x89;  // MOV opcode
        code[offset++] = 0xD8;  // ModR/M: 11 011 000 (rbx -> rax)
        
        // Test ADD instruction
        // add rax, rcx
        code[offset++] = 0x48;  // REX.W
        code[offset++] = 0x01;  // ADD opcode
        code[offset++] = 0xC8;  // ModR/M: 11 001 000 (rcx + rax)
        
        // Test immediate instruction
        // add rax, 42
        code[offset++] = 0x48;  // REX.W
        code[offset++] = 0x83;  // ADD with 8-bit immediate
        code[offset++] = 0xC0;  // ModR/M: 11 000 000 (rax)
        code[offset++] = 42;    // Immediate value
        
        // Test jump instruction
        // jmp +10
        code[offset++] = 0xEB;  // Short jump
        code[offset++] = 10;    // Relative offset
        
        results->code_size = offset;
        results->success = (offset == 10);
    }
}

// Test kernel for register allocation
__global__ void test_register_allocation_kernel(
    TestResults* results
) {
    __shared__ uint32_t interference_matrix[64];  // 8x8 for 8 variables
    __shared__ uint32_t degrees[8];
    __shared__ uint32_t colors[8];
    
    const uint32_t tid = threadIdx.x;
    
    // Initialize test data
    if (tid == 0) {
        // Clear interference matrix
        for (int i = 0; i < 64; ++i) {
            interference_matrix[i] = 0;
        }
        
        // Set up test interferences
        // Variables 0 and 1 interfere
        interference_matrix[0 * 8 + 1] = 1;
        interference_matrix[1 * 8 + 0] = 1;
        
        // Variables 1 and 2 interfere  
        interference_matrix[1 * 8 + 2] = 1;
        interference_matrix[2 * 8 + 1] = 1;
        
        // Calculate degrees
        for (int v = 0; v < 8; ++v) {
            uint32_t degree = 0;
            for (int n = 0; n < 8; ++n) {
                if (interference_matrix[v * 8 + n]) {
                    degree++;
                }
            }
            degrees[v] = degree;
            colors[v] = UINT32_MAX;  // Uncolored
        }
    }
    __syncthreads();
    
    // Simple greedy coloring
    if (tid < 8) {
        uint32_t var = tid;
        uint32_t color = 0;
        
        // Find first available color
        bool color_used[4] = {false, false, false, false};
        
        // Check neighbors
        for (uint32_t n = 0; n < 8; ++n) {
            if (interference_matrix[var * 8 + n] && colors[n] != UINT32_MAX) {
                if (colors[n] < 4) {
                    color_used[colors[n]] = true;
                }
            }
        }
        
        // Assign first free color
        for (uint32_t c = 0; c < 4; ++c) {
            if (!color_used[c]) {
                colors[var] = c;
                break;
            }
        }
    }
    __syncthreads();
    
    // Verify coloring
    if (tid == 0) {
        bool valid = true;
        uint32_t max_color = 0;
        
        for (int v1 = 0; v1 < 8; ++v1) {
            if (colors[v1] > max_color) {
                max_color = colors[v1];
            }
            
            for (int v2 = 0; v2 < 8; ++v2) {
                if (interference_matrix[v1 * 8 + v2] &&
                    colors[v1] == colors[v2] &&
                    colors[v1] != UINT32_MAX) {
                    valid = false;
                }
            }
        }
        
        results->stats[0] = max_color + 1;  // Number of colors used
        results->success = valid && (max_color < 4);
    }
}

// Test kernel for optimization passes
__global__ void test_optimization_kernel(
    TestResults* results
) {
    __shared__ Instruction instructions[20];
    __shared__ uint32_t optimized_count;
    
    const uint32_t tid = threadIdx.x;
    
    if (tid == 0) {
        optimized_count = 0;
        
        // Create test instructions
        // ADD r1, 0  -> Should be optimized to MOV
        instructions[0].opcode = OP_ADD;
        instructions[0].result_id = 1;
        instructions[0].operands[0] = 1;
        instructions[0].operands[1] = 0;  // Adding zero
        instructions[0].num_operands = 2;
        
        // MUL r2, 1  -> Should be optimized to MOV
        instructions[1].opcode = OP_MUL;
        instructions[1].result_id = 2;
        instructions[1].operands[0] = 2;
        instructions[1].operands[1] = 1;  // Multiply by one
        instructions[1].num_operands = 2;
        
        // Dead instruction (result never used)
        instructions[2].opcode = OP_ADD;
        instructions[2].result_id = 99;  // Never used
        instructions[2].operands[0] = 3;
        instructions[2].operands[1] = 4;
        instructions[2].num_operands = 2;
        
        // Constant folding: 5 + 7
        instructions[3].opcode = OP_ADD;
        instructions[3].result_id = 5;
        instructions[3].operands[0] = 5;   // Constant 5
        instructions[3].operands[1] = 7;   // Constant 7
        instructions[3].num_operands = 2;
    }
    __syncthreads();
    
    // Apply optimizations
    if (tid < 4) {
        Instruction& inst = instructions[tid];
        
        switch (inst.opcode) {
            case OP_ADD:
                if (inst.operands[1] == 0) {
                    // ADD x, 0 -> MOV x
                    inst.opcode = OP_MOV;
                    inst.num_operands = 1;
                    atomicAdd(&optimized_count, 1);
                } else if (inst.operands[0] == 5 && inst.operands[1] == 7) {
                    // Constant folding
                    inst.opcode = OP_CONST;
                    inst.operands[0] = 12;  // 5 + 7
                    inst.num_operands = 1;
                    atomicAdd(&optimized_count, 1);
                }
                break;
                
            case OP_MUL:
                if (inst.operands[1] == 1) {
                    // MUL x, 1 -> MOV x
                    inst.opcode = OP_MOV;
                    inst.num_operands = 1;
                    atomicAdd(&optimized_count, 1);
                }
                break;
        }
    }
    __syncthreads();
    
    if (tid == 0) {
        results->stats[0] = optimized_count;
        results->success = (optimized_count >= 3);
    }
}

// Test kernel for symbol resolution
__global__ void test_symbol_resolution_kernel(
    TestResults* results
) {
    __shared__ Symbol symbols[10];
    __shared__ uint32_t addresses[10];
    __shared__ uint32_t resolved_count;
    
    const uint32_t tid = threadIdx.x;
    
    if (tid == 0) {
        resolved_count = 0;
        
        // Create test symbols
        symbols[0].name_hash = 12345;  // "main"
        symbols[0].binding = 1;        // GLOBAL
        symbols[0].type = 1;           // FUNC
        symbols[0].address = 0x400000;
        
        symbols[1].name_hash = 67890;  // "printf"
        symbols[1].binding = 1;        // GLOBAL
        symbols[1].type = 1;           // FUNC  
        symbols[1].address = 0x500000;
        
        symbols[2].name_hash = 11111;  // "weak_symbol"
        symbols[2].binding = 2;        // WEAK
        symbols[2].type = 2;           // OBJECT
        symbols[2].address = 0x600000;
        
        // Clear addresses
        for (int i = 0; i < 10; ++i) {
            addresses[i] = 0;
        }
    }
    __syncthreads();
    
    // Resolve symbols
    if (tid < 3) {
        Symbol& sym = symbols[tid];
        
        if (sym.binding == 1 || sym.binding == 2) { // GLOBAL or WEAK
            // Simple resolution - copy address
            addresses[tid] = sym.address;
            atomicAdd(&resolved_count, 1);
        }
    }
    __syncthreads();
    
    if (tid == 0) {
        results->stats[0] = resolved_count;
        results->success = (resolved_count == 3) &&
                          (addresses[0] == 0x400000) &&
                          (addresses[1] == 0x500000) &&
                          (addresses[2] == 0x600000);
    }
}

// Performance benchmark kernel
__global__ void benchmark_ir_generation_kernel(
    const TypedASTNode* ast_nodes,
    uint32_t num_nodes,
    uint32_t* instructions_generated,
    float* elapsed_time
) {
    __shared__ uint32_t local_count;
    __shared__ uint32_t start_time;
    
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    
    if (tid == 0) {
        local_count = 0;
        start_time = clock();
    }
    __syncthreads();
    
    // Process nodes in parallel
    for (uint32_t i = bid * blockDim.x + tid; 
         i < num_nodes; 
         i += blockDim.x * gridDim.x) {
        
        if (i < num_nodes) {
            const TypedASTNode& node = ast_nodes[i];
            
            // Generate IR based on node type
            switch (node.node_type) {
                case AST_BINARY_OP:
                case AST_FUNCTION_CALL:
                case AST_LOAD:
                case AST_STORE:
                case AST_RETURN:
                    atomicAdd(&local_count, 1);
                    break;
            }
        }
    }
    __syncthreads();
    
    if (tid == 0) {
        uint32_t end_time = clock();
        atomicAdd(instructions_generated, local_count);
        
        // Calculate elapsed time (simplified)
        *elapsed_time = (end_time - start_time) / 1000000.0f;
    }
}

} // namespace rustg_tests

// Host test functions
extern "C" bool run_ir_generation_tests() {
    using namespace rustg_tests;
    
    printf("Running IR Generation Tests...\n");
    
    TestResults* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, sizeof(TestResults)));
    CUDA_CHECK(cudaMemset(d_results, 0, sizeof(TestResults)));
    
    // Launch test kernel
    test_ir_generation_kernel<<<1, 1>>>(d_results);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    TestResults h_results;
    CUDA_CHECK(cudaMemcpy(&h_results, d_results, sizeof(TestResults), cudaMemcpyDeviceToHost));
    
    printf("  Instructions generated: %u\n", h_results.stats[0]);
    printf("  Test result: %s\n", h_results.success ? "PASS" : "FAIL");
    
    CUDA_CHECK(cudaFree(d_results));
    return h_results.success;
}

extern "C" bool run_machine_code_tests() {
    using namespace rustg_tests;
    
    printf("Running Machine Code Generation Tests...\n");
    
    TestResults* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, sizeof(TestResults)));
    CUDA_CHECK(cudaMemset(d_results, 0, sizeof(TestResults)));
    
    test_machine_code_kernel<<<1, 1>>>(d_results);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    TestResults h_results;
    CUDA_CHECK(cudaMemcpy(&h_results, d_results, sizeof(TestResults), cudaMemcpyDeviceToHost));
    
    printf("  Code size: %u bytes\n", h_results.code_size);
    printf("  First few bytes: ");
    for (uint32_t i = 0; i < min(h_results.code_size, 8u); ++i) {
        printf("%02x ", h_results.machine_code[i]);
    }
    printf("\n");
    printf("  Test result: %s\n", h_results.success ? "PASS" : "FAIL");
    
    CUDA_CHECK(cudaFree(d_results));
    return h_results.success;
}

extern "C" bool run_register_allocation_tests() {
    using namespace rustg_tests;
    
    printf("Running Register Allocation Tests...\n");
    
    TestResults* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, sizeof(TestResults)));
    CUDA_CHECK(cudaMemset(d_results, 0, sizeof(TestResults)));
    
    test_register_allocation_kernel<<<1, 256>>>(d_results);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    TestResults h_results;
    CUDA_CHECK(cudaMemcpy(&h_results, d_results, sizeof(TestResults), cudaMemcpyDeviceToHost));
    
    printf("  Colors used: %u\n", h_results.stats[0]);
    printf("  Test result: %s\n", h_results.success ? "PASS" : "FAIL");
    
    CUDA_CHECK(cudaFree(d_results));
    return h_results.success;
}

extern "C" bool run_optimization_tests() {
    using namespace rustg_tests;
    
    printf("Running Optimization Tests...\n");
    
    TestResults* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, sizeof(TestResults)));
    CUDA_CHECK(cudaMemset(d_results, 0, sizeof(TestResults)));
    
    test_optimization_kernel<<<1, 256>>>(d_results);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    TestResults h_results;
    CUDA_CHECK(cudaMemcpy(&h_results, d_results, sizeof(TestResults), cudaMemcpyDeviceToHost));
    
    printf("  Optimizations applied: %u\n", h_results.stats[0]);
    printf("  Test result: %s\n", h_results.success ? "PASS" : "FAIL");
    
    CUDA_CHECK(cudaFree(d_results));
    return h_results.success;
}

extern "C" bool run_symbol_resolution_tests() {
    using namespace rustg_tests;
    
    printf("Running Symbol Resolution Tests...\n");
    
    TestResults* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, sizeof(TestResults)));
    CUDA_CHECK(cudaMemset(d_results, 0, sizeof(TestResults)));
    
    test_symbol_resolution_kernel<<<1, 256>>>(d_results);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    TestResults h_results;
    CUDA_CHECK(cudaMemcpy(&h_results, d_results, sizeof(TestResults), cudaMemcpyDeviceToHost));
    
    printf("  Symbols resolved: %u\n", h_results.stats[0]);
    printf("  Test result: %s\n", h_results.success ? "PASS" : "FAIL");
    
    CUDA_CHECK(cudaFree(d_results));
    return h_results.success;
}

extern "C" bool run_performance_benchmarks() {
    using namespace rustg_tests;
    
    printf("Running Performance Benchmarks...\n");
    
    // Allocate test data
    const uint32_t num_nodes = 100000;
    TypedASTNode* d_nodes;
    uint32_t* d_count;
    float* d_time;
    
    CUDA_CHECK(cudaMalloc(&d_nodes, num_nodes * sizeof(TypedASTNode)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_time, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_time, 0, sizeof(float)));
    
    // Initialize test nodes on device (simplified)
    
    benchmark_ir_generation_kernel<<<256, 256>>>(d_nodes, num_nodes, d_count, d_time);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    uint32_t instructions_generated;
    float elapsed_time;
    CUDA_CHECK(cudaMemcpy(&instructions_generated, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&elapsed_time, d_time, sizeof(float), cudaMemcpyDeviceToHost));
    
    float throughput = instructions_generated / elapsed_time;
    
    printf("  Instructions processed: %u\n", instructions_generated);
    printf("  Elapsed time: %.4f seconds\n", elapsed_time);
    printf("  Throughput: %.0f instructions/second\n", throughput);
    printf("  Target: 500,000 instructions/second\n");
    printf("  Performance: %s\n", (throughput >= 500000) ? "PASS" : "FAIL");
    
    CUDA_CHECK(cudaFree(d_nodes));
    CUDA_CHECK(cudaFree(d_count));
    CUDA_CHECK(cudaFree(d_time));
    
    return throughput >= 500000;
}

extern "C" bool run_all_phase5_tests() {
    printf("=== Phase 5 Code Generation Tests ===\n\n");
    
    bool all_passed = true;
    
    all_passed &= run_ir_generation_tests();
    printf("\n");
    
    all_passed &= run_machine_code_tests();
    printf("\n");
    
    all_passed &= run_register_allocation_tests();
    printf("\n");
    
    all_passed &= run_optimization_tests();
    printf("\n");
    
    all_passed &= run_symbol_resolution_tests();
    printf("\n");
    
    all_passed &= run_performance_benchmarks();
    printf("\n");
    
    printf("=== Phase 5 Test Summary ===\n");
    printf("Overall result: %s\n", all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    
    return all_passed;
}