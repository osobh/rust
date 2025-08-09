#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include "../../../include/gpu_types.h"

namespace cg = cooperative_groups;

namespace rustg {

// LLVM IR representation
enum OpCode : uint16_t {
    // Terminator instructions
    OP_RET,
    OP_BR,
    OP_SWITCH,
    OP_UNREACHABLE,
    
    // Binary operations
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_REM,
    OP_SHL,
    OP_SHR,
    OP_AND,
    OP_OR,
    OP_XOR,
    
    // Memory operations
    OP_ALLOCA,
    OP_LOAD,
    OP_STORE,
    OP_GEP,  // GetElementPtr
    
    // Conversion operations
    OP_TRUNC,
    OP_ZEXT,
    OP_SEXT,
    OP_BITCAST,
    OP_PTRTOINT,
    OP_INTTOPTR,
    
    // Other operations
    OP_ICMP,
    OP_FCMP,
    OP_PHI,
    OP_CALL,
    OP_SELECT
};

struct LLVMValue {
    uint32_t value_id;
    ValueKind kind;
    Type type;
    union {
        int64_t int_value;
        double float_value;
        uint32_t ref_id;
        struct {
            uint32_t func_id;
            uint32_t block_id;
        } block_ref;
    } data;
};

struct Instruction {
    OpCode opcode;
    uint32_t result_id;
    Type result_type;
    uint32_t operands[4];  // Most instructions have <= 4 operands
    uint32_t num_operands;
    uint32_t metadata;
    uint32_t debug_loc;
};

struct BasicBlock {
    uint32_t block_id;
    uint32_t function_id;
    uint32_t label_id;
    uint32_t* instructions;
    uint32_t instruction_count;
    uint32_t capacity;
    uint32_t* predecessors;
    uint32_t* successors;
    uint32_t num_preds;
    uint32_t num_succs;
    uint32_t dom_tree_parent;
    uint32_t loop_depth;
};

struct Function {
    uint32_t func_id;
    uint32_t name_hash;
    Type return_type;
    uint32_t* param_types;
    uint32_t num_params;
    BasicBlock* blocks;
    uint32_t num_blocks;
    uint32_t entry_block;
    uint32_t* local_values;
    uint32_t num_locals;
    uint32_t stack_size;
};

// Shared memory for IR generation
struct IRGenSharedMem {
    Instruction local_instructions[256];
    uint32_t instruction_count;
    LLVMValue local_values[128];
    uint32_t value_count;
    uint32_t current_block;
    uint32_t next_value_id;
    uint32_t next_block_id;
};

// Generate value ID
__device__ uint32_t allocate_value_id(IRGenSharedMem* shared) {
    return atomicAdd(&shared->next_value_id, 1);
}

// Create instruction
__device__ void emit_instruction(
    IRGenSharedMem* shared,
    OpCode opcode,
    Type result_type,
    uint32_t* operands,
    uint32_t num_operands
) {
    uint32_t idx = atomicAdd(&shared->instruction_count, 1);
    if (idx < 256) {
        Instruction& inst = shared->local_instructions[idx];
        inst.opcode = opcode;
        inst.result_id = allocate_value_id(shared);
        inst.result_type = result_type;
        inst.num_operands = min(num_operands, 4u);
        for (uint32_t i = 0; i < inst.num_operands; ++i) {
            inst.operands[i] = operands[i];
        }
    }
}

// Main IR generation kernel - processes typed AST to LLVM IR
__global__ void generate_ir_kernel(
    const TypedASTNode* ast_nodes,
    uint32_t num_nodes,
    Function* functions,
    uint32_t num_functions,
    BasicBlock* all_blocks,
    uint32_t max_blocks,
    Instruction* all_instructions,
    uint32_t max_instructions,
    LLVMValue* all_values,
    uint32_t max_values,
    uint32_t* stats  // [instructions_generated, blocks_created, values_created]
) {
    extern __shared__ char shared_mem_raw[];
    IRGenSharedMem* shared = reinterpret_cast<IRGenSharedMem*>(shared_mem_raw);
    
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // Initialize shared memory
    if (tid == 0) {
        shared->instruction_count = 0;
        shared->value_count = 0;
        shared->current_block = 0;
        shared->next_value_id = bid * 1000;  // Partition value ID space
        shared->next_block_id = bid * 10;    // Partition block ID space
    }
    __syncthreads();
    
    // Each block processes one function
    if (bid < num_functions) {
        Function& func = functions[bid];
        
        // Process function body AST nodes
        uint32_t func_start = func.func_id * 1000;  // Simplified indexing
        uint32_t func_end = min(func_start + 1000, num_nodes);
        
        // Generate entry block
        if (tid == 0) {
            uint32_t entry_block_id = atomicAdd(&shared->next_block_id, 1);
            BasicBlock& entry = all_blocks[entry_block_id];
            entry.block_id = entry_block_id;
            entry.function_id = func.func_id;
            entry.instruction_count = 0;
            entry.num_preds = 0;
            entry.num_succs = 0;
            func.entry_block = entry_block_id;
            shared->current_block = entry_block_id;
        }
        __syncthreads();
        
        // Process AST nodes in parallel
        for (uint32_t node_idx = func_start + tid; 
             node_idx < func_end; 
             node_idx += blockDim.x) {
            
            if (node_idx < num_nodes) {
                const TypedASTNode& node = ast_nodes[node_idx];
                
                switch (node.node_type) {
                    case AST_BINARY_OP: {
                        // Generate binary operation
                        OpCode op = OP_ADD;  // Map from AST op type
                        switch (node.data.binary_op.op_type) {
                            case BINOP_ADD: op = OP_ADD; break;
                            case BINOP_SUB: op = OP_SUB; break;
                            case BINOP_MUL: op = OP_MUL; break;
                            case BINOP_DIV: op = OP_DIV; break;
                            default: break;
                        }
                        
                        uint32_t operands[2] = {
                            node.data.binary_op.left,
                            node.data.binary_op.right
                        };
                        emit_instruction(shared, op, node.type, operands, 2);
                        break;
                    }
                    
                    case AST_FUNCTION_CALL: {
                        // Generate call instruction
                        uint32_t operands[4] = {
                            node.data.call.func_id,
                            node.data.call.arg1,
                            node.data.call.arg2,
                            node.data.call.arg3
                        };
                        emit_instruction(shared, OP_CALL, node.type, 
                                       operands, node.data.call.num_args + 1);
                        break;
                    }
                    
                    case AST_LOAD: {
                        // Generate load instruction
                        uint32_t operands[1] = { node.data.load.address };
                        emit_instruction(shared, OP_LOAD, node.type, operands, 1);
                        break;
                    }
                    
                    case AST_STORE: {
                        // Generate store instruction
                        uint32_t operands[2] = {
                            node.data.store.value,
                            node.data.store.address
                        };
                        Type void_type = { TYPE_VOID, 0, 0 };
                        emit_instruction(shared, OP_STORE, void_type, operands, 2);
                        break;
                    }
                    
                    case AST_RETURN: {
                        // Generate return instruction
                        uint32_t operands[1] = { node.data.ret.value };
                        emit_instruction(shared, OP_RET, func.return_type, 
                                       operands, node.data.ret.has_value ? 1 : 0);
                        break;
                    }
                    
                    case AST_IF: {
                        // Generate conditional branch
                        uint32_t true_block = atomicAdd(&shared->next_block_id, 1);
                        uint32_t false_block = atomicAdd(&shared->next_block_id, 1);
                        
                        uint32_t operands[3] = {
                            node.data.if_stmt.condition,
                            true_block,
                            false_block
                        };
                        emit_instruction(shared, OP_BR, { TYPE_VOID, 0, 0 }, 
                                       operands, 3);
                        break;
                    }
                    
                    case AST_ALLOCA: {
                        // Generate alloca for local variable
                        uint32_t operands[1] = { node.data.alloca.size };
                        emit_instruction(shared, OP_ALLOCA, 
                                       { TYPE_POINTER, node.type.size, 8 }, 
                                       operands, 1);
                        break;
                    }
                }
            }
        }
        __syncthreads();
        
        // Write instructions to global memory
        uint32_t global_inst_offset = func.func_id * 1000;
        for (uint32_t i = tid; i < shared->instruction_count; i += blockDim.x) {
            if (global_inst_offset + i < max_instructions) {
                all_instructions[global_inst_offset + i] = shared->local_instructions[i];
            }
        }
        
        // Update statistics
        if (tid == 0) {
            atomicAdd(&stats[0], shared->instruction_count);
            atomicAdd(&stats[1], shared->next_block_id - bid * 10);
            atomicAdd(&stats[2], shared->value_count);
        }
    }
}

// SSA construction kernel - converts to Static Single Assignment form
__global__ void construct_ssa_kernel(
    BasicBlock* blocks,
    uint32_t num_blocks,
    Instruction* instructions,
    uint32_t num_instructions,
    uint32_t* dominance_frontier,
    uint32_t* phi_nodes,
    uint32_t* phi_count
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
    
    // Each warp processes one basic block
    if (warp_id < num_blocks) {
        BasicBlock& block = blocks[warp_id];
        
        // Check if block is in dominance frontier
        for (uint32_t var = lane_id; var < 1024; var += 32) {
            bool needs_phi = false;
            
            // Check predecessors for different definitions
            uint32_t def_count = 0;
            uint32_t last_def = UINT32_MAX;
            
            for (uint32_t p = 0; p < block.num_preds; ++p) {
                uint32_t pred_block = block.predecessors[p];
                // Check if predecessor defines this variable
                // Simplified - would need def-use chains
                
                if (pred_block < num_blocks) {
                    def_count++;
                }
            }
            
            if (def_count > 1) {
                // Need PHI node
                needs_phi = true;
            }
            
            // Vote across warp
            uint32_t phi_mask = warp.ballot_sync(needs_phi);
            
            if (lane_id == 0 && phi_mask != 0) {
                // Insert PHI nodes
                uint32_t num_phis = __popc(phi_mask);
                uint32_t phi_offset = atomicAdd(phi_count, num_phis);
                
                // Record PHI node locations
                for (int i = 0; i < 32; ++i) {
                    if (phi_mask & (1 << i)) {
                        phi_nodes[phi_offset++] = block.block_id * 1024 + var + i;
                    }
                }
            }
        }
    }
}

// Control flow graph construction
__global__ void build_cfg_kernel(
    BasicBlock* blocks,
    uint32_t num_blocks,
    Instruction* instructions,
    uint32_t* cfg_edges,  // Adjacency matrix
    uint32_t* dom_tree,    // Dominance tree
    uint32_t* loop_headers // Loop header detection
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Build CFG edges from terminator instructions
    for (uint32_t b = tid; b < num_blocks; b += blockDim.x * gridDim.x) {
        BasicBlock& block = blocks[b];
        
        if (block.instruction_count > 0) {
            // Get last instruction (terminator)
            uint32_t last_inst_idx = block.instructions[block.instruction_count - 1];
            Instruction& terminator = instructions[last_inst_idx];
            
            switch (terminator.opcode) {
                case OP_BR:
                    if (terminator.num_operands == 1) {
                        // Unconditional branch
                        uint32_t target = terminator.operands[0];
                        cfg_edges[b * num_blocks + target] = 1;
                        
                        // Update successor/predecessor lists
                        blocks[target].predecessors[blocks[target].num_preds++] = b;
                        block.successors[block.num_succs++] = target;
                    } else if (terminator.num_operands == 3) {
                        // Conditional branch
                        uint32_t true_target = terminator.operands[1];
                        uint32_t false_target = terminator.operands[2];
                        
                        cfg_edges[b * num_blocks + true_target] = 1;
                        cfg_edges[b * num_blocks + false_target] = 1;
                        
                        blocks[true_target].predecessors[blocks[true_target].num_preds++] = b;
                        blocks[false_target].predecessors[blocks[false_target].num_preds++] = b;
                        block.successors[block.num_succs++] = true_target;
                        block.successors[block.num_succs++] = false_target;
                    }
                    break;
                    
                case OP_RET:
                case OP_UNREACHABLE:
                    // No successors
                    break;
                    
                case OP_SWITCH:
                    // Multiple targets
                    for (uint32_t i = 1; i < terminator.num_operands; ++i) {
                        uint32_t target = terminator.operands[i];
                        cfg_edges[b * num_blocks + target] = 1;
                        blocks[target].predecessors[blocks[target].num_preds++] = b;
                        block.successors[block.num_succs++] = target;
                    }
                    break;
            }
        }
        
        // Detect loop headers (back edges)
        for (uint32_t s = 0; s < block.num_succs; ++s) {
            uint32_t succ = block.successors[s];
            if (succ <= b) {  // Back edge
                loop_headers[succ] = 1;
                blocks[succ].loop_depth++;
            }
        }
    }
}

// Constant folding optimization
__global__ void constant_folding_kernel(
    Instruction* instructions,
    uint32_t num_instructions,
    LLVMValue* values,
    uint32_t* folded_count
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (uint32_t i = tid; i < num_instructions; i += blockDim.x * gridDim.x) {
        Instruction& inst = instructions[i];
        
        // Check if operands are constants
        bool all_constant = true;
        int64_t const_operands[4];
        
        for (uint32_t op = 0; op < inst.num_operands; ++op) {
            LLVMValue& val = values[inst.operands[op]];
            if (val.kind != VALUE_CONSTANT) {
                all_constant = false;
                break;
            }
            const_operands[op] = val.data.int_value;
        }
        
        if (all_constant && inst.num_operands >= 2) {
            int64_t result = 0;
            bool can_fold = true;
            
            // Perform constant folding
            switch (inst.opcode) {
                case OP_ADD:
                    result = const_operands[0] + const_operands[1];
                    break;
                case OP_SUB:
                    result = const_operands[0] - const_operands[1];
                    break;
                case OP_MUL:
                    result = const_operands[0] * const_operands[1];
                    break;
                case OP_DIV:
                    if (const_operands[1] != 0) {
                        result = const_operands[0] / const_operands[1];
                    } else {
                        can_fold = false;
                    }
                    break;
                case OP_AND:
                    result = const_operands[0] & const_operands[1];
                    break;
                case OP_OR:
                    result = const_operands[0] | const_operands[1];
                    break;
                case OP_XOR:
                    result = const_operands[0] ^ const_operands[1];
                    break;
                default:
                    can_fold = false;
            }
            
            if (can_fold) {
                // Replace instruction with constant
                values[inst.result_id].kind = VALUE_CONSTANT;
                values[inst.result_id].data.int_value = result;
                
                // Mark instruction as dead
                inst.opcode = OP_NOP;  // No-op
                
                atomicAdd(folded_count, 1);
            }
        }
    }
}

// Host launchers
extern "C" void launch_ir_generation(
    const TypedASTNode* ast_nodes,
    uint32_t num_nodes,
    Function* functions,
    uint32_t num_functions,
    BasicBlock* blocks,
    uint32_t max_blocks,
    Instruction* instructions,
    uint32_t max_instructions,
    LLVMValue* values,
    uint32_t max_values,
    uint32_t* stats
) {
    uint32_t threads = 256;
    uint32_t blocks_needed = num_functions;
    size_t shared_mem = sizeof(IRGenSharedMem);
    
    generate_ir_kernel<<<blocks_needed, threads, shared_mem>>>(
        ast_nodes, num_nodes,
        functions, num_functions,
        blocks, max_blocks,
        instructions, max_instructions,
        values, max_values,
        stats
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in ir_generation: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void launch_ssa_construction(
    BasicBlock* blocks,
    uint32_t num_blocks,
    Instruction* instructions,
    uint32_t num_instructions,
    uint32_t* dominance_frontier,
    uint32_t* phi_nodes,
    uint32_t* phi_count
) {
    uint32_t threads = 256;
    uint32_t blocks_needed = (num_blocks * 32 + threads - 1) / threads;
    
    construct_ssa_kernel<<<blocks_needed, threads>>>(
        blocks, num_blocks,
        instructions, num_instructions,
        dominance_frontier,
        phi_nodes, phi_count
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in ssa_construction: %s\n", cudaGetErrorString(err));
    }
}

} // namespace rustg