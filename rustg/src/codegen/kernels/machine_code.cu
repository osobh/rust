#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include "../../../include/gpu_types.h"

namespace cg = cooperative_groups;

namespace rustg {

// x86_64 instruction encoding
enum X86Opcode : uint8_t {
    // Data movement
    X86_MOV_REG_REG = 0x89,
    X86_MOV_REG_IMM = 0xB8,
    X86_MOV_MEM_REG = 0x89,
    X86_MOV_REG_MEM = 0x8B,
    X86_LEA = 0x8D,
    
    // Arithmetic
    X86_ADD = 0x01,
    X86_SUB = 0x29,
    X86_IMUL = 0xAF,
    X86_IDIV = 0xF7,
    X86_INC = 0xFF,
    X86_DEC = 0xFF,
    X86_NEG = 0xF7,
    
    // Logic
    X86_AND = 0x21,
    X86_OR = 0x09,
    X86_XOR = 0x31,
    X86_NOT = 0xF7,
    X86_SHL = 0xD3,
    X86_SHR = 0xD3,
    X86_SAR = 0xD3,
    
    // Control flow
    X86_JMP = 0xE9,
    X86_JE = 0x84,
    X86_JNE = 0x85,
    X86_JL = 0x8C,
    X86_JLE = 0x8E,
    X86_JG = 0x8F,
    X86_JGE = 0x8D,
    X86_CALL = 0xE8,
    X86_RET = 0xC3,
    
    // Stack operations
    X86_PUSH = 0x50,
    X86_POP = 0x58,
    
    // Comparison
    X86_CMP = 0x39,
    X86_TEST = 0x85
};

// Register encoding
enum X86Register : uint8_t {
    RAX = 0, RCX = 1, RDX = 2, RBX = 3,
    RSP = 4, RBP = 5, RSI = 6, RDI = 7,
    R8 = 8, R9 = 9, R10 = 10, R11 = 11,
    R12 = 12, R13 = 13, R14 = 14, R15 = 15
};

struct MachineInstr {
    X86Opcode opcode;
    uint8_t prefix[4];     // REX prefix and others
    uint8_t prefix_len;
    uint8_t modrm;         // ModR/M byte
    uint8_t sib;           // SIB byte
    bool has_sib;
    int32_t displacement;
    int8_t disp_size;      // 0, 1, or 4 bytes
    int64_t immediate;
    int8_t imm_size;       // 0, 1, 4, or 8 bytes
    uint32_t size;         // Total instruction size
    uint32_t address;      // Address in code segment
};

struct CodeBuffer {
    uint8_t* code;
    uint32_t size;
    uint32_t capacity;
    uint32_t* relocation_offsets;
    uint32_t* relocation_symbols;
    uint32_t num_relocations;
};

// Shared memory for machine code generation
struct MachineCodeSharedMem {
    uint8_t local_code[1024];      // Local code buffer
    uint32_t code_offset;
    MachineInstr local_instrs[128];
    uint32_t instr_count;
    uint32_t* jump_targets;         // Jump target resolution
    uint32_t num_jumps;
};

// Encode ModR/M byte
__device__ uint8_t encode_modrm(uint8_t mod, uint8_t reg, uint8_t rm) {
    return (mod << 6) | (reg << 3) | rm;
}

// Encode REX prefix for 64-bit mode
__device__ uint8_t encode_rex(bool w, bool r, bool x, bool b) {
    uint8_t rex = 0x40;
    if (w) rex |= 0x08;  // 64-bit operand
    if (r) rex |= 0x04;  // Extension of ModR/M reg field
    if (x) rex |= 0x02;  // Extension of SIB index field
    if (b) rex |= 0x01;  // Extension of ModR/M r/m, SIB base, or opcode reg
    return rex;
}

// Emit instruction bytes to buffer
__device__ void emit_bytes(
    MachineCodeSharedMem* shared,
    const uint8_t* bytes,
    uint32_t count
) {
    uint32_t offset = atomicAdd(&shared->code_offset, count);
    if (offset + count <= 1024) {
        for (uint32_t i = 0; i < count; ++i) {
            shared->local_code[offset + i] = bytes[i];
        }
    }
}

// Generate move instruction
__device__ void emit_mov(
    MachineCodeSharedMem* shared,
    X86Register dst,
    X86Register src
) {
    uint8_t bytes[3];
    uint32_t len = 0;
    
    // REX prefix for 64-bit
    if (dst >= R8 || src >= R8) {
        bytes[len++] = encode_rex(true, src >= R8, false, dst >= R8);
    } else {
        bytes[len++] = 0x48;  // REX.W for 64-bit
    }
    
    // Opcode
    bytes[len++] = X86_MOV_REG_REG;
    
    // ModR/M byte (11 = register direct)
    bytes[len++] = encode_modrm(0x3, src & 7, dst & 7);
    
    emit_bytes(shared, bytes, len);
}

// Generate arithmetic instruction
__device__ void emit_arithmetic(
    MachineCodeSharedMem* shared,
    X86Opcode op,
    X86Register dst,
    X86Register src
) {
    uint8_t bytes[3];
    uint32_t len = 0;
    
    // REX prefix
    if (dst >= R8 || src >= R8) {
        bytes[len++] = encode_rex(true, src >= R8, false, dst >= R8);
    } else {
        bytes[len++] = 0x48;
    }
    
    // Opcode
    bytes[len++] = op;
    
    // ModR/M
    bytes[len++] = encode_modrm(0x3, src & 7, dst & 7);
    
    emit_bytes(shared, bytes, len);
}

// Generate immediate instruction
__device__ void emit_immediate(
    MachineCodeSharedMem* shared,
    X86Opcode op,
    X86Register reg,
    int32_t immediate
) {
    uint8_t bytes[10];
    uint32_t len = 0;
    
    // REX prefix if needed
    if (reg >= R8) {
        bytes[len++] = encode_rex(true, false, false, reg >= R8);
    } else {
        bytes[len++] = 0x48;
    }
    
    // Opcode (ADD/SUB with immediate)
    if (op == X86_ADD) {
        bytes[len++] = 0x81;
    } else if (op == X86_SUB) {
        bytes[len++] = 0x81;
    }
    
    // ModR/M
    uint8_t opcode_ext = (op == X86_ADD) ? 0 : 5;
    bytes[len++] = encode_modrm(0x3, opcode_ext, reg & 7);
    
    // Immediate value (32-bit)
    *reinterpret_cast<int32_t*>(&bytes[len]) = immediate;
    len += 4;
    
    emit_bytes(shared, bytes, len);
}

// Generate jump instruction
__device__ void emit_jump(
    MachineCodeSharedMem* shared,
    X86Opcode jump_type,
    int32_t target_offset
) {
    uint8_t bytes[6];
    uint32_t len = 0;
    
    if (jump_type == X86_JMP) {
        // Unconditional jump
        bytes[len++] = 0xE9;
        *reinterpret_cast<int32_t*>(&bytes[len]) = target_offset;
        len += 4;
    } else {
        // Conditional jump (near)
        bytes[len++] = 0x0F;
        bytes[len++] = jump_type;
        *reinterpret_cast<int32_t*>(&bytes[len]) = target_offset;
        len += 4;
    }
    
    emit_bytes(shared, bytes, len);
}

// Main machine code generation kernel
__global__ void generate_machine_code_kernel(
    const Instruction* ir_instructions,
    uint32_t num_instructions,
    const BasicBlock* blocks,
    uint32_t num_blocks,
    const RegisterAllocation* reg_alloc,
    CodeBuffer* output,
    uint32_t* stats  // [bytes_generated, instructions_generated]
) {
    extern __shared__ char shared_mem_raw[];
    MachineCodeSharedMem* shared = reinterpret_cast<MachineCodeSharedMem*>(shared_mem_raw);
    
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // Initialize shared memory
    if (tid == 0) {
        shared->code_offset = 0;
        shared->instr_count = 0;
        shared->num_jumps = 0;
    }
    __syncthreads();
    
    // Each block processes one basic block
    if (bid < num_blocks) {
        const BasicBlock& bb = blocks[bid];
        
        // Generate function prologue if entry block
        if (bid == 0 && tid == 0) {
            // push rbp
            uint8_t push_rbp[] = {0x55};
            emit_bytes(shared, push_rbp, 1);
            
            // mov rbp, rsp
            uint8_t mov_rbp_rsp[] = {0x48, 0x89, 0xE5};
            emit_bytes(shared, mov_rbp_rsp, 3);
            
            // sub rsp, stack_size (allocate stack frame)
            uint8_t sub_rsp[] = {0x48, 0x83, 0xEC, 0x20};  // sub rsp, 32
            emit_bytes(shared, sub_rsp, 4);
        }
        __syncthreads();
        
        // Process instructions in parallel
        for (uint32_t i = bb.instructions[0] + tid; 
             i < bb.instructions[0] + bb.instruction_count;
             i += blockDim.x) {
            
            if (i < num_instructions) {
                const Instruction& inst = ir_instructions[i];
                
                // Map virtual registers to physical registers
                X86Register dst_reg = RAX;
                X86Register src1_reg = RCX;
                X86Register src2_reg = RDX;
                
                if (reg_alloc) {
                    dst_reg = static_cast<X86Register>(
                        reg_alloc->allocation[inst.result_id] & 0xF
                    );
                    if (inst.num_operands > 0) {
                        src1_reg = static_cast<X86Register>(
                            reg_alloc->allocation[inst.operands[0]] & 0xF
                        );
                    }
                    if (inst.num_operands > 1) {
                        src2_reg = static_cast<X86Register>(
                            reg_alloc->allocation[inst.operands[1]] & 0xF
                        );
                    }
                }
                
                // Generate machine code based on IR opcode
                switch (inst.opcode) {
                    case OP_ADD:
                        // mov dst, src1
                        emit_mov(shared, dst_reg, src1_reg);
                        // add dst, src2
                        emit_arithmetic(shared, X86_ADD, dst_reg, src2_reg);
                        break;
                        
                    case OP_SUB:
                        emit_mov(shared, dst_reg, src1_reg);
                        emit_arithmetic(shared, X86_SUB, dst_reg, src2_reg);
                        break;
                        
                    case OP_MUL:
                        // mov rax, src1
                        emit_mov(shared, RAX, src1_reg);
                        // imul src2
                        uint8_t imul_bytes[] = {0x48, 0xF7, 0xE0 | (src2_reg & 7)};
                        emit_bytes(shared, imul_bytes, 3);
                        // mov dst, rax
                        emit_mov(shared, dst_reg, RAX);
                        break;
                        
                    case OP_LOAD:
                        // mov dst, [src]
                        {
                            uint8_t load_bytes[3];
                            load_bytes[0] = 0x48;  // REX.W
                            load_bytes[1] = 0x8B;  // MOV r64, r/m64
                            load_bytes[2] = encode_modrm(0, dst_reg & 7, src1_reg & 7);
                            emit_bytes(shared, load_bytes, 3);
                        }
                        break;
                        
                    case OP_STORE:
                        // mov [dst], src
                        {
                            uint8_t store_bytes[3];
                            store_bytes[0] = 0x48;
                            store_bytes[1] = 0x89;
                            store_bytes[2] = encode_modrm(0, src1_reg & 7, src2_reg & 7);
                            emit_bytes(shared, store_bytes, 3);
                        }
                        break;
                        
                    case OP_BR:
                        if (inst.num_operands == 1) {
                            // Unconditional branch
                            emit_jump(shared, X86_JMP, 0);  // Fixup later
                        } else {
                            // Conditional branch
                            // test condition, condition
                            uint8_t test_bytes[] = {0x48, 0x85, 0xC0};
                            emit_bytes(shared, test_bytes, 3);
                            // je false_target
                            emit_jump(shared, X86_JE, 0);  // Fixup later
                            // jmp true_target
                            emit_jump(shared, X86_JMP, 0);  // Fixup later
                        }
                        break;
                        
                    case OP_RET:
                        // Function epilogue
                        // mov rsp, rbp
                        {
                            uint8_t mov_rsp_rbp[] = {0x48, 0x89, 0xEC};
                            emit_bytes(shared, mov_rsp_rbp, 3);
                        }
                        // pop rbp
                        {
                            uint8_t pop_rbp[] = {0x5D};
                            emit_bytes(shared, pop_rbp, 1);
                        }
                        // ret
                        {
                            uint8_t ret[] = {0xC3};
                            emit_bytes(shared, ret, 1);
                        }
                        break;
                        
                    case OP_CALL:
                        // Save caller-saved registers if needed
                        // call target
                        {
                            uint8_t call_bytes[5];
                            call_bytes[0] = 0xE8;
                            *reinterpret_cast<int32_t*>(&call_bytes[1]) = 0;  // Fixup
                            emit_bytes(shared, call_bytes, 5);
                        }
                        break;
                        
                    case OP_ALLOCA:
                        // sub rsp, size
                        {
                            uint8_t sub_bytes[7];
                            sub_bytes[0] = 0x48;
                            sub_bytes[1] = 0x81;
                            sub_bytes[2] = 0xEC;
                            *reinterpret_cast<int32_t*>(&sub_bytes[3]) = 
                                inst.operands[0];  // Stack size
                            emit_bytes(shared, sub_bytes, 7);
                        }
                        // mov dst, rsp
                        emit_mov(shared, dst_reg, RSP);
                        break;
                }
                
                atomicAdd(&shared->instr_count, 1);
            }
        }
        __syncthreads();
        
        // Write code to global memory
        if (tid == 0) {
            uint32_t global_offset = atomicAdd(&output->size, shared->code_offset);
            if (global_offset + shared->code_offset <= output->capacity) {
                for (uint32_t i = 0; i < shared->code_offset; ++i) {
                    output->code[global_offset + i] = shared->local_code[i];
                }
            }
            
            // Update statistics
            atomicAdd(&stats[0], shared->code_offset);
            atomicAdd(&stats[1], shared->instr_count);
        }
    }
}

// Instruction selection kernel - maps IR to machine instructions
__global__ void instruction_selection_kernel(
    const Instruction* ir,
    uint32_t num_instructions,
    MachineInstr* machine_instrs,
    uint32_t* selected_count
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_instructions) {
        const Instruction& ir_inst = ir[tid];
        MachineInstr& machine = machine_instrs[tid];
        
        // Select appropriate x86 instruction
        switch (ir_inst.opcode) {
            case OP_ADD:
                machine.opcode = X86_ADD;
                machine.size = 3;  // Basic add is 3 bytes
                break;
            case OP_SUB:
                machine.opcode = X86_SUB;
                machine.size = 3;
                break;
            case OP_MUL:
                machine.opcode = X86_IMUL;
                machine.size = 4;  // IMUL is longer
                break;
            case OP_DIV:
                machine.opcode = X86_IDIV;
                machine.size = 3;
                break;
            case OP_AND:
                machine.opcode = X86_AND;
                machine.size = 3;
                break;
            case OP_OR:
                machine.opcode = X86_OR;
                machine.size = 3;
                break;
            case OP_XOR:
                machine.opcode = X86_XOR;
                machine.size = 3;
                break;
            case OP_LOAD:
                machine.opcode = X86_MOV_REG_MEM;
                machine.size = 3;
                break;
            case OP_STORE:
                machine.opcode = X86_MOV_MEM_REG;
                machine.size = 3;
                break;
            case OP_BR:
                machine.opcode = (ir_inst.num_operands == 1) ? X86_JMP : X86_JE;
                machine.size = 5;  // Near jump
                break;
            case OP_CALL:
                machine.opcode = X86_CALL;
                machine.size = 5;
                break;
            case OP_RET:
                machine.opcode = X86_RET;
                machine.size = 1;
                break;
            default:
                machine.opcode = X86_NOP;  // No-op
                machine.size = 1;
        }
        
        atomicAdd(selected_count, 1);
    }
}

// Peephole optimization kernel
__global__ void peephole_optimization_kernel(
    MachineInstr* instructions,
    uint32_t num_instructions,
    uint32_t* optimized_count
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Look for patterns to optimize
    if (tid < num_instructions - 1) {
        MachineInstr& curr = instructions[tid];
        MachineInstr& next = instructions[tid + 1];
        
        // Pattern: mov reg1, reg2; mov reg2, reg1 -> swap
        if (curr.opcode == X86_MOV_REG_REG && 
            next.opcode == X86_MOV_REG_REG) {
            // Check if it's a redundant move pattern
            // Optimize by removing one or both
        }
        
        // Pattern: add reg, 0 -> nop
        if (curr.opcode == X86_ADD && curr.immediate == 0) {
            curr.opcode = X86_NOP;
            curr.size = 1;
            atomicAdd(optimized_count, 1);
        }
        
        // Pattern: mul reg, 1 -> nop
        if (curr.opcode == X86_IMUL && curr.immediate == 1) {
            curr.opcode = X86_NOP;
            curr.size = 1;
            atomicAdd(optimized_count, 1);
        }
        
        // Pattern: jmp next_instruction -> nop
        if (curr.opcode == X86_JMP && curr.displacement == curr.size) {
            curr.opcode = X86_NOP;
            curr.size = 1;
            atomicAdd(optimized_count, 1);
        }
    }
}

// Host launchers
extern "C" void launch_machine_code_generation(
    const Instruction* ir_instructions,
    uint32_t num_instructions,
    const BasicBlock* blocks,
    uint32_t num_blocks,
    const RegisterAllocation* reg_alloc,
    CodeBuffer* output,
    uint32_t* stats
) {
    uint32_t threads = 256;
    uint32_t blocks_needed = num_blocks;
    size_t shared_mem = sizeof(MachineCodeSharedMem);
    
    generate_machine_code_kernel<<<blocks_needed, threads, shared_mem>>>(
        ir_instructions, num_instructions,
        blocks, num_blocks,
        reg_alloc,
        output, stats
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in machine_code_generation: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void launch_instruction_selection(
    const Instruction* ir,
    uint32_t num_instructions,
    MachineInstr* machine_instrs,
    uint32_t* selected_count
) {
    uint32_t threads = 256;
    uint32_t blocks = (num_instructions + threads - 1) / threads;
    
    instruction_selection_kernel<<<blocks, threads>>>(
        ir, num_instructions,
        machine_instrs, selected_count
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in instruction_selection: %s\n", cudaGetErrorString(err));
    }
}

} // namespace rustg