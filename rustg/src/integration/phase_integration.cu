#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include "../../include/gpu_types.h"

namespace cg = cooperative_groups;

namespace rustg {

// Unified pipeline data structure
struct PipelineState {
    // Phase 1 outputs
    Token* tokens;
    uint32_t num_tokens;
    ASTNode* ast_nodes;
    uint32_t num_ast_nodes;
    
    // Phase 2 outputs
    Token* expanded_tokens;
    uint32_t num_expanded_tokens;
    MacroExpansion* expansions;
    uint32_t num_expansions;
    
    // Phase 3 inputs/outputs
    CrateNode* crates;
    uint32_t num_crates;
    Symbol* symbols;
    uint32_t num_symbols;
    ModuleNode* modules;
    uint32_t num_modules;
    
    // Shared structures
    uint32_t* csr_graph;
    uint32_t* hash_table;
    uint8_t* visibility_matrix;
    
    // Pipeline statistics
    uint32_t* phase_timings;
    uint32_t* memory_usage;
    uint32_t* error_counts;
};

// Phase 1 → Phase 3: Extract symbols from AST
__global__ void extract_symbols_from_ast_kernel(
    const ASTNode* ast_nodes,
    uint32_t num_ast_nodes,
    Symbol* symbols,
    uint32_t* symbol_count,
    ModuleNode* modules,
    uint32_t* module_count
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
    
    // Process AST nodes in parallel
    for (uint32_t i = tid; i < num_ast_nodes; i += blockDim.x * gridDim.x) {
        const ASTNode& node = ast_nodes[i];
        
        // Extract symbols based on node type
        switch (node.type) {
            case AST_FUNCTION: {
                uint32_t sym_idx = atomicAdd(symbol_count, 1);
                if (sym_idx < MAX_SYMBOLS) {
                    Symbol& sym = symbols[sym_idx];
                    sym.name_hash = node.name_hash;
                    sym.symbol_type = SYMBOL_FUNCTION;
                    sym.module_id = node.parent_module;
                    sym.visibility = node.visibility;
                    sym.definition_loc = node.source_loc;
                    sym.generic_params = node.generic_count;
                }
                break;
            }
            
            case AST_STRUCT: {
                uint32_t sym_idx = atomicAdd(symbol_count, 1);
                if (sym_idx < MAX_SYMBOLS) {
                    Symbol& sym = symbols[sym_idx];
                    sym.name_hash = node.name_hash;
                    sym.symbol_type = SYMBOL_STRUCT;
                    sym.module_id = node.parent_module;
                    sym.visibility = node.visibility;
                    sym.definition_loc = node.source_loc;
                }
                break;
            }
            
            case AST_MODULE: {
                uint32_t mod_idx = atomicAdd(module_count, 1);
                if (mod_idx < MAX_MODULES) {
                    ModuleNode& mod = modules[mod_idx];
                    mod.module_id = mod_idx;
                    mod.name_hash = node.name_hash;
                    mod.parent_id = node.parent_module;
                    mod.visibility_mask = node.visibility;
                    mod.file_id = node.file_id;
                }
                break;
            }
            
            case AST_USE_STATEMENT: {
                // Extract import information for symbol resolution
                // This would populate import tables
                break;
            }
        }
    }
}

// Phase 2 → Phase 3: Process macro-generated symbols
__global__ void process_macro_symbols_kernel(
    const MacroExpansion* expansions,
    uint32_t num_expansions,
    const Token* expanded_tokens,
    uint32_t num_expanded_tokens,
    Symbol* symbols,
    uint32_t* symbol_count,
    uint32_t* hygiene_table
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process macro expansions
    for (uint32_t i = tid; i < num_expansions; i += blockDim.x * gridDim.x) {
        const MacroExpansion& exp = expansions[i];
        
        // Check if expansion generated new symbols
        uint32_t token_start = exp.output_start;
        uint32_t token_end = exp.output_start + exp.output_count;
        
        for (uint32_t t = token_start; t < token_end && t < num_expanded_tokens; ++t) {
            const Token& tok = expanded_tokens[t];
            
            // Detect symbol declarations in expanded code
            if (tok.type == TOKEN_KEYWORD_FN || 
                tok.type == TOKEN_KEYWORD_STRUCT ||
                tok.type == TOKEN_KEYWORD_ENUM) {
                
                // Next token should be the symbol name
                if (t + 1 < token_end) {
                    const Token& name_tok = expanded_tokens[t + 1];
                    
                    uint32_t sym_idx = atomicAdd(symbol_count, 1);
                    if (sym_idx < MAX_SYMBOLS) {
                        Symbol& sym = symbols[sym_idx];
                        sym.name_hash = name_tok.value;
                        sym.symbol_type = (tok.type == TOKEN_KEYWORD_FN) ? 
                                         SYMBOL_FUNCTION : SYMBOL_STRUCT;
                        
                        // Apply hygiene rules
                        uint32_t hygiene_id = hygiene_table[exp.expansion_id];
                        sym.attributes = ATTR_MACRO_GENERATED | hygiene_id;
                        sym.module_id = exp.module_context;
                    }
                }
            }
        }
    }
}

// Unified pipeline kernel - processes all phases
__global__ void unified_pipeline_kernel(
    PipelineState* state,
    const uint8_t* source_code,
    uint32_t source_length,
    CompilationResult* result
) {
    extern __shared__ char shared_mem[];
    
    // Partition shared memory for different phases
    Token* shared_tokens = reinterpret_cast<Token*>(shared_mem);
    Symbol* shared_symbols = reinterpret_cast<Symbol*>(shared_tokens + 1024);
    uint32_t* shared_counters = reinterpret_cast<uint32_t*>(shared_symbols + 256);
    
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    
    cg::thread_block block = cg::this_thread_block();
    
    // Initialize shared counters
    if (tid == 0) {
        shared_counters[0] = 0; // token count
        shared_counters[1] = 0; // symbol count
        shared_counters[2] = 0; // error count
    }
    block.sync();
    
    // Phase 1: Tokenization (threads 0-63)
    if (tid < 64) {
        uint32_t chunk_size = source_length / 64;
        uint32_t start = tid * chunk_size;
        uint32_t end = min(start + chunk_size, source_length);
        
        for (uint32_t i = start; i < end; ++i) {
            // Simplified tokenization logic
            if (is_token_boundary(source_code[i])) {
                uint32_t tok_idx = atomicAdd(&shared_counters[0], 1);
                if (tok_idx < 1024) {
                    shared_tokens[tok_idx].start = i;
                    shared_tokens[tok_idx].type = classify_token(source_code + i);
                }
            }
        }
    }
    block.sync();
    
    // Phase 2: Symbol extraction (threads 64-127)
    if (tid >= 64 && tid < 128) {
        uint32_t token_id = tid - 64;
        if (token_id < shared_counters[0]) {
            Token& tok = shared_tokens[token_id];
            
            if (is_symbol_declaration(tok.type)) {
                uint32_t sym_idx = atomicAdd(&shared_counters[1], 1);
                if (sym_idx < 256) {
                    shared_symbols[sym_idx].name_hash = tok.value;
                    shared_symbols[sym_idx].symbol_type = get_symbol_type(tok.type);
                }
            }
        }
    }
    block.sync();
    
    // Phase 3: Build symbol table (threads 128-255)
    if (tid >= 128) {
        uint32_t sym_id = tid - 128;
        if (sym_id < shared_counters[1]) {
            Symbol& sym = shared_symbols[sym_id];
            
            // Insert into global hash table
            uint32_t hash = murmur3_32(sym.name_hash);
            uint32_t pos = hash % HASH_TABLE_SIZE;
            
            for (uint32_t i = 0; i < 32; ++i) {
                uint32_t slot = (pos + i) % HASH_TABLE_SIZE;
                uint32_t old = atomicCAS(&state->hash_table[slot * 2], 
                                        UINT32_MAX, sym.name_hash);
                
                if (old == UINT32_MAX) {
                    state->hash_table[slot * 2 + 1] = bid * 256 + sym_id;
                    break;
                }
            }
        }
    }
    block.sync();
    
    // Write results
    if (tid == 0) {
        result->num_tokens = shared_counters[0];
        result->num_symbols = shared_counters[1];
        result->num_errors = shared_counters[2];
        result->success = (shared_counters[2] == 0);
    }
}

// Cross-phase data flow coordinator
__global__ void coordinate_phase_dataflow_kernel(
    PipelineState* state,
    uint32_t current_phase,
    uint32_t* ready_flags,
    uint32_t* completion_flags
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid == 0) {
        // Check if previous phase is complete
        bool prev_complete = true;
        if (current_phase > 0) {
            prev_complete = (completion_flags[current_phase - 1] == 1);
        }
        
        if (prev_complete) {
            // Signal current phase can start
            ready_flags[current_phase] = 1;
            
            // Setup data pointers for current phase
            switch (current_phase) {
                case 1: // Phase 1 → Phase 2
                    state->expanded_tokens = state->tokens;
                    state->num_expanded_tokens = state->num_tokens;
                    break;
                    
                case 2: // Phase 2 → Phase 3
                    // Symbols already extracted during previous phases
                    break;
                    
                case 3: // Phase 3 → Phase 4
                    // Setup for type checking
                    break;
            }
        }
    }
    
    // All threads participate in data migration if needed
    if (ready_flags[current_phase] == 1) {
        // Migrate data between phases
        uint32_t data_size = 0;
        void* src_data = nullptr;
        void* dst_data = nullptr;
        
        switch (current_phase) {
            case 1:
                data_size = state->num_tokens * sizeof(Token);
                src_data = state->tokens;
                dst_data = state->expanded_tokens;
                break;
            case 2:
                data_size = state->num_symbols * sizeof(Symbol);
                src_data = state->symbols;
                dst_data = state->symbols; // In-place processing
                break;
        }
        
        // Parallel memory copy
        if (data_size > 0 && src_data && dst_data && src_data != dst_data) {
            uint32_t bytes_per_thread = (data_size + blockDim.x * gridDim.x - 1) / 
                                       (blockDim.x * gridDim.x);
            uint32_t start = tid * bytes_per_thread;
            uint32_t end = min(start + bytes_per_thread, data_size);
            
            if (start < data_size) {
                memcpy((char*)dst_data + start, 
                      (char*)src_data + start, 
                      end - start);
            }
        }
    }
}

// Performance monitoring kernel
__global__ void monitor_pipeline_performance_kernel(
    PipelineState* state,
    uint32_t* phase_active,
    uint32_t* phase_timings,
    uint32_t* memory_usage,
    uint32_t* throughput_metrics
) {
    const uint32_t tid = threadIdx.x;
    
    __shared__ uint32_t shared_metrics[32];
    
    // Each thread monitors different metrics
    if (tid < 4) {
        // Monitor phase timings
        if (phase_active[tid]) {
            atomicAdd(&phase_timings[tid], 1);
        }
    } else if (tid < 8) {
        // Monitor memory usage
        uint32_t phase = tid - 4;
        uint32_t mem_used = 0;
        
        switch (phase) {
            case 0:
                mem_used = state->num_tokens * sizeof(Token);
                break;
            case 1:
                mem_used = state->num_expanded_tokens * sizeof(Token);
                break;
            case 2:
                mem_used = state->num_symbols * sizeof(Symbol);
                break;
            case 3:
                mem_used = state->num_modules * sizeof(ModuleNode);
                break;
        }
        
        memory_usage[phase] = mem_used;
    } else if (tid < 12) {
        // Calculate throughput
        uint32_t phase = tid - 8;
        uint32_t items_processed = 0;
        uint32_t time_elapsed = phase_timings[phase];
        
        switch (phase) {
            case 0:
                items_processed = state->num_tokens;
                break;
            case 1:
                items_processed = state->num_expansions;
                break;
            case 2:
                items_processed = state->num_symbols;
                break;
            case 3:
                items_processed = state->num_modules;
                break;
        }
        
        if (time_elapsed > 0) {
            throughput_metrics[phase] = items_processed / time_elapsed;
        }
    }
    
    __syncthreads();
    
    // Aggregate metrics
    if (tid == 0) {
        uint32_t total_memory = 0;
        uint32_t total_throughput = 0;
        
        for (int i = 0; i < 4; ++i) {
            total_memory += memory_usage[i];
            total_throughput += throughput_metrics[i];
        }
        
        // Store aggregated metrics
        state->memory_usage[0] = total_memory;
        state->phase_timings[4] = total_throughput; // Overall throughput
    }
}

// Host-side integration functions
extern "C" void launch_phase_integration(
    PipelineState* state,
    uint32_t source_phase,
    uint32_t target_phase
) {
    uint32_t threads = 256;
    uint32_t blocks = 64;
    
    switch (source_phase) {
        case 1: // Phase 1 → Phase 3
            extract_symbols_from_ast_kernel<<<blocks, threads>>>(
                state->ast_nodes, state->num_ast_nodes,
                state->symbols, &state->num_symbols,
                state->modules, &state->num_modules
            );
            break;
            
        case 2: // Phase 2 → Phase 3
            uint32_t* d_hygiene_table;
            cudaMalloc(&d_hygiene_table, MAX_EXPANSIONS * sizeof(uint32_t));
            
            process_macro_symbols_kernel<<<blocks, threads>>>(
                state->expansions, state->num_expansions,
                state->expanded_tokens, state->num_expanded_tokens,
                state->symbols, &state->num_symbols,
                d_hygiene_table
            );
            
            cudaFree(d_hygiene_table);
            break;
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in phase_integration: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void launch_unified_pipeline(
    PipelineState* state,
    const uint8_t* source_code,
    uint32_t source_length,
    CompilationResult* result
) {
    uint32_t threads = 256;
    uint32_t blocks = 32;
    size_t shared_mem = (1024 * sizeof(Token) + 256 * sizeof(Symbol) + 
                        32 * sizeof(uint32_t));
    
    unified_pipeline_kernel<<<blocks, threads, shared_mem>>>(
        state, source_code, source_length, result
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in unified_pipeline: %s\n", cudaGetErrorString(err));
    }
}

} // namespace rustg