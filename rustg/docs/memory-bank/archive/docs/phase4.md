# Phase 4: GPU-MIR Pass Pipeline

## Technical Documentation for rustg Compiler

### Executive Summary

Phase 4 implements a complete intermediate representation (IR) transformation pipeline on the GPU, translating the AST to a MIR-like format and executing optimization passes entirely in parallel. This phase establishes the core compilation pipeline with SSA form, optimization transforms, and monomorphization capabilities.

### Prerequisites

- Phase 1-3: AST, macro expansion, and crate graph complete
- GPU memory pools for IR storage
- Basic type information available from Phase 3
- Minimum 8GB GPU memory for large programs

### Technical Architecture

#### 4.1 Custom MIR Design for GPU

**Core IR Data Structures:**

```
struct MIRFunction {
    func_id: u32,
    signature: u32,              // Type signature offset
    num_locals: u16,
    num_blocks: u16,
    blocks_offset: u32,          // Offset to basic blocks
    locals_offset: u32,          // Local variable declarations
    generic_params: u32,         // Generic parameter info
    attributes: u64,             // Function attributes/flags
}

struct BasicBlock {
    block_id: u32,
    function: u32,               // Parent function
    statements_offset: u32,      // Statement array offset
    num_statements: u16,
    terminator: Terminator,      // Block terminator
    predecessors: u32,           // Predecessor list offset
    successors: u32,             // Successor list offset
    dom_tree_info: u32,          // Dominance information
}

struct Statement {
    kind: u8,                    // Assign/StorageLive/StorageDead/etc
    source_info: u32,            // Source location
    operands: [u32; 3],          // Operand indices (place/rvalue)
    type_info: u32,              // Type of operation
}

struct Terminator {
    kind: u8,                    // Goto/Switch/Call/Return
    targets: u32,                // Target blocks offset
    num_targets: u8,
    call_info: u32,              // For Call terminators
}
```

**SSA Representation:**

```
struct SSAValue {
    value_id: u32,
    def_block: u32,              // Defining block
    def_statement: u32,          // Defining statement
    type_id: u32,                // Value type
    num_uses: u16,
    uses_offset: u32,            // Use sites array
    version: u16,                // SSA version number
}

struct PhiNode {
    result: u32,                 // SSA value produced
    block: u32,                  // Containing block
    num_inputs: u16,
    inputs: u32,                 // (block, value) pairs offset
}
```

**Memory Layout (SoA for GPU efficiency):**

```
MIR Memory Pools:
- functions: MIRFunction[]       // All functions
- blocks: BasicBlock[]           // All basic blocks
- statements: Statement[]        // All statements
- ssa_values: SSAValue[]         // SSA value definitions
- phi_nodes: PhiNode[]           // Phi nodes
- type_pool: Type[]              // Type information
- const_pool: Constant[]         // Constant values
```

#### 4.2 AST to MIR Translation

**Translation Kernel Design:**

```
Kernel: ast_to_mir_translation
- Block size: 256 threads
- Each block handles one function
- Warps collaborate on basic block construction
```

**Translation Process:**

1. **Function Discovery:**

```cuda
__global__ void discover_functions(
    ASTNode* ast,
    MIRFunction* functions,
    u32* func_count
) {
    // Parallel scan for function definitions
    // Each thread examines AST node range
    // Atomic allocation of function slots
}
```

2. **Basic Block Construction:**

```cuda
__global__ void build_basic_blocks(
    ASTNode* ast,
    MIRFunction* func,
    BasicBlock* blocks
) {
    // Identify control flow boundaries
    // Create blocks at branch points
    // Link predecessors/successors
}
```

3. **Statement Generation:**

```cuda
__global__ void generate_statements(
    ASTNode* ast,
    BasicBlock* block,
    Statement* statements
) {
    // Convert AST expressions to MIR
    // Flatten nested expressions
    // Generate temporary variables
}
```

**Pattern Matching for Lowering:**

- Match AST patterns to MIR templates
- Use lookup tables for common patterns
- Parallel pattern matching across expressions

#### 4.3 GPU Optimization Passes

**Pass Infrastructure:**

```
struct OptimizationPass {
    pass_id: u32,
    pass_type: enum {
        Analysis,
        Transform,
        Cleanup
    },
    kernel_ptr: void*,           // Kernel function pointer
    dependencies: u32,           // Required passes bitmap
    invalidates: u32,            // Invalidated analyses bitmap
}
```

**4.3.1 Constant Folding:**

```cuda
__global__ void constant_folding_pass(
    Statement* statements,
    Constant* const_pool,
    u32 num_statements
) {
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_statements) {
        Statement& stmt = statements[tid];
        if (is_arithmetic_op(stmt.kind)) {
            // Check if operands are constants
            // Compute result at compile time
            // Replace with constant value
        }
    }
}
```

**4.3.2 Inlining:**

```cuda
__global__ void inline_functions(
    MIRFunction* functions,
    CallSite* call_sites,
    InlineMetrics* metrics
) {
    // Calculate inline cost in parallel
    // Each warp handles one call site
    // Clone and substitute function body
    // Update SSA form
}
```

**Inline Decision Heuristics:**

- Function size threshold
- Call frequency analysis
- Recursive depth limit
- Generic instantiation count

**4.3.3 Dead Code Elimination:**

```cuda
__global__ void mark_and_sweep_dce(
    SSAValue* values,
    Statement* statements,
    u32* live_mask
) {
    // Phase 1: Mark reachable values
    // Start from function returns and side effects
    // Propagate liveness backwards

    // Phase 2: Sweep dead statements
    // Parallel compaction of live statements
}
```

**4.3.4 Control Flow Simplification:**

```cuda
__global__ void simplify_cfg(
    BasicBlock* blocks,
    Terminator* terminators
) {
    // Remove empty blocks
    // Merge linear block sequences
    // Eliminate unreachable blocks
    // Simplify conditional branches
}
```

**4.3.5 Common Subexpression Elimination:**

```cuda
__global__ void cse_pass(
    Statement* statements,
    u64* expr_hashes,
    u32* value_map
) {
    // Hash expressions in parallel
    // Build expression -> value map
    // Replace redundant computations
}
```

#### 4.4 Monomorphization

**Generic Instantiation:**

```
struct GenericInstance {
    generic_func: u32,           // Original generic function
    type_args: u32,              // Type argument array
    instance_id: u32,            // Unique instance ID
    mir_func: u32,               // Generated MIR function
}
```

**Parallel Monomorphization:**

```cuda
__global__ void monomorphize_generics(
    MIRFunction* generic_funcs,
    CallSite* call_sites,
    TypeArg* type_args,
    MIRFunction* output_funcs
) {
    // Each block handles one instantiation
    // Clone generic function
    // Substitute type parameters
    // Generate specialized code
}
```

**Type Substitution:**

- Parallel type parameter replacement
- Update all type references
- Adjust memory layouts
- Recompute size/alignment

**Deduplication:**

- Hash (function, type_args) pairs
- Detect duplicate instantiations
- Share monomorphized instances

#### 4.5 IR Versioning and Rollback

**Versioned Memory Pools:**

```
struct IRVersion {
    version_id: u32,
    timestamp: u64,
    functions_snapshot: u32,    // Memory pool offset
    blocks_snapshot: u32,
    statements_snapshot: u32,
    metadata: VersionMetadata,
}
```

**Checkpoint Management:**

- Create snapshots before major passes
- Copy-on-write for memory efficiency
- Parallel snapshot creation
- Fast rollback capability

### Performance Optimizations

#### 5.1 Memory Management

**Pool Allocation:**

- Pre-allocated pools for each IR type
- Power-of-2 sizing for fast allocation
- Parallel compaction to reduce fragmentation

**Memory Reuse:**

- Recycle deallocated blocks
- Generation-based garbage collection
- Incremental compaction

#### 5.2 Pass Scheduling

**Dependency Analysis:**

- Build pass dependency graph
- Identify independent passes
- Schedule parallel execution

**Pass Fusion:**

- Combine compatible passes
- Reduce memory traffic
- Share intermediate results

#### 5.3 GPU Utilization

**Kernel Configuration:**

- Dynamic block size based on workload
- Occupancy optimization
- Shared memory utilization

**Load Balancing:**

- Work stealing for irregular functions
- Dynamic parallelism for nested structures
- Warp specialization

### Testing Strategy

**Correctness Tests:**

1. AST to MIR translation validation
2. SSA form verification
3. Optimization correctness
4. Monomorphization accuracy

**Performance Tests:**

1. Pass execution time
2. Memory usage patterns
3. Optimization effectiveness
4. Scaling with program size

**Regression Tests:**

1. Standard library compilation
2. Known optimization patterns
3. Edge cases and corner cases
4. Performance benchmarks

### Deliverables

1. **MIR Translation System:** Complete AST->MIR converter
2. **Optimization Passes:** Full suite of GPU optimizations
3. **Monomorphization Engine:** Generic instantiation system
4. **Pass Manager:** Orchestration and scheduling
5. **Verification Tools:** IR validation utilities
6. **Performance Analysis:** Profiling and metrics

### Success Criteria

- Translate 100% of Rust constructs to MIR
- Achieve >20% code size reduction via optimization
- Monomorphize all generic functions correctly
- Process 1M LOC in <1 second
- Memory usage <100 bytes per MIR statement

### Dependencies and Risks

**Dependencies:**

- Complete AST from Phase 1
- Type information from Phase 3
- Sufficient GPU memory

**Risks:**

- Complex optimization interactions
- Memory explosion from monomorphization
- SSA construction complexity

**Mitigation:**

- Conservative optimization defaults
- Monomorphization limits
- Incremental SSA updates

### Timeline Estimate

- Week 1-2: MIR data structure and translation
- Week 3: Basic optimization passes
- Week 4: Advanced optimizations
- Week 5: Monomorphization system
- Week 6: Pass manager and scheduling
- Week 7: Testing and tuning

### Next Phase Preview

Phase 5 will implement type resolution and a simplified borrow checking system, leveraging GPU parallelism for constraint solving and lifetime analysis.
