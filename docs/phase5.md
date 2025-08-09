# Phase 5: Type Resolution and Borrow Analysis (Hybrid)

## Technical Documentation for rustg Compiler

### Executive Summary

Phase 5 implements parallel type inference, trait resolution, and a simplified borrow checking system using GPU-accelerated constraint solving. This phase leverages SAT-style solvers and symbolic constraint systems to handle Rust's complex type system and ownership rules with massive parallelism.

### Prerequisites

- Phase 4: MIR representation complete
- Type signatures from crate graph (Phase 3)
- Generic instances identified (Phase 4)
- Minimum 6GB GPU memory for constraint systems

### Technical Architecture

#### 5.1 Type System Representation

**Core Type Structures:**

```
struct Type {
    type_id: u32,
    kind: TypeKind,              // Primitive/Struct/Enum/etc
    size: u32,                   // Size in bytes
    align: u16,                  // Alignment requirement
    flags: u16,                  // Copy/Send/Sync/etc
    generic_args: u32,           // Type parameters offset
    constraints: u32,            // Associated constraints
}

enum TypeKind {
    Primitive(PrimitiveType),
    Struct(struct_id),
    Enum(enum_id),
    Reference(region, mutability, inner_type),
    Array(element_type, length),
    Slice(element_type),
    Tuple(elements_offset),
    Function(signature_id),
    Generic(param_id),
    Projection(trait_id, type_id),
    Infer(var_id),
}

struct TypeVariable {
    var_id: u32,
    kind: InferKind,             // Type/Int/Float
    constraints: u32,            // Constraint list offset
    solution: u32,               // Resolved type (0 if unsolved)
    rank: u16,                   // Union-find rank
    parent: u32,                 // Union-find parent
}
```

**Constraint Representation:**

```
struct TypeConstraint {
    constraint_id: u32,
    kind: ConstraintKind,
    lhs: u32,                    // Left operand (type/var)
    rhs: u32,                    // Right operand
    origin: u32,                 // Source location
    priority: u8,                // Resolution priority
}

enum ConstraintKind {
    Equality,                    // T1 = T2
    Subtype,                     // T1 <: T2
    TraitImpl,                   // T: Trait
    Projection,                  // T::Assoc = U
    Lifetime,                    // 'a: 'b
}
```

#### 5.2 GPU Type Unification

**Parallel Union-Find for Type Variables:**

```cuda
__global__ void unify_types(
    TypeVariable* vars,
    TypeConstraint* constraints,
    u32 num_constraints,
    u8* changed_flag
) {
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_constraints) {
        TypeConstraint& c = constraints[tid];

        // Find representatives with path compression
        u32 lhs_root = find_root(vars, c.lhs);
        u32 rhs_root = find_root(vars, c.rhs);

        if (lhs_root != rhs_root) {
            // Union by rank with atomic CAS
            if (atomic_union(vars, lhs_root, rhs_root)) {
                *changed_flag = 1;
            }
        }
    }
}

__device__ u32 find_root(TypeVariable* vars, u32 var) {
    u32 root = var;
    while (vars[root].parent != root) {
        // Path compression
        u32 grandparent = vars[vars[root].parent].parent;
        vars[root].parent = grandparent;
        root = grandparent;
    }
    return root;
}
```

**Constraint Propagation:**

```cuda
__global__ void propagate_constraints(
    TypeConstraint* constraints,
    TypeVariable* vars,
    Type* types,
    u32* worklist,
    u32* worklist_size
) {
    __shared__ TypeConstraint shared_constraints[256];

    // Load constraints to shared memory
    // Process constraint implications in parallel
    // Generate new constraints atomically
    // Update worklist for next iteration
}
```

**Type Inference Algorithm:**

1. **Initialize:** Create type variables for unknowns
2. **Generate:** Extract constraints from MIR
3. **Solve:** Iterate parallel unification
4. **Validate:** Check solution consistency
5. **Substitute:** Replace variables with solutions

#### 5.3 Trait Resolution System

**Trait Implementation Cache:**

```
struct TraitImpl {
    impl_id: u32,
    trait_id: u32,
    implementing_type: u32,
    generic_params: u32,         // Generic parameters
    where_clauses: u32,          // Additional constraints
    methods: u32,                // Method implementations
    cache_key: u64,              // Fast lookup hash
}

struct TraitCache {
    hash_table: u64[],           // Open addressing
    entries: TraitCacheEntry[],
    num_entries: u32,
    capacity: u32,
}
```

**Parallel Trait Resolution:**

```cuda
__global__ void resolve_trait_bounds(
    Type* types,
    TraitImpl* impls,
    TypeConstraint* constraints,
    TraitCache* cache,
    u8* resolution_status
) {
    // Each warp handles one trait bound
    // Search implementation candidates in parallel
    // Check generic parameter compatibility
    // Update cache with results
}
```

**Trait Selection Algorithm:**

1. **Candidate Collection:**

   - Parallel search through impl blocks
   - Filter by trait and self type
   - Check where clause applicability

2. **Ambiguity Detection:**

   - Compare candidates pairwise
   - Detect overlapping implementations
   - Report coherence violations

3. **Confirmation:**
   - Verify selected implementation
   - Instantiate generic parameters
   - Generate method resolutions

**Higher-Ranked Trait Bounds (HRTB):**

```cuda
__global__ void resolve_hrtb(
    TypeConstraint* constraints,
    LifetimeVar* lifetimes,
    u32* skolem_counter
) {
    // Skolemization of lifetime parameters
    // Parallel constraint generation
    // Special handling for for<'a> bounds
}
```

#### 5.4 Simplified Borrow Checker

**Lifetime Representation:**

```
struct Lifetime {
    lifetime_id: u32,
    kind: LifetimeKind,
    scope_start: u32,            // MIR location
    scope_end: u32,
    constraints: u32,            // Outlives relations
}

struct BorrowState {
    place: u32,                  // Memory location
    borrow_kind: BorrowKind,     // Shared/Mutable/Unique
    lifetime: u32,
    active: bool,
    source_location: u32,
}
```

**Parallel Borrow Analysis:**

```cuda
__global__ void analyze_borrows(
    Statement* statements,
    BorrowState* borrows,
    ConflictReport* conflicts,
    u32 num_statements
) {
    __shared__ BorrowState active_borrows[MAX_BORROWS];

    // Each block analyzes one function
    // Threads handle different statements
    // Track active borrows in shared memory
    // Detect conflicts via parallel comparison
}
```

**Dataflow Analysis for Initialization:**

```cuda
__global__ void initialization_analysis(
    BasicBlock* blocks,
    Statement* statements,
    u32* init_state,  // Bitvector per block
    u8* changed
) {
    // Parallel dataflow iteration
    // Each thread handles one block
    // Compute gen/kill sets
    // Propagate to successors
}
```

**Move and Copy Semantics:**

```cuda
__global__ void check_moves(
    Statement* statements,
    Type* types,
    MoveError* errors
) {
    // Check if type implements Copy
    // Track moved values
    // Detect use-after-move
    // Generate error reports
}
```

#### 5.5 SAT-Style Constraint Solver

**Boolean Constraint Encoding:**

```
struct SATClause {
    literals: u32,               // Offset to literal array
    num_literals: u16,
    learned: bool,               // Learned clause flag
    activity: f32,               // Clause activity score
}

struct SATVariable {
    var_id: u32,
    value: i8,                   // -1/0/1 (unassigned/false/true)
    decision_level: u16,
    antecedent: u32,             // Implication clause
    activity: f32,               // Variable activity
}
```

**Parallel CDCL Solver:**

```cuda
__global__ void cdcl_solve(
    SATClause* clauses,
    SATVariable* variables,
    u32* watch_lists,
    u32* conflict_clause
) {
    // Parallel unit propagation
    // Each warp handles watch list updates
    // Detect conflicts cooperatively
    // Learn clauses from conflicts
}
```

**GPU-Optimized BCP (Boolean Constraint Propagation):**

1. Two-watched literals scheme
2. Parallel watch list traversal
3. Warp-synchronous implication
4. Conflict detection via reduction

### Performance Optimizations

#### 6.1 Constraint System Optimizations

**Constraint Deduplication:**

- Hash constraints for uniqueness
- Merge equivalent constraints
- Eliminate redundant checks

**Incremental Solving:**

- Reuse previous solutions
- Update only affected constraints
- Maintain partial solutions

**Constraint Prioritization:**

- Process simpler constraints first
- Defer complex trait bounds
- Early termination on errors

#### 6.2 Memory Optimizations

**Type Interning:**

- Canonical type representation
- Shared type pool
- Fast equality comparison

**Compressed Constraints:**

- Pack constraint data
- Use indices instead of pointers
- Bit-packed flags

#### 6.3 Parallelization Strategies

**Work Distribution:**

- Group related constraints
- Balance load across SMs
- Dynamic work stealing

**Synchronization Reduction:**

- Minimize atomic operations
- Batch updates
- Lock-free data structures

### Error Reporting

**Type Errors:**

```
struct TypeError {
    error_kind: TypeErrorKind,
    expected_type: u32,
    found_type: u32,
    location: SourceLocation,
    notes: u32,                  // Additional notes offset
}

enum TypeErrorKind {
    TypeMismatch,
    TraitNotImplemented,
    LifetimeMismatch,
    BorrowConflict,
    MoveError,
}
```

**Diagnostic Generation:**

- Parallel error collection
- Type difference computation
- Suggestion generation
- Source location mapping

### Testing Strategy

**Unit Tests:**

1. Type unification correctness
2. Trait resolution accuracy
3. Borrow checker soundness
4. Lifetime inference

**Integration Tests:**

1. Standard library type checking
2. Complex trait hierarchies
3. Lifetime polymorphism
4. Associated types

**Stress Tests:**

1. Deep type nesting
2. Many type variables
3. Complex trait bounds
4. Recursive types

### Deliverables

1. **Type Inference Engine:** Complete unification system
2. **Trait Resolver:** Parallel trait selection
3. **Borrow Checker:** Simplified ownership analysis
4. **SAT Solver:** GPU-optimized constraint solver
5. **Error Reporter:** Comprehensive diagnostics
6. **Test Suite:** Type system validation

### Success Criteria

- Type check 95% of crates.io packages
- <100ms for 10K line codebase
- Correct trait resolution for std library
- Basic borrow checking coverage
- Memory usage <1KB per type variable

### Limitations and Future Work

**Current Limitations:**

- Simplified lifetime inference
- Basic trait object support
- Limited const generics
- No full NLL (Non-Lexical Lifetimes)

**Future Enhancements:**

- Full Polonius-style borrow checking
- Complete const evaluation
- Advanced trait solver
- Incremental type checking

### Dependencies and Risks

**Dependencies:**

- MIR from Phase 4
- Type signatures from Phase 3
- Constraint solver primitives

**Risks:**

- Trait resolution complexity
- Lifetime inference accuracy
- Memory usage explosion

**Mitigation:**

- Fallback to simpler algorithms
- Constraint limits
- Incremental solving

### Timeline Estimate

- Week 1-2: Type unification system
- Week 3: Trait resolution basics
- Week 4: Borrow checker implementation
- Week 5: SAT solver integration
- Week 6: Error reporting
- Week 7: Testing and refinement

### Next Phase Preview

Phase 6 will implement the final code generation stage, translating the type-checked and optimized MIR directly to SPIR-V/PTX on the GPU, producing executable code without CPU involvement.
