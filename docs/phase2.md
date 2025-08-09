# Phase 2: GPU-Based Macro Expansion

## Technical Documentation for rustg Compiler

### Executive Summary

Phase 2 implements a GPU-native macro expansion system for rustg, focusing on declarative macro patterns that can be efficiently processed in parallel. This phase transforms macro invocations into expanded token trees while maintaining hygiene and scoping rules, all within GPU memory without CPU intervention.

### Prerequisites

- Phase 1 complete: GPU parser producing valid AST/token trees
- Token buffer infrastructure operational
- AST DAG resident in GPU memory
- Pattern matching primitives available in GPU kernels

### Technical Architecture

#### 2.1 Declarative Macro Subset Design

**Supported Macro Features:**

```rust
// Supported patterns
- Literal matches: $ident, $expr, $ty, $pat, $stmt
- Repetition operators: $(...)*  $(...)+
- Optional groups: $(...)?
- Token tree captures: $tt
- Simple conditionals: cfg-like attributes
```

**Restricted Features (GPU-incompatible):**

- No procedural macros (requires host compilation)
- No recursive macro expansion beyond fixed depth
- No dynamic code generation
- Limited stringify! and concat! operations

**Macro Representation in GPU Memory:**

```
struct MacroDefinition {
    macro_id: u32,
    pattern_offset: u32,      // Offset into pattern buffer
    pattern_count: u16,       // Number of pattern rules
    body_offset: u32,         // Offset into expansion template
    hygiene_context: u32,     // Hygiene scope identifier
    attributes: u32,          // Flags and metadata
}

struct MacroPattern {
    matcher_tokens: u32,      // Offset to matcher token sequence
    matcher_length: u16,      // Number of tokens in matcher
    bindings_mask: u64,       // Bitmask of binding positions
    repetition_info: u32,     // Packed repetition metadata
    guard_condition: u32,     // Optional guard clause
}
```

#### 2.2 Parallel Pattern Matching Implementation

**Matching Algorithm:**

**Phase 1: Coarse-Grain Matching**

```
Kernel: macro_pattern_match_coarse
- Each block handles one macro invocation
- Warps test different pattern rules in parallel
- Use ballot operations for early termination
```

**Phase 2: Fine-Grain Token Matching**

```
Kernel: macro_pattern_match_fine
- Each thread matches one token position
- Warp-wide reductions for sequence validation
- Shared memory for binding capture
```

**Pattern Matching State Machine:**

```
enum MatchState {
    Start,
    Matching(position, depth),
    Binding(var_id, start, end),
    Repetition(count, max),
    Success(bindings),
    Failure(reason)
}
```

**Warp-Level Pattern Execution:**

1. Leader thread coordinates pattern traversal
2. Worker threads validate token matches in parallel
3. Binding captures stored in shared memory
4. Repetition counting via atomic operations

**Optimized Matching Strategies:**

- **Fast Path:** Direct token comparison for literals
- **Slow Path:** Complex pattern matching with backtracking
- **Cache:** Recently matched patterns in shared memory

#### 2.3 GPU Buffer Rewriting for Expansion

**Token Buffer Management:**

```
struct ExpansionBuffer {
    tokens: Token[],          // Expanded token stream
    positions: u32[],         // Source positions for hygiene
    contexts: u32[],          // Hygiene contexts
    capacity: u32,            // Buffer capacity
    write_head: u32,          // Atomic write position
}
```

**Parallel Expansion Process:**

**Step 1: Space Calculation**

```
Kernel: calculate_expansion_size
- Each thread calculates size for one binding
- Reduction to find total expansion size
- Allocate output buffer atomically
```

**Step 2: Parallel Token Generation**

```
Kernel: generate_expanded_tokens
- Each warp handles one template fragment
- Substitute bindings in parallel
- Write to output buffer with atomic offsets
```

**Step 3: Token Stream Merging**

```
Kernel: merge_token_streams
- Compact expanded tokens
- Update AST references
- Maintain source mapping
```

**Memory Allocation Strategy:**

- Pre-allocate expansion pools (power-of-2 sizes)
- Use ring buffer for temporary expansions
- Garbage collect unused expansions per batch

#### 2.4 Hygiene and Context Tracking

**Hygiene Implementation:**

```
struct HygieneContext {
    context_id: u32,
    parent_context: u32,
    introduction_site: u32,   // Where macro was defined
    call_site: u32,           // Where macro was invoked
    transparency: u8,         // Opaque/Transparent/Semitransparent
}
```

**Parallel Hygiene Management:**

1. **Context Creation:** Atomic allocation of new contexts
2. **Scope Resolution:** Parallel lookup in context DAG
3. **Name Resolution:** Warp-wide name disambiguation
4. **Context Propagation:** Update expanded tokens with contexts

**Hygiene-Aware Name Resolution:**

```
Kernel: resolve_hygienic_names
- Each thread handles one identifier
- Traverse context chain in parallel
- Cache resolution results in shared memory
```

### Performance Optimizations

#### 3.1 Memory Layout Optimizations

**Token Stream Compression:**

- Pack tokens into 64-bit values where possible
- Use dictionary encoding for common tokens
- Compress repetitive patterns

**Caching Strategy:**

- LRU cache for frequently used macros
- Memoize expansion results for identical invocations
- Share common sub-patterns across macros

#### 3.2 Execution Optimizations

**Kernel Fusion:**

- Combine pattern matching + expansion when possible
- Merge multiple small expansions into single kernel
- Pipeline expansion stages for large macros

**Warp Specialization:**

- Dedicate warps to specific macro types
- Use persistent threads for hot macros
- Dynamic load balancing across SMs

### Error Handling

**Macro Expansion Errors:**

```
struct MacroError {
    error_type: u32,          // Error category
    macro_id: u32,            // Failing macro
    token_position: u32,      // Error location
    pattern_index: u16,       // Which pattern failed
    details: u64,             // Error-specific data
}
```

**Error Recovery:**

- Mark failed expansions in AST
- Preserve original tokens for diagnostics
- Continue expansion of independent macros
- Generate placeholder tokens for type checking

### Testing Strategy

**Unit Tests:**

1. Pattern matching correctness for all Rust macro patterns
2. Hygiene preservation across expansion
3. Performance tests for common macro patterns
4. Memory usage validation

**Integration Tests:**

1. Standard library macro expansion (vec!, format!, etc.)
2. Complex derive macro patterns
3. Nested macro expansion
4. Cross-crate macro usage

**Stress Tests:**

1. Deeply nested expansions (up to limit)
2. Large repetition counts
3. Maximum binding count scenarios
4. Concurrent expansion of many macros

### Deliverables

1. **Pattern Matching Kernels:** Complete matcher implementation
2. **Expansion Kernels:** Token generation and substitution
3. **Hygiene System:** Context tracking and resolution
4. **Macro Registry:** GPU-resident macro database
5. **Test Suite:** Comprehensive macro expansion tests
6. **Performance Report:** Benchmarks vs rustc macro expansion

### Success Criteria

- Support 90% of declarative macros in standard library
- Achieve >50x speedup for common macro patterns
- Maintain 100% hygiene correctness
- Memory overhead <2x of expanded token size
- Support minimum 16-level expansion depth

### Limitations and Future Work

**Current Limitations:**

- No procedural macro support
- Fixed maximum expansion depth
- Limited compile-time computation
- No custom derive macros

**Future Enhancements:**

- WASM bridge for proc macros (Phase 7+)
- Dynamic expansion depth
- Incremental re-expansion
- Macro debugging support

### Dependencies and Risks

**Dependencies:**

- Phase 1 AST and token infrastructure
- Sufficient GPU memory for expansion buffers
- Atomic operation support

**Risks:**

- Expansion state explosion
- Complex hygiene edge cases
- Memory fragmentation from many small expansions

**Mitigation:**

- Implement expansion limits
- Comprehensive hygiene test suite
- Memory pool compaction

### Timeline Estimate

- Week 1: Basic pattern matching kernel
- Week 2: Token substitution system
- Week 3: Repetition and optional groups
- Week 4: Hygiene context implementation
- Week 5: Buffer management and optimization
- Week 6: Testing and benchmarking

### Next Phase Preview

Phase 3 will leverage the expanded token trees to build a GPU-native crate graph, enabling parallel dependency resolution and module system implementation entirely on the GPU.
