# Phase 1: GPU-Based Parsing and Tokenization

## Technical Documentation for rustg Compiler

### Executive Summary

Phase 1 establishes the foundational parsing infrastructure for rustg, implementing fully parallel lexical analysis and syntactic parsing directly on the GPU. This phase transforms raw Rust source code into a GPU-resident Abstract Syntax Tree (AST) using massively parallel processing techniques, eliminating traditional sequential parsing bottlenecks.

### Prerequisites

Before beginning Phase 1, ensure completion of:

- Phase 0: Foundational Architecture (complete)
- GPU-pinned memory infrastructure operational
- Runtime orchestrator capable of launching GPU kernels
- Source files loaded into unified/shared memory

### Technical Architecture

#### 1.1 Parallel Lexer Design

The parallel lexer operates as a GPU kernel with the following characteristics:

**Core Architecture:**

- Each CUDA thread/Metal thread group processes a fixed-size character span (typically 32-64 bytes)
- Threads collaborate via warp-level primitives for token boundary detection
- Output tokens written to a compacted flat buffer using atomic operations

**Implementation Details:**

```
Thread Assignment Model:
- Block size: 256 threads (8 warps)
- Each thread processes: source[thread_id * SPAN_SIZE : (thread_id + 1) * SPAN_SIZE]
- Overlapping boundary regions handled via warp voting
```

**Token Recognition Strategy:**

- Implement finite state machines (FSMs) per thread for local token recognition
- Use lookup tables in constant memory for character classification
- Handle multi-character tokens via warp-synchronous state sharing
- Resolve token boundaries using ballot operations across warps

**Memory Layout:**

```
Token Buffer Structure (SoA):
- token_types: u32[]     // Token type IDs
- token_starts: u32[]    // Byte offset in source
- token_lengths: u16[]   // Token byte length
- token_metadata: u64[]  // Line/column info, flags
```

**Challenges and Solutions:**

- **Challenge:** Variable-length tokens crossing thread boundaries
- **Solution:** Implement overlap zones with collaborative resolution via warp shuffle operations

- **Challenge:** String literals and comments spanning multiple blocks
- **Solution:** Two-pass approach - first pass marks boundaries, second pass processes content

#### 1.2 GPU-Based Parser Implementation

**Parser Architecture Choice:**

Implement a hybrid Pratt parser with recursive-descent fallback, adapted for SIMD execution:

**Warp-Wide Stack Management:**

- Each warp maintains a shared operator precedence stack
- Stack operations synchronized via \_\_syncwarp()
- Shared memory used for intra-warp communication

**Parsing Strategy:**

```
Primary Expression Parsing:
1. Each warp handles one statement/expression
2. Leader thread coordinates precedence climbing
3. Worker threads validate sub-expressions in parallel
4. Results merged into AST nodes via reduction
```

**AST Node Representation:**

```
GPU-Friendly AST Node (32 bytes aligned):
struct ASTNode {
    node_type: u32,      // Node type enum
    parent_idx: u32,     // Parent node index
    child_start: u32,    // First child index
    child_count: u16,    // Number of children
    token_idx: u16,      // Associated token
    metadata: u64,       // Type hints, attributes
    semantic_data: u64   // Reserved for later phases
}
```

**Parallel Parsing Phases:**

1. **Statement Boundary Detection:** Identify top-level items in parallel
2. **Expression Parsing:** Parse expressions within statements using Pratt algorithm
3. **Structure Assembly:** Build nested structures (structs, impls, traits)
4. **AST Validation:** Parallel verification of structural correctness

#### 1.3 AST Storage and DAG Construction

**Memory Layout Design:**

Implement Structure-of-Arrays (SoA) for cache efficiency:

```
AST Storage Layout:
- nodes: ASTNode[]           // All AST nodes
- node_children: u32[]       // Child indices (flattened)
- string_pool: char[]        // Deduplicated string data
- string_offsets: u32[]      // String pool offsets
- source_maps: SourceLoc[]   // Source location mapping
```

**DAG Construction Process:**

1. Convert tree structure to directed acyclic graph
2. Identify and merge common sub-expressions
3. Build index structures for fast traversal
4. Create type annotation placeholders

**Memory Management:**

- Use GPU memory pools with power-of-2 allocation
- Implement compaction passes to reduce fragmentation
- Maintain free lists in device memory

### Performance Considerations

**Optimization Strategies:**

1. **Coalesced Memory Access:** Ensure sequential threads access sequential memory
2. **Warp Divergence Minimization:** Group similar parsing tasks within warps
3. **Shared Memory Utilization:** Cache frequently accessed lookup tables
4. **Occupancy Optimization:** Balance register usage vs thread count

**Expected Performance Metrics:**

- Target: 1GB/s source code throughput
- Latency: <10ms for 100K LOC codebase
- Memory usage: ~10x source size for full AST

### Error Handling

**GPU-Side Error Detection:**

- Maintain error buffer in global memory
- Use atomic operations to append error records
- Include source location and error type

**Error Recovery Strategy:**

- Continue parsing after errors when possible
- Mark invalid AST nodes with error flags
- Preserve partial AST for IDE scenarios

### Testing Strategy

**Unit Tests:**

1. Lexer correctness on all Rust token types
2. Parser validation against rust-analyzer test suite
3. AST structural integrity verification
4. Performance regression tests

**Integration Tests:**

1. Parse entire rust standard library
2. Validate against reference AST from rustc
3. Stress test with malformed input
4. Memory leak detection

### Deliverables

1. **GPU Lexer Kernel:** Complete parallel tokenization implementation
2. **GPU Parser Kernel:** Pratt parser with full Rust syntax support
3. **AST Memory Manager:** Efficient SoA storage with DAG support
4. **Test Suite:** Comprehensive correctness and performance tests
5. **Documentation:** Kernel launch parameters and tuning guide

### Success Criteria

- Successfully parse 95% of crates.io packages
- Achieve >100x speedup vs single-threaded rustc parser
- Memory usage within 15x of source size
- Zero CPU intervention during parsing phase
- Error messages maintain source location accuracy

### Dependencies and Risks

**Dependencies:**

- CUDA 11.0+ / Metal 3.0+
- Unified memory support
- Phase 0 infrastructure

**Risks:**

- Complex Rust syntax edge cases
- Memory bandwidth limitations
- Warp divergence in complex expressions

**Mitigation Strategies:**

- Implement CPU fallback for edge cases
- Use compression for AST storage
- Profile and optimize hot paths

### Timeline Estimate

- Weeks 1-2: Parallel lexer implementation
- Weeks 3-4: Basic parser framework
- Weeks 5-6: Full Rust syntax support
- Week 7: AST DAG conversion
- Week 8: Testing and optimization

### Next Phase Preview

Phase 2 will build upon the AST infrastructure to implement GPU-based macro expansion, requiring careful design of pattern matching kernels and hygiene tracking mechanisms.
