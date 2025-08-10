# Phase 3: GPU-Based Crate Graph Resolution

## Technical Documentation for rustg Compiler

### Executive Summary

Phase 3 constructs and analyzes a comprehensive dependency graph entirely on the GPU, managing crate relationships, module hierarchies, and symbol resolution. This phase transforms the expanded AST into a navigable knowledge graph that serves as the foundation for type checking and code generation.

### Prerequisites

- Phase 1: AST representation in GPU memory
- Phase 2: Macro-expanded token trees available
- GPU graph processing primitives
- Sufficient memory for graph representation (~100MB per 10K crates)

### Technical Architecture

#### 3.1 GPU-Native Knowledge Graph Design

**Graph Data Structure:**

```
struct CrateNode {
    crate_id: u32,
    name_hash: u64,              // FNV hash of crate name
    root_module: u32,            // Index to module table
    dependencies: u32,           // Offset into edge list
    dep_count: u16,
    features: u64,               // Feature flags bitmap
    edition: u8,                 // Rust edition (2015/2018/2021)
    metadata_offset: u32,        // Additional metadata location
}

struct ModuleNode {
    module_id: u32,
    parent_module: u32,          // Parent in module tree
    crate_id: u32,               // Owning crate
    visibility: u8,              // pub/pub(crate)/private
    children_offset: u32,        // Child modules offset
    items_offset: u32,           // Items defined in module
    imports_offset: u32,         // Use statements offset
    attributes: u64,             // Module attributes
}

struct DependencyEdge {
    from_crate: u32,
    to_crate: u32,
    edge_type: u8,               // Regular/Dev/Build dependency
    version_req: u32,            // Semver requirement (simplified)
    features: u32,               // Required features offset
    rename: u32,                 // Optional rename offset
}
```

**Memory Layout (CSR Format):**

```
Graph Storage:
- node_array: CrateNode[]        // All crate nodes
- edge_array: DependencyEdge[]   // All edges (sorted by source)
- edge_offsets: u32[]            // CSR row pointers
- module_array: ModuleNode[]     // All modules across crates
- string_pool: char[]            // Names and paths
- symbol_table: Symbol[]         // All exported symbols
```

**Graph Statistics Tracking:**

```
struct GraphMetrics {
    total_crates: u32,
    total_edges: u32,
    max_depth: u16,              // Longest dependency chain
    total_symbols: u32,
    cycle_count: u16,            // Detected circular dependencies
    memory_usage: u64,
}
```

#### 3.2 Parallel Graph Traversal Algorithms

**Dependency Resolution Kernel:**

```
Kernel: resolve_dependencies
- Block dimension: 256 threads (1 block per crate)
- Shared memory: 48KB for local graph cache
```

**Algorithm: Parallel BFS for Dependency Tree:**

1. Each block processes one root crate
2. Threads collaborate on frontier expansion
3. Use atomic operations for visited marking
4. Detect cycles via parallel color propagation

```
__global__ void traverse_dependencies(
    CrateNode* nodes,
    DependencyEdge* edges,
    u32* edge_offsets,
    u32* visit_order,
    u32* cycle_flags
) {
    __shared__ u32 frontier[MAX_FRONTIER];
    __shared__ u32 next_frontier[MAX_FRONTIER];
    __shared__ u32 visited[MAX_CRATES/32];  // Bitmap

    // Warp-synchronous BFS
    while (frontier_not_empty) {
        parallel_expand_frontier();
        detect_cycles();
        update_visit_order();
    }
}
```

**Parallel Topological Sort:**

```
Kernel: topological_sort
- Implements Kahn's algorithm in parallel
- Each thread manages in-degree for subset of nodes
- Atomic decrements for processed dependencies
- Output: Linear ordering for compilation
```

**Cycle Detection:**

```
Kernel: detect_dependency_cycles
- Three-color DFS implemented via warp voting
- Each warp explores different path
- Shared memory for cycle recording
- Output: Cycle paths for error reporting
```

#### 3.3 Name Resolution and Symbol Table

**Symbol Representation:**

```
struct Symbol {
    symbol_id: u32,
    name_hash: u64,
    defining_module: u32,
    visibility: u8,
    symbol_type: u8,             // Function/Struct/Trait/etc
    generic_params: u16,         // Number of generic parameters
    attributes: u32,             // Attribute flags
    resolution_cache: u32,       // Fast lookup cache
}

struct ImportResolution {
    import_id: u32,
    source_path: u64,           // Hashed import path
    target_symbol: u32,         // Resolved symbol ID
    visibility: u8,
    status: u8,                 // Resolved/Pending/Error
}
```

**Parallel Name Resolution:**

```
Kernel: resolve_imports
- Each warp handles one module's imports
- Parallel path traversal through module tree
- Hash table lookups in shared memory
- Ambiguity detection via reduction
```

**Resolution Algorithm:**

1. **Phase 1: Build Export Maps**

   - Parallel scan of all modules
   - Create hash maps of exported symbols
   - Handle glob imports specially

2. **Phase 2: Resolve Direct Imports**

   - Each thread handles one import
   - Traverse module paths in parallel
   - Update resolution cache atomically

3. **Phase 3: Handle Glob Imports**
   - Expand glob patterns
   - Detect ambiguities
   - Priority resolution for conflicts

**Optimization: Resolution Cache:**

```
struct ResolutionCache {
    hash_table: u64[],          // Open addressing hash table
    entries: CacheEntry[],      // Path -> Symbol mappings
    stats: CacheStats,          // Hit/miss statistics
}
```

#### 3.4 Module System Implementation

**Module Tree Operations:**

```
Kernel: build_module_tree
- Construct hierarchical module structure
- Each thread processes module declarations
- Parent-child relationships via atomic linking
- Visibility inheritance calculation
```

**Module Path Resolution:**

```
Kernel: resolve_module_paths
- Convert relative paths to absolute
- Handle `super`, `self`, `crate` keywords
- Parallel string manipulation
- Cache canonical paths
```

**Visibility Checking:**

```
Kernel: check_visibility
- Parallel visibility rule validation
- Each thread checks one access
- Traverse module hierarchy for pub(crate)
- Generate visibility errors
```

### Performance Optimizations

#### 4.1 Graph Compression

**Edge List Compression:**

- Delta encoding for sequential crate IDs
- Bit packing for version requirements
- Feature flag deduplication

**String Interning:**

- Global string pool with hash index
- Parallel string deduplication
- Compressed pointers (32-bit offsets)

#### 4.2 Traversal Optimizations

**Work Distribution:**

- Dynamic load balancing for irregular graphs
- Persistent threads for hot paths
- Warp specialization by graph region

**Memory Access Patterns:**

- Coalesced reads for edge lists
- Texture memory for frequently accessed nodes
- Shared memory caching for BFS frontier

**Algorithmic Optimizations:**

- Early termination for resolved symbols
- Incremental resolution for partial graphs
- Parallel pruning of unreachable nodes

### Error Handling

**Dependency Errors:**

```
struct DependencyError {
    error_type: enum {
        CycleDetected,
        VersionConflict,
        MissingDependency,
        FeatureNotFound,
    },
    involved_crates: u32[],     // Crate IDs involved
    error_path: u32[],          // Path through graph
    details: char[],            // Human-readable message
}
```

**Resolution Errors:**

- Unresolved imports
- Ambiguous names
- Visibility violations
- Missing modules

**Error Recovery:**

- Continue resolution with placeholders
- Mark affected symbols as errored
- Preserve partial resolution for IDE support

### Graph Persistence and Caching

**Serialization Format:**

```
struct GraphSnapshot {
    version: u32,
    timestamp: u64,
    node_count: u32,
    edge_count: u32,
    compressed_nodes: byte[],   // LZ4 compressed
    compressed_edges: byte[],
    symbol_index: byte[],
}
```

**Incremental Updates:**

- Detect changed crates via hash comparison
- Rebuild only affected subgraphs
- Merge updates into existing graph
- Maintain consistency via versioning

### Testing Strategy

**Unit Tests:**

1. Graph construction from various crate structures
2. Cycle detection accuracy
3. Symbol resolution correctness
4. Module visibility rules

**Integration Tests:**

1. Full crates.io registry graph (100K+ crates)
2. Complex workspace resolution
3. Feature flag combinations
4. Cross-edition compatibility

**Performance Tests:**

1. Graph traversal throughput
2. Memory usage scaling
3. Cache hit rates
4. Incremental update speed

### Deliverables

1. **Graph Construction Kernels:** Complete graph building system
2. **Traversal Kernels:** BFS, DFS, topological sort implementations
3. **Resolution System:** Name and symbol resolution
4. **Module System:** Full module tree implementation
5. **Caching Layer:** Persistent graph cache
6. **Visualization Tools:** Graph debugging utilities

### Success Criteria

- Build graph for entire crates.io in <5 seconds
- Resolve 99.9% of symbols correctly
- Detect all circular dependencies
- Memory usage <1GB for 100K crates
- Support incremental updates in <100ms

### Dependencies and Risks

**Dependencies:**

- Phase 1-2 AST and expansion complete
- Sufficient GPU memory (minimum 4GB)
- Hash table primitives available

**Risks:**

- Graph size explosion with many crates
- Complex versioning requirements
- Ambiguous resolution rules

**Mitigation:**

- Implement graph pruning strategies
- Simplify version resolution model
- Comprehensive test coverage

### Timeline Estimate

- Week 1: Basic graph structure implementation
- Week 2: Traversal algorithm implementation
- Week 3: Symbol resolution system
- Week 4: Module system completion
- Week 5: Error handling and recovery
- Week 6: Optimization and testing

### Next Phase Preview

Phase 4 will utilize the crate graph to implement a GPU-native MIR (Mid-level Intermediate Representation) transformation pipeline, enabling parallel optimization passes and monomorphization directly on the GPU.
