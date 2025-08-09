# GPU Memory Architecture: Memory Management Strategies

## Memory Hierarchy and Strategy

The rustg compiler's performance depends critically on efficient utilization of the GPU memory hierarchy. This document outlines the memory management strategies, data layouts, and optimization techniques used throughout the compilation pipeline.

## GPU Memory Types and Usage

### Global Memory (Device DRAM)
**Characteristics**: Large capacity (8-80GB), high bandwidth (1-3TB/s), high latency (200-800 cycles)

**Primary Usage in rustg**:
- **Source Code Storage**: Original Rust source files loaded via unified memory
- **AST Storage**: Structure-of-Arrays layout for abstract syntax trees  
- **Symbol Tables**: Hash tables for name resolution and type information
- **MIR Storage**: Intermediate representation in optimized layout
- **Output Buffers**: Generated code and binary data

**Optimization Strategies**:
```cuda
// Coalesced access pattern - good
for (int i = threadIdx.x; i < n; i += blockDim.x) {
    data[i] = process(input[i]);  // Adjacent threads access adjacent elements
}

// Strided access pattern - avoid
for (int i = threadIdx.x; i < n; i += blockDim.x) {
    data[i * stride] = process(input[i]);  // Poor memory coalescing
}
```

### Shared Memory (On-Chip SRAM) 
**Characteristics**: Small capacity (48-128KB per SM), ultra-fast access (1-2 cycles), limited by banking

**Primary Usage in rustg**:
- **Token Buffers**: Temporary storage during parallel tokenization
- **Parser Stacks**: Precedence stacks for warp-wide expression parsing
- **Reduction Operations**: Intermediate results for parallel reductions
- **Caching**: Frequently accessed lookup tables and intermediate data

**Banking Optimization**:
```cuda
// Avoid bank conflicts through padding
__shared__ int shared_data[32 + 1][32];  // +1 padding avoids power-of-2 stride

// Use for warp-level cooperation
__shared__ Token token_buffer[256];  // One token per thread in block
__syncthreads();  // Synchronize before access
```

### Constant Memory (Cached Global Memory)
**Characteristics**: Small capacity (64KB), read-only, broadcast capability, cached

**Primary Usage in rustg**:
- **Lookup Tables**: Character classification tables, operator precedence
- **Grammar Rules**: Parsing rules and syntax patterns  
- **Type Information**: Basic type sizes and alignment requirements
- **Configuration**: Compiler flags and target-specific parameters

**Optimal Usage Pattern**:
```cuda
__constant__ char char_types[256];  // All threads access same element efficiently
__constant__ u32 precedence_table[MAX_OPERATORS];

__device__ inline CharType get_char_type(char c) {
    return char_types[c];  // Broadcast to all threads in warp
}
```

### Texture Memory (Cached Global Memory)
**Characteristics**: Read-only, 2D spatial locality caching, filtered access

**Primary Usage in rustg**:
- **Source Code Access**: Spatial locality for parsing adjacent characters
- **AST Traversal**: Cache-friendly access to tree structures
- **Symbol Lookup**: Spatial locality in hash table access
- **Reference Data**: Type signatures and metadata

**Texture Memory Configuration**:
```cuda
texture<char, cudaTextureType1D, cudaReadModeElementType> source_texture;

__device__ char get_source_char(int index) {
    return tex1Dfetch(source_texture, index);  // Hardware caching
}
```

### Registers (Thread-Private Storage)
**Characteristics**: Highest performance (1 cycle), limited quantity (64KB per SM), affects occupancy

**Optimization Strategy**:
- Minimize register usage to maximize occupancy
- Use shared memory for spilling when necessary
- Careful variable scoping to reduce register pressure

## Memory Layout Patterns

### Structure-of-Arrays (SoA) vs Array-of-Structures (AoS)

**AoS Layout (CPU-friendly, GPU-poor)**:
```cpp
struct Token {
    TokenType type;
    u32 start_pos;
    u32 length;
    u64 metadata;
};
Token tokens[N];  // Poor coalescing - threads access scattered data
```

**SoA Layout (GPU-optimized)**:
```cpp
struct TokenBuffer {
    TokenType* types;      // All types together
    u32* start_positions;  // All positions together  
    u32* lengths;          // All lengths together
    u64* metadata;         // All metadata together
};
// Excellent coalescing - threads access adjacent elements
```

**Performance Impact**: SoA provides 5-10x memory bandwidth improvement for parallel access patterns.

### AST Memory Layout

**Flattened Tree Structure**:
```cpp
struct ASTNode {
    u32 node_type;      // Node kind (function, struct, etc.)
    u32 parent_index;   // Parent node (0 for root)  
    u32 first_child;    // Index of first child
    u16 child_count;    // Number of children
    u16 token_index;    // Associated token
    u64 type_info;      // Type annotations
    u64 metadata;       // Additional flags and data
};

// Stored as flat array for GPU efficiency
ASTNode nodes[MAX_NODES];
```

**Benefits**:
- Eliminates pointer chasing
- Enables coalesced memory access
- Supports parallel tree traversal
- Cache-friendly layout

### Symbol Table Layout

**Hash Table with Open Addressing**:
```cpp
struct Symbol {
    u64 name_hash;      // FNV hash of symbol name
    u32 type_id;        // Reference to type information
    u32 module_id;      // Defining module
    u16 visibility;     // Public/private/etc.
    u16 kind;          // Function/struct/trait/etc.
};

struct SymbolTable {
    Symbol* symbols;    // Flat array of symbols
    u32 capacity;       // Power-of-2 for efficient modulo
    u32 count;         // Current symbol count
    u32* probe_counts; // Linear probing statistics
};
```

## Memory Pool Management

### Pool Allocation Strategy

**Power-of-2 Block Sizes**:
```cpp
enum PoolSize {
    POOL_64B   = 0,   // Small objects (tokens, simple nodes)
    POOL_256B  = 1,   // Medium objects (AST nodes)  
    POOL_1KB   = 2,   // Large objects (symbol tables)
    POOL_4KB   = 3,   // Very large objects (function bodies)
    NUM_POOLS  = 4
};

struct MemoryPool {
    void* base_address;
    u32 block_size;
    u32 num_blocks;
    u32* free_bitmap;   // Atomic allocation bitmap
};
```

**Atomic Allocation**:
```cuda
__device__ void* allocate_from_pool(MemoryPool* pool) {
    // Find free block using atomic operations
    u32 word_index = 0;
    u32 bit_index = 0;
    
    // Scan for free block
    for (u32 i = 0; i < pool->num_blocks / 32; i++) {
        u32 word = pool->free_bitmap[i];
        if (word != 0xFFFFFFFF) {  // Has free blocks
            bit_index = __ffs(~word) - 1;  // Find first free bit
            u32 old = atomicOr(&pool->free_bitmap[i], 1U << bit_index);
            if (!(old & (1U << bit_index))) {  // Successfully allocated
                word_index = i;
                break;
            }
        }
    }
    
    u32 block_index = word_index * 32 + bit_index;
    return (char*)pool->base_address + block_index * pool->block_size;
}
```

### Memory Compaction

**Parallel Compaction Algorithm**:
1. **Mark Phase**: Identify live objects in parallel
2. **Compute Phase**: Calculate new positions using parallel prefix scan  
3. **Move Phase**: Relocate objects in parallel
4. **Update Phase**: Fix pointer references

```cuda
__global__ void compact_memory_pool(
    MemoryPool* pool,
    u32* live_mask,      // Bitmap of live objects
    u32* new_positions   // Computed new positions
) {
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < pool->num_blocks && live_mask[tid]) {
        u32 old_pos = tid * pool->block_size;
        u32 new_pos = new_positions[tid] * pool->block_size;
        
        // Parallel memory copy
        memcpy((char*)pool->base_address + new_pos,
               (char*)pool->base_address + old_pos, 
               pool->block_size);
    }
}
```

## Garbage Collection Strategy

### Generational Collection

**Generation Classification**:
- **Gen 0**: Short-lived objects (temporary parsing results)
- **Gen 1**: Medium-lived objects (AST nodes, tokens)
- **Gen 2**: Long-lived objects (symbol tables, type information)

**Collection Frequency**:
- Gen 0: Every compilation phase
- Gen 1: Every few phases  
- Gen 2: Only when memory pressure high

### Parallel Mark-and-Sweep

**Mark Phase**:
```cuda
__global__ void mark_live_objects(
    void** roots,           // Root object pointers
    u32 num_roots,
    u8* mark_bits          // Mark bitmap
) {
    __shared__ void* work_queue[BLOCK_SIZE];
    __shared__ u32 queue_head, queue_tail;
    
    // Initialize with roots
    if (threadIdx.x == 0) {
        queue_head = 0;
        queue_tail = min(num_roots, BLOCK_SIZE);
        for (u32 i = 0; i < queue_tail; i++) {
            work_queue[i] = roots[i];
        }
    }
    __syncthreads();
    
    // Parallel marking with work queue
    while (queue_head < queue_tail) {
        void* obj = work_queue[threadIdx.x + queue_head];
        if (obj && !is_marked(mark_bits, obj)) {
            mark_object(mark_bits, obj);
            enqueue_children(obj, work_queue, &queue_tail);
        }
        __syncthreads();
        
        if (threadIdx.x == 0) {
            queue_head += blockDim.x;
        }
        __syncthreads();
    }
}
```

**Sweep Phase**:
```cuda
__global__ void sweep_unmarked_objects(
    MemoryPool* pool,
    u8* mark_bits,
    u32* free_bitmap
) {
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < pool->num_blocks) {
        if (!is_marked(mark_bits, tid)) {
            // Free the block atomically
            u32 word = tid / 32;
            u32 bit = tid % 32;
            atomicAnd(&free_bitmap[word], ~(1U << bit));
        }
    }
}
```

## Memory Access Optimization Techniques

### Coalescing Strategies

**Perfect Coalescing Pattern**:
```cuda
// All threads in warp access consecutive 32-bit words
__global__ void coalesced_access(u32* data, u32 n) {
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        data[tid] = process(data[tid]);  // Perfect coalescing
    }
}
```

**Avoiding Uncoalesced Access**:
```cuda
// Bad: Strided access pattern
for (int i = threadIdx.x; i < n; i += 32) {
    data[i * stride] = process(data[i]);  // Poor coalescing
}

// Good: Block-strided with consecutive access
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    data[i] = process(data[i]);  // Good coalescing
}
```

### Cache Optimization

**Temporal Locality**:
```cuda
// Cache frequently accessed data in shared memory
__shared__ char lookup_table[256];

if (threadIdx.x < 256) {
    lookup_table[threadIdx.x] = global_lookup_table[threadIdx.x];
}
__syncthreads();

// Now use cached data
char result = lookup_table[input[tid]];
```

**Spatial Locality**:
```cuda
// Process data in cache-friendly order
for (int block = 0; block < n; block += CACHE_BLOCK_SIZE) {
    for (int i = 0; i < CACHE_BLOCK_SIZE; i++) {
        process(data[block + i]);  // Good spatial locality
    }
}
```

## Memory Usage Budgeting

### Per-Phase Memory Allocation

**Phase 1 (Parsing)**:
- Source code: 1x (input size)
- Token buffer: 3x (tokens + metadata)  
- AST storage: 8x (structured representation)
- Total: ~12x source size

**Phase 2 (Macro Expansion)**:
- Expanded tokens: +5x (macro expansion overhead)
- Pattern buffers: +2x (temporary storage)
- Total: ~19x source size

**Phase 3 (Crate Graph)**:
- Symbol tables: +3x (name resolution data)
- Dependency graphs: +2x (relationship data)
- Total: ~24x source size

**Target Optimization**: Compress to <10x through phases 4-7

### Memory Pressure Handling

**Early Warning System**:
```cuda
__device__ bool check_memory_pressure(MemoryManager* mgr) {
    float usage = (float)mgr->allocated / mgr->total_capacity;
    return usage > 0.85f;  // 85% threshold
}
```

**Pressure Relief Strategies**:
1. **Aggressive GC**: Force garbage collection
2. **Compression**: Apply data compression techniques
3. **Streaming**: Process data in chunks
4. **CPU Fallback**: Move some data to host memory

This memory architecture provides the foundation for rustg's high-performance GPU-native compilation, ensuring efficient utilization of GPU memory hierarchy while maintaining scalability and correctness.