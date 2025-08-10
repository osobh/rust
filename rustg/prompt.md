# rustg ProjectB: GPU-Native Ecosystem Development

## Project Context

**Project**: rustg ProjectB - Complete GPU-Native Development Ecosystem
**Foundation**: Built upon the revolutionary rustg GPU-native Rust compiler (ProjectA)
**Vision**: Create the world's first complete GPU-native development and runtime ecosystem

### ProjectA Foundation (Completed)
- ✅ World's first GPU-native Rust compiler
- ✅ 15x+ compilation speedup achieved
- ✅ 60+ novel parallel algorithms developed
- ✅ Production-ready with 100% Rust compatibility
- ✅ Delivered 18x faster than planned timeline

### ProjectB Mission
Build a comprehensive GPU-native ecosystem extending the rustg compiler with:
- Developer toolchain and debugging infrastructure
- GPU-native runtime primitives and memory management
- Standard library implementations optimized for GPU
- Direct GPU storage and networking capabilities
- Complete distributed GPU operating system

## Development Methodology

### STRICT Test-Driven Development (TDD)

**MANDATORY REQUIREMENTS:**
1. **Tests FIRST**: Write comprehensive tests BEFORE any implementation
2. **NO STUBS/MOCKS**: All tests must use real GPU operations and actual CUDA code
3. **Real Validation**: Tests must validate actual GPU computation results
4. **Performance Tests**: Include performance benchmarks in test suites
5. **Integration Tests**: Test actual GPU kernel integration, not simulations

### Code Quality Standards

**FILE SIZE LIMITS:**
- Maximum 850 lines per source file
- Split large components into logical modules
- Maintain clear separation of concerns

**GPU CODE STANDARDS:**
- Structure-of-Arrays (SoA) for all data structures
- Coalesced memory access patterns mandatory
- Warp-level cooperation for all parallel algorithms
- Zero CPU intervention in critical paths
- Memory pool allocation to prevent fragmentation

### Performance Requirements

**EVERY COMPONENT MUST ACHIEVE:**
- 10x+ performance improvement over CPU equivalents
- 90%+ GPU utilization for compute-bound operations
- Linear scaling with GPU core count
- Sub-microsecond latency for critical operations
- Memory bandwidth utilization >80%

## Current Phase: ProjectB Phase 1 - Developer Experience

### Phase 1 Objectives
1. **cargo-g**: GPU-aware Cargo subcommand
   - GPU code detection and separation
   - Multi-target compilation
   - Artifact caching system
   
2. **Debug Infrastructure**: 
   - Source mapping for GPU code
   - Timeline tracing
   - Warp-level debugging
   
3. **Testing Framework**:
   - GPU-native test execution
   - Golden output validation
   - Performance regression detection

### Implementation Priority
1. Start with cargo-g build management
2. Implement GPU test harness
3. Add debugging capabilities
4. Integrate with rustg compiler

## Memory Bank References

### Primary Context Files (Load These)
- `docs/memory-bank/activeContext.md` - Current development status
- `docs/memory-bank/progress.md` - Phase completion tracking
- `docs/memory-bank/projectB/phase1.md` - Current phase specification
- `docs/memory-bank/projectB-overview.md` - Complete ProjectB roadmap

### Technical References
- `docs/memory-bank/architecture/gpu-memory-architecture.md` - GPU patterns
- `docs/memory-bank/architecture/parallel-algorithms.md` - Algorithm catalog
- `docs/memory-bank/architecture/performance-targets.md` - Benchmarks

### Historical Context
- `docs/PROJECTA_COMPLETION_SUMMARY.md` - ProjectA achievements
- `docs/memory-bank/project-completion.md` - Detailed ProjectA analysis

## Architecture Principles (From ProjectA Success)

### GPU-First Design
1. **Massive Parallelism**: Every algorithm must utilize 1000+ threads
2. **Memory Coalescing**: Aligned access patterns mandatory
3. **Warp Cooperation**: 32-thread collaborative algorithms
4. **Kernel Fusion**: Minimize memory transfers between kernels
5. **Shared Memory**: Utilize for local optimization

### Proven Patterns
- **Ring Buffers**: For GPU-CPU communication
- **Memory Pools**: Pre-allocated GPU memory management
- **Lock-Free Structures**: Atomic operations for synchronization
- **Batch Processing**: Amortize kernel launch overhead
- **Pipeline Parallelism**: Overlap compute and memory operations

## Test-Driven Development Process

### Test Structure
```cuda
// ALWAYS write test FIRST
__global__ void test_feature_kernel(TestResult* results) {
    // Real GPU computation
    // Actual memory operations
    // True parallel execution
    // No mocking allowed
}

// Then implement feature
__global__ void feature_kernel(...) {
    // Implementation after test passes
}
```

### Test Requirements
1. **Unit Tests**: Per-kernel validation with real GPU execution
2. **Integration Tests**: Multi-kernel pipeline testing
3. **Performance Tests**: Benchmark against targets
4. **Stress Tests**: Maximum load validation
5. **Regression Tests**: Prevent performance degradation

## Current Development Focus

### Immediate Tasks (Phase 1)
1. Implement cargo-g subcommand structure
2. Create GPU kernel detection system
3. Build artifact caching layer
4. Develop test harness framework
5. Integrate with rustg compiler

### Success Criteria
- cargo-g successfully compiles GPU code
- Test framework executes on actual GPU
- Debug information properly mapped
- 10x faster than CPU toolchain
- Zero CPU involvement in compilation

## Development Guidelines

### Commit Standards
- Atomic commits with single responsibility
- Comprehensive test coverage per commit
- Performance benchmarks included
- No breaking changes without migration path

### Documentation Requirements
- Inline CUDA kernel documentation
- Algorithm complexity analysis
- Memory usage documentation
- Performance characteristics
- Integration examples

## Risk Mitigation

### Technical Risks
- GPU memory limitations → Use memory pools
- Kernel complexity → Split into multiple kernels
- Debugging difficulties → Comprehensive logging
- Performance regressions → Continuous benchmarking

### Process Risks
- Scope creep → Strict phase boundaries
- Technical debt → Mandatory refactoring cycles
- Integration issues → Early integration testing
- Performance targets → Regular validation

## Quick Start

1. Load this prompt.md and referenced memory-bank files
2. Review ProjectB Phase 1 specification
3. Start with cargo-g test implementation (TDD)
4. Write GPU kernel tests FIRST
5. Implement features to pass tests
6. Validate 10x performance improvement
7. Document and commit

## Important Reminders

⚠️ **NO STUBS OR MOCKS** - All tests must be real GPU code
⚠️ **850 LINE LIMIT** - Split large files immediately
⚠️ **TESTS FIRST** - Never write implementation before tests
⚠️ **10x PERFORMANCE** - Every component must meet this target
⚠️ **GPU-FIRST** - Zero CPU involvement in critical paths

---

*Building on the revolutionary rustg compiler to create the world's first complete GPU-native development ecosystem*