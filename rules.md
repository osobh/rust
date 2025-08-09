# rustg GPU Compiler Safety Rules and Constraints

## Critical GPU Safety Rules

1. **NEVER access host memory from device code** - All data must be in GPU memory
2. **NEVER exceed shared memory limits** - 48KB per block maximum
3. **NEVER ignore memory coalescing** - Uncoalesced access kills performance
4. **NEVER skip cuda-memcheck validation** - Race conditions are catastrophic
5. **NEVER deploy kernels without performance validation** - Must meet speedup targets

## Project-Specific Rules

### Memory Bank Integrity

- **ALWAYS read ALL memory bank files** before starting GPU kernel work
- **ALWAYS update memory bank files** after performance milestones
- **NEVER delete or rename memory bank files**
- **MAINTAIN consistency** between memory bank documentation and implementation
- **TRACK performance metrics** in progress.md

### GPU Code Quality Standards

- **ALWAYS write CPU reference implementation first** - baseline for correctness
- **ALWAYS validate kernel output** against CPU reference (>95% accuracy)
- **ALWAYS profile memory access patterns** - must be coalesced
- **ALWAYS check for race conditions** with cuda-memcheck
- **ALWAYS measure performance** against baseline (>100x target)

### GPU Memory Management

- **ENFORCE memory hierarchy** - Shared > Constant > Texture > Global
- **USE Structure-of-Arrays (SoA)** for coalesced access patterns
- **MINIMIZE global memory traffic** - use shared memory for reuse
- **ALIGN memory allocations** - 256-byte boundaries for performance
- **POOL memory allocations** - avoid frequent malloc/free

### Kernel Architecture Adherence

- **FOLLOW warp-level cooperation** - 32 threads work together
- **MAINTAIN high occupancy** - >85% SM utilization target
- **MINIMIZE warp divergence** - avoid thread-dependent branches
- **USE appropriate block sizes** - multiples of 32 (typically 256)
- **IMPLEMENT proper synchronization** - __syncthreads() where needed

### Test-Driven Development (TDD) - MANDATORY

- **RED PHASE**: Write failing tests FIRST - no exceptions
  - Unit tests for each function/kernel
  - Integration tests for module interactions
  - E2E tests for complete workflows
  - All tests must fail initially
- **GREEN PHASE**: Write MINIMAL code to make tests pass
  - No extra features or optimizations
  - Focus only on passing tests
  - Simplest implementation that works
- **REFACTOR PHASE**: Improve code while tests stay green
  - Optimize performance
  - Clean up code structure
  - All tests must remain passing
- **NO CODE WITHOUT TESTS** - Tests must exist before implementation
- **NO SKIPPING PHASES** - Must complete red-green-refactor cycle

### Development Workflow

- **FOLLOW TDD strictly** - Red → Green → Refactor for every feature
- **CREATE kernels in phases** - Correctness → Optimization → Validation
- **WRITE comprehensive kernel tests** before implementation
- **PROFILE every kernel** with nvprof or nsight
- **DOCUMENT performance characteristics** for each kernel
- **VALIDATE against CPU reference** for correctness

### GPU Resource Constraints

- **LIMIT kernel complexity** - 100 lines maximum per kernel
- **FOCUS on memory bandwidth** - >80% utilization target
- **OPTIMIZE for specific GPU** - Tune for target architecture
- **BATCH operations** - Minimize kernel launches
- **FUSE kernels when possible** - Reduce memory traffic

### File Length Restrictions - STRICT ENFORCEMENT

- **MAXIMUM 850 lines per file** - No exceptions for hand-written code
- **CUDA kernels**: 100 lines maximum per kernel function
- **Rust files**: 850 lines maximum per file
- **C++ files**: 850 lines maximum per file
- **When approaching 700 lines**: Start planning refactoring
- **Refactoring strategies**:
  - Split into submodules
  - Extract helper functions
  - Separate concerns into different files
  - Create trait implementations in separate files
  - Move tests to separate test files
- **Exceptions**: Only for generated code (protobuf, bindgen, macros)
- **ENFORCEMENT**: CI/CD must fail if limits exceeded

### Communication Standards

- **USE clear kernel names** - operation_kernel suffix
- **DOCUMENT thread/block configuration** requirements
- **SPECIFY memory requirements** (shared, constant, registers)
- **EXPLAIN parallelization strategy** in comments

## Working Directory Rules

- **WORK ONLY within the rustg project directory**
- **CREATE kernels in appropriate directories** (src/lexer/kernels/, etc.)
- **USE CMake build system** for CUDA compilation
- **MAINTAIN separation** between host and device code

## Build Rules

- **USE CMake for builds** - not cargo for CUDA code
- **COMPILE with appropriate architecture** flags (-arch=sm_75, etc.)
- **ENABLE all CUDA warnings** during development
- **BUILD in both Debug and Release** modes for testing

## GPU Testing Rules - STRICT PERFORMANCE GATES

### Mandatory Test Coverage - ALL THREE TYPES REQUIRED

#### UNIT TESTS (Required for EVERY function/kernel)
- **Minimum 90% code coverage** for GPU kernels
- **Minimum 85% code coverage** for Rust code
- Test all edge cases and error conditions
- Test both success and failure paths
- CPU reference implementation for all GPU kernels
- Mock external dependencies appropriately
- Test individual functions in isolation
- Validate all return values and error codes

#### INTEGRATION TESTS (Required for ALL module interactions)
- Test data flow between modules
- Test GPU-CPU communication and memory transfers
- Test kernel pipeline sequences
- Verify memory management across boundaries
- Test error propagation between components
- Validate module initialization and cleanup
- Test concurrent access patterns
- Verify resource sharing and synchronization

#### END-TO-END TESTS (Required before ANY feature is complete)
- Complete compilation workflow tests
- Real Rust source file compilation
- Performance validation (10x speedup minimum)
- Memory usage validation (<15x source size)
- Output correctness vs rustc
- Large file handling (>100KB sources)
- Error recovery and graceful degradation
- Stress testing with complex code patterns

### Additional Test Requirements

- **CORRECTNESS TESTS**: Required for EVERY kernel
  - CPU reference implementation required
  - >95% accuracy vs CPU baseline
  - Edge case validation
  - Boundary condition handling

- **MEMORY TESTS**: Required for ALL kernels
  - cuda-memcheck --tool memcheck (no errors)
  - cuda-memcheck --tool racecheck (no races)
  - cuda-memcheck --tool synccheck (proper synchronization)
  - Verify coalesced access patterns

- **PERFORMANCE TESTS**: Required before deployment
  - Baseline CPU implementation timing
  - GPU kernel must exceed speedup target
  - Memory bandwidth utilization >80%
  - SM occupancy >85%

### GPU Testing Workflow - TDD MANDATORY

1. **RED PHASE - Write Tests First**:
   - Write unit tests for kernel functionality
   - Write integration tests for GPU pipeline
   - Write E2E tests for compilation workflow
   - Write performance benchmarks with targets
   - ALL tests must fail initially

2. **GREEN PHASE - Minimal Implementation**:
   - Implement CPU reference first
   - Implement basic GPU kernel
   - Make unit tests pass
   - Make integration tests pass
   - Make E2E tests pass
   - No optimizations yet

3. **REFACTOR PHASE - Optimize**:
   - Optimize memory access patterns
   - Implement shared memory usage
   - Reduce warp divergence
   - Improve algorithm efficiency
   - All tests must stay green

4. **PERFORMANCE PHASE - Validate**:
   - Run performance benchmarks
   - Verify >100x speedup for parsing
   - Verify memory bandwidth >80%
   - Profile with nvprof/nsight
   - Document performance characteristics

5. **Coverage Enforcement**:
   - All code paths tested
   - Minimum 90% coverage for kernels
   - No untested error paths

### Test Execution Commands

- `make test` - Run all kernel tests
- `./build/kernel_tests` - Kernel correctness tests
- `./build/benchmarks` - Performance validation
- `cuda-memcheck ./build/kernel_tests` - Memory safety
- `nvprof ./build/benchmarks` - Performance profiling

### Test Organization Requirements

- **Unit tests**: `tests/unit/` or in-file `#[cfg(test)]` modules for Rust
- **Integration tests**: `tests/integration/` directory
- **E2E tests**: `tests/e2e/` directory
- **Performance tests**: `tests/benchmarks/` directory
- **Test fixtures**: `tests/fixtures/` for test data
- **GPU kernel tests**: Must include CPU reference implementation
- **Test naming**: Descriptive names explaining what is tested
- **Test independence**: Each test must be runnable in isolation
- **Test determinism**: No random or time-dependent behavior
- **Test speed**: Unit tests <100ms, integration <1s, E2E <10s

### BLOCKING CRITERIA - ZERO TOLERANCE

- **NO CODE** without tests written first (TDD Red phase required)
- **NO FILES** exceeding 850 lines (refactor immediately)
- **NO MERGE** without:
  - Unit tests passing (>90% coverage for kernels, >85% for Rust)
  - Integration tests passing (all module interactions covered)
  - E2E tests passing (complete workflows validated)
  - Performance targets met (>100x for parsing)
  - TDD cycle complete (Red-Green-Refactor)
  - cuda-memcheck clean (no errors, races, or sync issues)
- **NO REFACTORING** without complete green test suite
- **NO OPTIMIZATION** without correctness tests passing
- **NO FEATURE** marked complete without E2E validation
- **NO DEPLOYMENT** without:
  - Full test suite passing
  - Performance benchmarks validated
  - Memory usage within limits
  - Documentation complete
- **NO SKIPPING** any phase of TDD
- **NO EXCEPTIONS** to these rules

## Error Handling

- **USE CUDA error checking** after every API call
- **IMPLEMENT kernel error flags** for device-side errors
- **LOG performance metrics** for analysis
- **PROVIDE meaningful error context**

## Performance Considerations

- **OPTIMIZE memory access first** - bandwidth is critical
- **MINIMIZE warp divergence** - keep threads synchronized
- **USE shared memory** for data reuse
- **COALESCE global memory access** - critical for performance
- **PROFILE before optimizing** - measure, don't guess

## Phase 1 Specific Rules

### Current Focus: Parallel Parsing (Week 2 of 8)

- **PRIORITIZE token boundary resolution** - Critical bottleneck
- **IMPLEMENT warp cooperation** - 32 threads per token region
- **VALIDATE parsing accuracy** - Must match rustc output
- **ACHIEVE 50x speedup milestone** - This week's target
- **OPTIMIZE memory patterns** - SoA for token buffers

### Performance Requirements

- Tokenization: 1 GB/s throughput minimum
- Parsing: >100x speedup vs single-threaded rustc
- Memory usage: <15x source file size for AST
- GPU utilization: >90% SM occupancy

## Remember

- This is a GPU-native compiler - performance is everything
- Every kernel must contribute to 10x overall speedup
- Correctness without performance is failure
- Memory patterns determine GPU performance
- Test, measure, optimize - in that order