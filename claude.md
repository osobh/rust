# Claude Code Instructions for rustg GPU Compiler

This project implements a GPU-native Rust compiler achieving 10x compilation speedup through massive parallelization.

## IMPORTANT: Read Memory Bank First

Before starting any work, ALWAYS read ALL files in the `rustg/docs/memory-bank/` directory to understand the project context and current phase.

## Memory Bank Structure

- `rustg/docs/memory-bank/projectbrief.md` - rustg vision: GPU-native Rust compiler
- `rustg/docs/memory-bank/productContext.md` - Why GPU compilation matters (10x speedup goal)
- `rustg/docs/memory-bank/systemPatterns.md` - GPU architecture patterns (SoA, warp cooperation)
- `rustg/docs/memory-bank/techContext.md` - CUDA technology stack and requirements
- `rustg/docs/memory-bank/activeContext.md` - Current Phase 1 parsing focus
- `rustg/docs/memory-bank/progress.md` - Phase implementation status (25% of Phase 1)
- `rustg/docs/memory-bank/architecture/` - GPU memory hierarchy and parallel algorithms
- `rustg/docs/memory-bank/phases/` - Detailed 7-phase implementation roadmap

## Project: rustg - GPU-Native Rust Compiler

rustg is a revolutionary Rust compiler that performs the entire compilation pipeline on GPU, from parsing to code generation. Built with CUDA, it leverages thousands of GPU cores to achieve >10x compilation speedup compared to traditional CPU-based rustc. The compiler implements parallel parsing algorithms, GPU-optimized data structures (Structure-of-Arrays), and warp-level cooperation for complex compilation tasks.

## GPU Compiler Workflow

1. Start each session by reading all Memory Bank files
2. Check current phase status in `progress.md` (Currently Phase 1, Week 2 of 8)
3. For EVERY GPU kernel implementation:
   - Write CPU reference implementation FIRST (baseline for correctness)
   - Write CUDA kernel tests (validate against CPU reference)
   - Implement minimal kernel functionality
   - Optimize for performance (coalesced access, shared memory)
   - Verify performance targets (>100x for parsing)
4. Run GPU quality gates before proceeding:
   - Kernel correctness tests (match CPU reference)
   - Memory access pattern verification (>80% bandwidth utilization)
   - Race condition detection (cuda-memcheck)
   - Performance benchmarks (must meet phase targets)
5. Update Memory Bank files with results
6. Keep `activeContext.md` current with implementation progress
7. Document GPU patterns in `systemPatterns.md`
8. Track performance metrics in `progress.md`

## Update Triggers

Update Memory Bank files when:

- Implementing new GPU kernels (after performance validation)
- Achieving performance milestones (e.g., 50x, 75x, 100x speedup)
- Discovering new parallel algorithms or patterns
- Completing phase milestones
- Resolving technical challenges (token boundaries, memory optimization)
- Before ending work sessions

## Key Project Commands

### Build Commands
- Initialize: `mkdir build && cd build && cmake ..`
- Build: `make -j$(nproc)`
- Clean: `make clean`

### Test Commands
- All Tests: `make test`
- Unit Tests: `./build/unit_tests`
- Kernel Tests: `./build/kernel_tests`
- Integration: `./build/integration_tests`
- Benchmarks: `./build/benchmarks`

### GPU Validation
- Memory Check: `cuda-memcheck --tool memcheck ./build/kernel_tests`
- Race Detection: `cuda-memcheck --tool racecheck ./build/kernel_tests`
- Profile: `nvprof ./build/benchmarks` or `nsight-compute`
- Bandwidth: `./build/benchmarks --bandwidth`

### Development
- Format: `make format` (clang-format for CUDA)
- Static Analysis: `make lint`
- Coverage: `make coverage`

## Current Implementation Focus

### Phase 1: GPU-Based Parsing and Tokenization (Week 2 of 8)

**Current Status**: 25% Complete

**Completed**:
- ✅ Thread assignment model (256 threads, 32-64 byte spans)
- ✅ Token buffer SoA memory layout design
- ✅ Character classification lookup tables
- ✅ Basic finite state machine framework
- ✅ Warp-level cooperation primitives

**In Progress**:
- 🔄 Parallel lexer implementation (70% complete)
- 🔄 Token boundary resolution algorithms (40% complete)
- 🔄 String literal and comment handling (30% complete)
- 🔄 AST node structure design (20% complete)

**This Week's Priorities**:
1. Complete token boundary resolution with warp voting
2. Implement string literal parallel parsing
3. Begin AST construction kernels
4. Achieve 50x speedup milestone

### Performance Targets (Phase 1)
- Tokenization: 1 GB/s throughput (currently 450 MB/s)
- Parsing: >100x speedup vs single-threaded
- Memory: <15x source file size for AST
- GPU Utilization: >90% SM occupancy

## Architecture Overview

**rustg GPU Compilation Layers:**

1. **GPU Kernels**: CUDA kernels for all compilation phases
   - Parallel lexing with warp cooperation
   - Token boundary resolution across threads
   - AST construction in GPU memory
   - Pattern matching for macro expansion

2. **Memory Architecture**: GPU-optimized data structures
   - Structure-of-Arrays (SoA) for coalesced access
   - Shared memory for warp cooperation
   - Constant memory for lookup tables
   - Texture memory for source code access

3. **Performance Optimization**: Meeting 10x speedup goal
   - Memory bandwidth optimization (>80% utilization)
   - Warp divergence minimization
   - Kernel fusion for reduced memory traffic
   - Dynamic parallelism for complex parsing

**Key Design Principles:**
- GPU-first: All compilation on GPU, minimal CPU involvement
- Massive parallelism: Thousands of threads working simultaneously
- Memory efficiency: Optimized access patterns for GPU architecture
- Performance critical: Every kernel must meet speedup targets
- Correctness verified: CPU reference implementation for validation

## Important Notes

- Phase 1 is foundational - parsing performance determines overall success
- Token boundary resolution is the critical bottleneck to solve
- Memory access patterns make or break GPU performance
- Warp cooperation essential for complex parsing tasks
- Performance validation required before moving to next phase
- Current target: Complete Phase 1 by Week 8 with 100x parsing speedup

## Testing Enforcement - GPU CRITICAL

**NEVER mark a GPU kernel complete without:**

1. CPU reference implementation with identical results
2. Kernel correctness tests (>95% accuracy vs CPU)
3. Memory access pattern verification (coalesced access)
4. Race condition testing (cuda-memcheck clean)
5. Performance benchmarks meeting targets (>100x for parsing)
6. Memory bandwidth utilization >80%
7. GPU occupancy metrics >85%

**GPU testing is critical - incorrect parallel algorithms can produce subtle bugs that are extremely difficult to debug.**

## Implementation References

- Architecture docs in `rustg/docs/memory-bank/architecture/`
- Phase details in `rustg/docs/memory-bank/phases/`
- GPU patterns in `rustg/docs/memory-bank/systemPatterns.md`
- Performance targets in `rustg/docs/memory-bank/architecture/performance-targets.md`
- Current focus in `rustg/docs/memory-bank/activeContext.md`

Remember: We're building a compiler that runs entirely on GPU to achieve 10x speedup. Every line of code should contribute to massive parallelization and performance optimization.