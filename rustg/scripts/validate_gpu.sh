#!/bin/bash
# GPU validation script for rustg compiler

set -e

echo "=== rustg GPU Validation Suite ==="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Error: CUDA not found. Please install CUDA toolkit.${NC}"
    exit 1
fi

if ! command -v cuda-memcheck &> /dev/null; then
    echo -e "${RED}Error: cuda-memcheck not found. Please install CUDA toolkit.${NC}"
    exit 1
fi

echo "CUDA installation found:"
nvcc --version | head -n 1
echo ""

# Build the project
echo "Building rustg with CUDA support..."
make clean
make build PROFILE=debug
echo -e "${GREEN}Build complete!${NC}"
echo ""

# Run memory checks
echo "=== Memory Error Detection ==="
echo "Running cuda-memcheck for memory errors..."
cuda-memcheck --tool memcheck --leak-check full ./target/debug/rustg tests/fixtures/simple.rs 2>&1 | tee memcheck.log

if grep -q "ERROR SUMMARY: 0 errors" memcheck.log; then
    echo -e "${GREEN}✓ No memory errors detected${NC}"
else
    echo -e "${RED}✗ Memory errors found (see memcheck.log)${NC}"
    exit 1
fi
echo ""

# Run race condition detection
echo "=== Race Condition Detection ==="
echo "Running cuda-memcheck for race conditions..."
cuda-memcheck --tool racecheck ./target/debug/rustg tests/fixtures/simple.rs 2>&1 | tee racecheck.log

if grep -q "ERROR SUMMARY: 0 errors" racecheck.log; then
    echo -e "${GREEN}✓ No race conditions detected${NC}"
else
    echo -e "${YELLOW}⚠ Potential race conditions found (see racecheck.log)${NC}"
fi
echo ""

# Run synchronization checks
echo "=== Synchronization Check ==="
echo "Running cuda-memcheck for synchronization issues..."
cuda-memcheck --tool synccheck ./target/debug/rustg tests/fixtures/simple.rs 2>&1 | tee synccheck.log

if grep -q "ERROR SUMMARY: 0 errors" synccheck.log; then
    echo -e "${GREEN}✓ No synchronization issues detected${NC}"
else
    echo -e "${YELLOW}⚠ Synchronization issues found (see synccheck.log)${NC}"
fi
echo ""

# Run initcheck
echo "=== Uninitialized Memory Check ==="
echo "Running cuda-memcheck for uninitialized memory..."
cuda-memcheck --tool initcheck ./target/debug/rustg tests/fixtures/simple.rs 2>&1 | tee initcheck.log

if grep -q "ERROR SUMMARY: 0 errors" initcheck.log; then
    echo -e "${GREEN}✓ No uninitialized memory access detected${NC}"
else
    echo -e "${RED}✗ Uninitialized memory access found (see initcheck.log)${NC}"
fi
echo ""

# Performance profiling (CUDA 13.0 - Nsight Systems preferred)
echo "=== Performance Profiling ==="

if command -v nsys &> /dev/null; then
    echo "Running Nsight Systems for performance analysis (CUDA 13.0)..."
    # Profile with GPU metrics and CUDA API tracing
    nsys profile \
        --stats=true \
        --trace=cuda,nvtx,osrt \
        --gpu-metrics-device=all \
        --cuda-memory-usage=true \
        --output=rustg_profile \
        ./target/debug/rustg tests/fixtures/large.rs
    echo -e "${GREEN}Performance profile saved to rustg_profile.nsys-rep${NC}"
    echo "View with: nsys-ui rustg_profile.nsys-rep"
elif command -v nvprof &> /dev/null; then
    echo -e "${YELLOW}Warning: nvprof is deprecated in CUDA 13.0${NC}"
    echo "Running legacy nvprof for compatibility..."
    nvprof --print-gpu-trace --print-api-trace ./target/debug/rustg tests/fixtures/large.rs 2>&1 | tee nvprof.log
    echo -e "${GREEN}Legacy profile saved to nvprof.log${NC}"
    echo -e "${YELLOW}Consider installing Nsight Systems for better profiling${NC}"
else
    echo -e "${YELLOW}No profiling tool found (install nsys for CUDA 13.0)${NC}"
    echo "Install with: apt-get install nsight-systems-2024.7"
fi

# Nsight Compute for detailed kernel analysis
if command -v ncu &> /dev/null; then
    echo ""
    echo "Running Nsight Compute for kernel analysis..."
    ncu \
        --kernel-name regex:tokenize \
        --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
        --export rustg_kernel_analysis \
        ./target/debug/rustg tests/fixtures/small.rs
    echo -e "${GREEN}Kernel analysis saved to rustg_kernel_analysis.ncu-rep${NC}"
fi
echo ""

# Run benchmarks
echo "=== Performance Benchmarks ==="
echo "Running tokenizer benchmarks..."
cargo bench --bench tokenizer_bench -- --save-baseline validation

# Extract throughput from benchmark results
if [ -f "target/criterion/tokenizer_comparison/gpu/base/estimates.json" ]; then
    throughput=$(grep -o '"throughput":[^,]*' target/criterion/tokenizer_comparison/gpu/base/estimates.json | head -1 | cut -d: -f2)
    echo "GPU Tokenizer Throughput: $throughput bytes/sec"
    
    # Check if we meet the 1GB/s target
    if (( $(echo "$throughput > 1000000000" | bc -l) )); then
        echo -e "${GREEN}✓ Performance target met (>1GB/s)${NC}"
    else
        echo -e "${YELLOW}⚠ Performance below target (<1GB/s)${NC}"
    fi
fi
echo ""

# Summary
echo "=== Validation Summary ==="
echo ""
echo "Memory Checks:"
echo "  - Memory errors: Check memcheck.log"
echo "  - Race conditions: Check racecheck.log"
echo "  - Synchronization: Check synccheck.log"
echo "  - Uninitialized memory: Check initcheck.log"
echo ""
echo "Performance:"
echo "  - Profiling data: Check nvprof.log or rustg_profile.qdrep"
echo "  - Benchmark results: target/criterion/"
echo ""
echo -e "${GREEN}Validation complete!${NC}"