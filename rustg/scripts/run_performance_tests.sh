#!/bin/bash
# Performance validation script for rustg GPU compiler

set -e

echo "🚀 rustg GPU Compiler - Performance Validation"
echo "=============================================="
echo ""

# Check for CUDA installation
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA not found. Please install CUDA toolkit."
    exit 1
fi

# Check for GPU
if ! nvidia-smi &> /dev/null; then
    echo "❌ No NVIDIA GPU detected or nvidia-smi not available."
    exit 1
fi

echo "✅ CUDA installation found"
nvcc --version | head -n 1
echo ""

echo "📊 GPU Information:"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
echo ""

# Build directory
BUILD_DIR="build_perf"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

echo "🔨 Building performance tests..."
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) perf_validation

echo ""
echo "🏃 Running performance validation..."
echo ""

# Set CUDA environment for profiling
export CUDA_LAUNCH_BLOCKING=0
export CUDA_DEVICE_MAX_CONNECTIONS=32

# Run performance tests
if [ -f ./perf_validation ]; then
    ./perf_validation
else
    echo "❌ Performance validation executable not found"
    exit 1
fi

echo ""
echo "📈 Performance Summary"
echo "====================="

# If nsight is available, offer to profile
if command -v nsys &> /dev/null; then
    echo ""
    echo "💡 Tip: For detailed profiling, run:"
    echo "   nsys profile --stats=true ./perf_validation"
fi

echo ""
echo "✅ Performance validation complete!"