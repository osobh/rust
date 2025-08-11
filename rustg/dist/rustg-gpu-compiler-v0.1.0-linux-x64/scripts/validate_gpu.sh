#!/bin/bash

# GPU Validation Script for RustG Compiler
# Validates CUDA installation and GPU capabilities

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔍 RustG GPU Validation Script${NC}"
echo "Checking GPU setup and CUDA installation..."
echo ""

# Check if CUDA is installed
check_cuda() {
    echo -e "${BLUE}Checking CUDA installation...${NC}"
    
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        echo -e "${GREEN}✓${NC} CUDA compiler found: version $CUDA_VERSION"
        
        # Check if CUDA version is supported
        if [[ $(echo "$CUDA_VERSION >= 11.0" | bc -l) -eq 1 ]]; then
            echo -e "${GREEN}✓${NC} CUDA version is supported (>= 11.0)"
        else
            echo -e "${RED}✗${NC} CUDA version too old (need >= 11.0, found $CUDA_VERSION)"
            return 1
        fi
    else
        echo -e "${RED}✗${NC} CUDA compiler (nvcc) not found"
        echo "Please install CUDA toolkit from https://developer.nvidia.com/cuda-downloads"
        return 1
    fi
}

# Check GPU hardware
check_gpu_hardware() {
    echo -e "\n${BLUE}Checking GPU hardware...${NC}"
    
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✓${NC} nvidia-smi found"
        
        # Get GPU information
        GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
        echo -e "${GREEN}✓${NC} Found $GPU_COUNT GPU(s)"
        
        # List GPU details
        echo -e "\n${BLUE}GPU Details:${NC}"
        nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader | while read line; do
            echo "  GPU $line"
        done
        
        # Check compute capability
        MIN_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -n | head -1)
        if [[ $(echo "$MIN_COMPUTE_CAP >= 6.0" | bc -l) -eq 1 ]]; then
            echo -e "${GREEN}✓${NC} GPU compute capability sufficient (>= 6.0)"
        else
            echo -e "${YELLOW}⚠${NC} GPU compute capability may be too low (found $MIN_COMPUTE_CAP, need >= 6.0)"
        fi
        
    else
        echo -e "${RED}✗${NC} nvidia-smi not found"
        echo "Please install NVIDIA drivers"
        return 1
    fi
}

# Check GPU memory
check_gpu_memory() {
    echo -e "\n${BLUE}Checking GPU memory...${NC}"
    
    TOTAL_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,units=MB | head -1 | awk '{print $1}')
    USED_MEMORY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,units=MB | head -1 | awk '{print $1}')
    FREE_MEMORY=$((TOTAL_MEMORY - USED_MEMORY))
    
    echo "  Total GPU Memory: ${TOTAL_MEMORY}MB"
    echo "  Used GPU Memory:  ${USED_MEMORY}MB"
    echo "  Free GPU Memory:  ${FREE_MEMORY}MB"
    
    if [[ $FREE_MEMORY -gt 4000 ]]; then
        echo -e "${GREEN}✓${NC} Sufficient GPU memory available"
    elif [[ $FREE_MEMORY -gt 2000 ]]; then
        echo -e "${YELLOW}⚠${NC} Limited GPU memory available - consider closing GPU applications"
    else
        echo -e "${RED}✗${NC} Insufficient GPU memory available"
        return 1
    fi
}

# Check CUDA libraries
check_cuda_libraries() {
    echo -e "\n${BLUE}Checking CUDA libraries...${NC}"
    
    # Check for CUDA runtime library
    if ldconfig -p | grep -q "libcudart"; then
        echo -e "${GREEN}✓${NC} CUDA runtime library found"
    else
        echo -e "${YELLOW}⚠${NC} CUDA runtime library not found in ldconfig"
    fi
    
    # Check for cuBLAS
    if ldconfig -p | grep -q "libcublas"; then
        echo -e "${GREEN}✓${NC} cuBLAS library found"
    else
        echo -e "${YELLOW}⚠${NC} cuBLAS library not found"
    fi
    
    # Check CUDA installation path
    if [[ -d "/usr/local/cuda" ]]; then
        echo -e "${GREEN}✓${NC} CUDA installation found at /usr/local/cuda"
        CUDA_PATH="/usr/local/cuda"
    elif [[ -d "/opt/cuda" ]]; then
        echo -e "${GREEN}✓${NC} CUDA installation found at /opt/cuda"  
        CUDA_PATH="/opt/cuda"
    else
        echo -e "${YELLOW}⚠${NC} CUDA installation path not found in standard locations"
        CUDA_PATH=""
    fi
}

# Test basic CUDA functionality
test_cuda_basic() {
    echo -e "\n${BLUE}Testing basic CUDA functionality...${NC}"
    
    # Create simple CUDA test
    cat > /tmp/cuda_test.cu << 'EOF'
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void hello_kernel() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main() {
    cudaDeviceProp prop;
    int device = 0;
    
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    printf("Device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Memory: %.2f GB\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    
    hello_kernel<<<1, 4>>>();
    cudaDeviceSynchronize();
    
    return 0;
}
EOF

    # Compile and run CUDA test
    if nvcc /tmp/cuda_test.cu -o /tmp/cuda_test 2>/dev/null; then
        echo -e "${GREEN}✓${NC} CUDA compilation successful"
        
        if /tmp/cuda_test > /tmp/cuda_output 2>&1; then
            echo -e "${GREEN}✓${NC} CUDA execution successful"
            if grep -q "Hello from GPU" /tmp/cuda_output; then
                echo -e "${GREEN}✓${NC} GPU kernel execution verified"
            fi
        else
            echo -e "${RED}✗${NC} CUDA execution failed"
            cat /tmp/cuda_output
        fi
    else
        echo -e "${RED}✗${NC} CUDA compilation failed"
    fi
    
    # Cleanup
    rm -f /tmp/cuda_test.cu /tmp/cuda_test /tmp/cuda_output
}

# Check RustG tools
check_rustg_tools() {
    echo -e "\n${BLUE}Checking RustG tools...${NC}"
    
    if command -v cargo-g &> /dev/null; then
        echo -e "${GREEN}✓${NC} cargo-g found"
        cargo-g --version
    else
        echo -e "${YELLOW}⚠${NC} cargo-g not found - install RustG GPU compiler"
    fi
    
    if command -v clippy-f &> /dev/null; then
        echo -e "${GREEN}✓${NC} clippy-f found"
        clippy-f --version  
    else
        echo -e "${YELLOW}⚠${NC} clippy-f not found - install RustG GPU compiler"
    fi
}

# Generate report
generate_report() {
    echo -e "\n${BLUE}📊 GPU Validation Report${NC}"
    echo "=========================="
    
    # System info
    echo "System: $(uname -s) $(uname -r)"
    echo "Architecture: $(uname -m)"
    
    # CUDA info
    if command -v nvcc &> /dev/null; then
        echo "CUDA: $(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)"
    fi
    
    # GPU info
    if command -v nvidia-smi &> /dev/null; then
        echo "Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
        echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
        echo "Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
        echo "Compute: $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
    fi
    
    echo ""
    if [[ $VALIDATION_SUCCESS -eq 1 ]]; then
        echo -e "${GREEN}✅ GPU validation passed - Ready for RustG GPU compilation!${NC}"
    else
        echo -e "${YELLOW}⚠️  GPU validation has warnings - RustG may work with limitations${NC}"
    fi
}

# Main validation flow
main() {
    VALIDATION_SUCCESS=1
    
    check_cuda || VALIDATION_SUCCESS=0
    check_gpu_hardware || VALIDATION_SUCCESS=0  
    check_gpu_memory || VALIDATION_SUCCESS=0
    check_cuda_libraries
    test_cuda_basic
    check_rustg_tools
    
    generate_report
    
    return $VALIDATION_SUCCESS
}

# Handle command line arguments
case "${1:-validate}" in
    "validate")
        main
        ;;
    "quick")
        check_cuda
        check_gpu_hardware
        check_rustg_tools
        ;;
    "install-deps")
        echo "Installing GPU validation dependencies..."
        sudo apt-get update
        sudo apt-get install -y bc nvidia-utils-* cuda-toolkit
        ;;
    *)
        echo "Usage: $0 [validate|quick|install-deps]"
        echo "  validate     - Full GPU validation (default)"
        echo "  quick        - Quick check of GPU and CUDA"
        echo "  install-deps - Install validation dependencies"
        exit 1
        ;;
esac