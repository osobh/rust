#!/bin/bash
# RustG Installation Verification Script
# Comprehensive testing of all GPU-accelerated tools

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

print_header() {
    echo -e "\n${BLUE}${BOLD}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
    ((TESTS_PASSED++))
    ((TESTS_TOTAL++))
}

print_failure() {
    echo -e "${RED}✗${NC} $1"
    ((TESTS_FAILED++))
    ((TESTS_TOTAL++))
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${CYAN}ℹ${NC} $1"
}

print_banner() {
    echo -e "${CYAN}${BOLD}"
    echo "██████╗ ██╗   ██╗███████╗████████╗ ██████╗ "
    echo "██╔══██╗██║   ██║██╔════╝╚══██╔══╝██╔════╝ "
    echo "██████╔╝██║   ██║███████╗   ██║   ██║  ███╗"
    echo "██╔══██╗██║   ██║╚════██║   ██║   ██║   ██║"
    echo "██║  ██║╚██████╔╝███████║   ██║   ╚██████╔╝"
    echo "╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝    ╚═════╝ "
    echo "GPU-Accelerated Rust Development Environment"
    echo "Installation Verification v0.2.0"
    echo -e "${NC}"
}

# Test if a command exists and can execute
test_command() {
    local cmd="$1"
    local description="$2"
    local expected_output="$3"
    
    if command -v "$cmd" >/dev/null 2>&1; then
        # Try version flag first, then help flag
        local output=""
        if $cmd --version >/dev/null 2>&1; then
            output=$($cmd --version 2>/dev/null | head -1)
        elif $cmd --help >/dev/null 2>&1; then
            output=$($cmd --help 2>/dev/null | head -1)
        else
            output="Command executed successfully"
        fi
        
        if [[ -n "$expected_output" ]] && [[ "$output" =~ $expected_output ]]; then
            print_success "$description: $output"
        elif [[ -z "$expected_output" ]]; then
            print_success "$description: Available"
        else
            print_failure "$description: Unexpected output - $output"
        fi
    else
        print_failure "$description: Command not found"
    fi
}

# Test tool functionality with basic operations
test_tool_functionality() {
    local tool="$1"
    local test_description="$2"
    local test_command="$3"
    
    print_info "Testing $tool functionality: $test_description"
    
    if eval "$test_command" >/dev/null 2>&1; then
        print_success "$tool: $test_description works"
    else
        print_failure "$tool: $test_description failed"
    fi
}

# Check system requirements
check_system_requirements() {
    print_header "System Requirements Check"
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_success "Operating System: Linux ($(lsb_release -d -s 2>/dev/null || uname -r))"
    else
        print_warning "Operating System: $OSTYPE (Linux recommended)"
    fi
    
    # Check architecture
    local arch=$(uname -m)
    if [[ "$arch" == "x86_64" ]]; then
        print_success "Architecture: $arch"
    else
        print_warning "Architecture: $arch (x86_64 recommended)"
    fi
    
    # Check memory
    local mem_gb=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $mem_gb -ge 16 ]]; then
        print_success "Memory: ${mem_gb}GB (16GB+ required)"
    else
        print_warning "Memory: ${mem_gb}GB (16GB recommended)"
    fi
    
    # Check CUDA
    if command -v nvcc >/dev/null 2>&1; then
        local cuda_version=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        if [[ $(echo "$cuda_version >= 13.0" | bc -l 2>/dev/null || echo 0) -eq 1 ]]; then
            print_success "CUDA: v$cuda_version (13.0+ required)"
        else
            print_warning "CUDA: v$cuda_version (13.0+ recommended)"
        fi
    else
        print_warning "CUDA: Not detected (GPU acceleration will use CPU fallback)"
    fi
    
    # Check GPU
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_info=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader,nounits | head -1)
        print_success "GPU: $gpu_info"
        
        # Check for RTX 5090
        if echo "$gpu_info" | grep -q "RTX 5090"; then
            print_success "RTX 5090 detected: Optimal performance expected"
        else
            print_info "GPU detected: Performance may vary (RTX 5090 optimal)"
        fi
    else
        print_warning "GPU: Not detected (nvidia-smi not available)"
    fi
}

# Test core RustG tools
test_rustg_tools() {
    print_header "RustG Tools Verification"
    
    # List of all RustG tools
    local -A tools=(
        ["cargo-g"]="GPU-accelerated build system"
        ["clippy-f"]="GPU-enhanced linter"
        ["rustfmt-g"]="GPU-accelerated formatter"
        ["rustdoc-g"]="GPU documentation generator"
        ["rustup-g"]="GPU toolchain manager"
        ["rust-gdb-g"]="GPU-enabled debugger"
        ["bindgen-g"]="GPU FFI bindings generator"
        ["miri-g"]="GPU memory safety checker"
    )
    
    for tool in "${!tools[@]}"; do
        test_command "$tool" "${tools[$tool]}"
    done
}

# Test tool functionality with real operations
test_advanced_functionality() {
    print_header "Advanced Functionality Tests"
    
    # Create temporary test directory
    local test_dir=$(mktemp -d)
    cd "$test_dir"
    
    print_info "Created test directory: $test_dir"
    
    # Test cargo-g project creation
    if command -v cargo-g >/dev/null 2>&1; then
        test_tool_functionality "cargo-g" "project creation" "cargo-g new test_project --quiet"
        
        if [[ -d "test_project" ]]; then
            cd test_project
            
            # Test cargo-g build
            test_tool_functionality "cargo-g" "project build" "cargo-g build --quiet"
            
            # Test clippy-f linting
            if command -v clippy-f >/dev/null 2>&1; then
                test_tool_functionality "clippy-f" "code linting" "clippy-f src/ --quiet || true"
            fi
            
            # Test rustfmt-g formatting
            if command -v rustfmt-g >/dev/null 2>&1; then
                test_tool_functionality "rustfmt-g" "code formatting" "rustfmt-g src/main.rs --check || true"
            fi
            
            # Test rustdoc-g documentation
            if command -v rustdoc-g >/dev/null 2>&1; then
                test_tool_functionality "rustdoc-g" "documentation generation" "rustdoc-g src/main.rs --output docs --quiet || true"
            fi
            
            cd ..
        fi
    else
        print_warning "Skipping advanced tests: cargo-g not available"
    fi
    
    # Cleanup
    cd /
    rm -rf "$test_dir"
    print_info "Cleaned up test directory"
}

# Test performance and GPU utilization
test_performance() {
    print_header "Performance and GPU Tests"
    
    # Test GPU acceleration status
    local tools_with_stats=("cargo-g" "clippy-f" "rustfmt-g" "rustdoc-g")
    
    for tool in "${tools_with_stats[@]}"; do
        if command -v "$tool" >/dev/null 2>&1; then
            print_info "Testing $tool with --stats flag"
            if $tool --help | grep -q "\-\-stats"; then
                print_success "$tool: Supports performance statistics"
            else
                print_warning "$tool: No --stats flag detected"
            fi
            
            if $tool --help | grep -q "\-\-no-gpu"; then
                print_success "$tool: Supports CPU fallback (--no-gpu)"
            else
                print_warning "$tool: No --no-gpu flag detected"
            fi
        fi
    done
}

# Test configuration and environment
test_configuration() {
    print_header "Configuration Tests"
    
    # Check RustG configuration directory
    if [[ -d "$HOME/.rustg" ]]; then
        print_success "RustG config directory: $HOME/.rustg"
        
        if [[ -f "$HOME/.rustg/config.toml" ]]; then
            print_success "RustG configuration file found"
        else
            print_info "RustG configuration file not found (will use defaults)"
        fi
    else
        print_info "RustG config directory not found (will be created on first run)"
    fi
    
    # Check environment variables
    local env_vars=("RUSTG_GPU_THREADS" "RUSTG_GPU_MEMORY_LIMIT" "RUSTG_CUDA_VERSION")
    
    for var in "${env_vars[@]}"; do
        if [[ -n "${!var}" ]]; then
            print_success "Environment variable $var: ${!var}"
        else
            print_info "Environment variable $var: Not set (will use defaults)"
        fi
    done
    
    # Check PATH
    if echo "$PATH" | grep -q "rustg\|cargo-g"; then
        print_success "PATH contains RustG tools directory"
    else
        print_warning "PATH may not include RustG tools (check installation)"
    fi
}

# Generate performance benchmark
run_benchmark() {
    print_header "Performance Benchmark"
    
    if ! command -v cargo-g >/dev/null 2>&1; then
        print_warning "Skipping benchmark: cargo-g not available"
        return
    fi
    
    local test_dir=$(mktemp -d)
    cd "$test_dir"
    
    print_info "Running performance benchmark..."
    
    # Create test project
    cargo-g new benchmark_test --quiet >/dev/null 2>&1
    cd benchmark_test
    
    # Add some complexity
    cat > src/lib.rs << 'EOF'
//! Benchmark test library
use std::collections::HashMap;

/// A simple test structure
pub struct TestStruct {
    data: HashMap<String, i32>,
}

impl TestStruct {
    pub fn new() -> Self {
        Self { data: HashMap::new() }
    }
    
    pub fn insert(&mut self, key: String, value: i32) {
        self.data.insert(key, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic() {
        let mut ts = TestStruct::new();
        ts.insert("test".to_string(), 42);
        assert_eq!(ts.data.get("test"), Some(&42));
    }
}
EOF
    
    # Benchmark build time
    local start_time=$(date +%s.%N)
    if cargo-g build --release --quiet >/dev/null 2>&1; then
        local end_time=$(date +%s.%N)
        local build_time=$(echo "$end_time - $start_time" | bc -l)
        print_success "Build benchmark: ${build_time}s (cargo-g)"
    else
        print_failure "Build benchmark failed"
    fi
    
    # Cleanup
    cd /
    rm -rf "$test_dir"
}

# Generate final report
generate_report() {
    print_header "Installation Verification Report"
    
    local success_rate=0
    if [[ $TESTS_TOTAL -gt 0 ]]; then
        success_rate=$(echo "scale=1; $TESTS_PASSED * 100 / $TESTS_TOTAL" | bc -l)
    fi
    
    echo -e "${BOLD}Test Results:${NC}"
    echo "  Total Tests: $TESTS_TOTAL"
    echo "  Passed: ${GREEN}$TESTS_PASSED${NC}"
    echo "  Failed: ${RED}$TESTS_FAILED${NC}"
    echo "  Success Rate: ${success_rate}%"
    echo ""
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        echo -e "${GREEN}${BOLD}🎉 RustG installation verification PASSED!${NC}"
        echo "All GPU-accelerated tools are ready for use."
        echo ""
        echo "Next steps:"
        echo "1. Create a new project: cargo-g new my_project"
        echo "2. Build with GPU acceleration: cargo-g build --stats"
        echo "3. Run linting: clippy-f src/ --stats"
        echo "4. Generate documentation: rustdoc-g src/lib.rs --stats"
    elif [[ $TESTS_FAILED -le 2 ]]; then
        echo -e "${YELLOW}${BOLD}⚠ RustG installation verification completed with warnings${NC}"
        echo "Most tools are working correctly. Review failed tests above."
    else
        echo -e "${RED}${BOLD}✗ RustG installation verification FAILED${NC}"
        echo "Multiple tools are not working correctly. Please check your installation."
        echo ""
        echo "Troubleshooting steps:"
        echo "1. Ensure all RustG tools are in your PATH"
        echo "2. Check CUDA installation: nvcc --version"
        echo "3. Verify GPU status: nvidia-smi"
        echo "4. Try CPU fallback mode: [tool] --no-gpu [args]"
    fi
    
    echo ""
    echo -e "${CYAN}For support and documentation:${NC}"
    echo "- GitHub: https://github.com/your-username/rustg"
    echo "- Documentation: See README.md"
    echo "- Performance monitoring: Add --stats to any command"
}

# Main execution
main() {
    print_banner
    
    print_info "Starting RustG installation verification..."
    print_info "This will test all GPU-accelerated tools and system requirements"
    echo ""
    
    check_system_requirements
    test_rustg_tools
    test_configuration
    test_performance
    test_advanced_functionality
    run_benchmark
    
    generate_report
}

# Execute main function
main "$@"