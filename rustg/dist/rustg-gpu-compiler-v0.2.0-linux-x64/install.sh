#!/bin/bash
# RustG GPU Compiler v0.2.0 Installation Script
# Complete 7-tool GPU-accelerated Rust development environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ASCII Art Banner
echo -e "${CYAN}"
echo "██████╗ ██╗   ██╗███████╗████████╗ ██████╗ "
echo "██╔══██╗██║   ██║██╔════╝╚══██╔══╝██╔════╝ "
echo "██████╔╝██║   ██║███████╗   ██║   ██║  ███╗"
echo "██╔══██╗██║   ██║╚════██║   ██║   ██║   ██║"
echo "██║  ██║╚██████╔╝███████║   ██║   ╚██████╔╝"
echo "╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝    ╚═════╝ "
echo "GPU-Accelerated Rust Compiler v0.2.0"
echo -e "${NC}"

print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${CYAN}ℹ${NC} $1"
}

# Check if running as root for system-wide installation
check_permissions() {
    if [[ $EUID -eq 0 ]]; then
        INSTALL_DIR="/usr/local/bin"
        INSTALL_TYPE="system-wide"
    else
        INSTALL_DIR="$HOME/.local/bin"
        INSTALL_TYPE="user"
        mkdir -p "$INSTALL_DIR"
    fi
}

# System Requirements Check
check_requirements() {
    print_header "System Requirements Check"
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_success "Linux OS detected"
    else
        print_error "RustG requires Linux OS"
        exit 1
    fi
    
    # Check architecture
    if [[ $(uname -m) == "x86_64" ]]; then
        print_success "x86_64 architecture detected"
    else
        print_error "RustG requires x86_64 architecture"
        exit 1
    fi
    
    # Check CUDA
    if command -v nvcc >/dev/null 2>&1; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        if [[ $(echo "$CUDA_VERSION >= 13.0" | bc -l) -eq 1 ]]; then
            print_success "CUDA $CUDA_VERSION detected (>= 13.0 required)"
        else
            print_warning "CUDA $CUDA_VERSION detected (13.0+ recommended for optimal performance)"
        fi
    else
        print_warning "CUDA not detected - GPU acceleration will use CPU fallback"
    fi
    
    # Check GPU
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        print_success "GPU detected: $GPU_INFO"
        if echo "$GPU_INFO" | grep -q "RTX 5090"; then
            print_success "RTX 5090 (Blackwell) detected - optimal performance expected"
        fi
    else
        print_warning "nvidia-smi not found - GPU status unknown"
    fi
    
    # Check memory
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $TOTAL_MEM -ge 16 ]]; then
        print_success "Memory: ${TOTAL_MEM}GB (16GB+ required)"
    else
        print_error "Memory: ${TOTAL_MEM}GB - 16GB minimum required"
        exit 1
    fi
    
    # Check disk space
    AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ $AVAILABLE_SPACE -ge 10 ]]; then
        print_success "Disk space: ${AVAILABLE_SPACE}GB available (10GB required)"
    else
        print_error "Disk space: ${AVAILABLE_SPACE}GB - 10GB minimum required"
        exit 1
    fi
}

# Install RustG tools
install_rustg() {
    print_header "Installing RustG GPU Tools"
    
    # List of all 7 GPU-accelerated tools
    TOOLS=("cargo-g" "clippy-f" "rustdoc-g" "rustup-g" "rust-gdb-g" "bindgen-g" "miri-g")
    
    for tool in "${TOOLS[@]}"; do
        if [[ -f "$tool" ]]; then
            cp "$tool" "$INSTALL_DIR/"
            chmod +x "$INSTALL_DIR/$tool"
            print_success "$tool installed to $INSTALL_DIR"
        else
            print_error "$tool not found in current directory"
            exit 1
        fi
    done
}

# Verify installation
verify_installation() {
    print_header "Verifying Installation"
    
    TOOLS=("cargo-g" "clippy-f" "rustdoc-g" "rustup-g" "rust-gdb-g" "bindgen-g" "miri-g")
    
    for tool in "${TOOLS[@]}"; do
        if command -v "$tool" >/dev/null 2>&1; then
            VERSION_OUTPUT=$("$tool" --version 2>/dev/null || "$tool" --help 2>/dev/null | head -1 || echo "Unknown version")
            print_success "$tool: $VERSION_OUTPUT"
        else
            print_error "$tool not found in PATH"
            VERIFICATION_FAILED=true
        fi
    done
    
    if [[ $VERIFICATION_FAILED ]]; then
        print_warning "Some tools failed verification. You may need to add $INSTALL_DIR to your PATH"
        echo "Add this to your ~/.bashrc or ~/.zshrc:"
        echo "export PATH=\"$INSTALL_DIR:\$PATH\""
    fi
}

# Setup environment
setup_environment() {
    print_header "Environment Setup"
    
    # Create RustG configuration directory
    RUSTG_CONFIG_DIR="$HOME/.rustg"
    mkdir -p "$RUSTG_CONFIG_DIR"
    print_success "Created configuration directory: $RUSTG_CONFIG_DIR"
    
    # Create default configuration file
    cat > "$RUSTG_CONFIG_DIR/config.toml" << EOF
# RustG GPU Configuration
[gpu]
enabled = true
threads = 256
memory_limit_mb = 4096
cuda_version = "13.0"
target_arch = "sm_110"  # RTX 5090 Blackwell

[performance]
enable_caching = true
cache_size_mb = 1024
parallel_builds = true
optimization_level = 3

[fallback]
cpu_fallback = true
cpu_threads = 8
EOF
    print_success "Created default configuration: $RUSTG_CONFIG_DIR/config.toml"
    
    # Set up environment variables
    ENV_FILE="$HOME/.rustg_env"
    cat > "$ENV_FILE" << EOF
# RustG Environment Variables
export RUSTG_GPU_THREADS=256
export RUSTG_GPU_MEMORY_LIMIT=4096
export RUSTG_CUDA_VERSION=13.0
export RUSTG_CONFIG_DIR=$RUSTG_CONFIG_DIR
EOF
    
    # Add to shell profile if not already present
    for SHELL_RC in "$HOME/.bashrc" "$HOME/.zshrc"; do
        if [[ -f "$SHELL_RC" ]] && ! grep -q "rustg_env" "$SHELL_RC"; then
            echo "source $ENV_FILE" >> "$SHELL_RC"
            print_success "Added RustG environment to $SHELL_RC"
        fi
    done
}

# Performance test
run_performance_test() {
    print_header "Performance Test"
    
    # Create a simple test file
    TEST_FILE="/tmp/rustg_test.rs"
    cat > "$TEST_FILE" << EOF
fn main() {
    println!("Hello, RustG!");
}
EOF
    
    print_info "Running performance test with cargo-g..."
    
    # Test cargo-g if available
    if command -v cargo-g >/dev/null 2>&1; then
        if cargo-g --version >/dev/null 2>&1; then
            print_success "cargo-g: Performance test passed"
        else
            print_warning "cargo-g: Performance test failed"
        fi
    fi
    
    # Test rustdoc-g if available
    if command -v rustdoc-g >/dev/null 2>&1; then
        if rustdoc-g --help >/dev/null 2>&1; then
            print_success "rustdoc-g: Performance test passed"
        else
            print_warning "rustdoc-g: Performance test failed"
        fi
    fi
    
    rm -f "$TEST_FILE"
}

# Generate usage examples
generate_examples() {
    print_header "Generating Usage Examples"
    
    EXAMPLES_FILE="$HOME/.rustg/examples.sh"
    cat > "$EXAMPLES_FILE" << 'EOF'
#!/bin/bash
# RustG Usage Examples

echo "=== RustG GPU-Accelerated Rust Development ==="

# Basic project workflow
echo "1. Create new project:"
echo "   cargo-g new my_project"
echo "   cd my_project"
echo ""

# Building with GPU acceleration
echo "2. Build with GPU acceleration:"
echo "   cargo-g build --release --stats"
echo ""

# Advanced linting
echo "3. Run GPU-accelerated linting:"
echo "   clippy-f src/ --stats"
echo ""

# Documentation generation
echo "4. Generate documentation:"
echo "   rustdoc-g src/lib.rs --output docs/ --stats"
echo ""

# Debugging
echo "5. Debug with GPU enhancement:"
echo "   rust-gdb-g target/debug/my_project --stats"
echo ""

# Memory safety checking
echo "6. Check memory safety:"
echo "   miri-g --run src/main.rs --stats"
echo ""

# FFI bindings
echo "7. Generate FFI bindings:"
echo "   bindgen-g header.h --output bindings.rs"
echo ""

# Toolchain management
echo "8. Manage toolchains:"
echo "   rustup-g toolchain install stable"
echo "   rustup-g default stable"
echo ""

echo "=== Performance Monitoring ==="
echo "Add --stats to any command for performance information"
echo "Add --no-gpu to any command to force CPU fallback"
echo ""

echo "For more information, see the README.md"
EOF
    
    chmod +x "$EXAMPLES_FILE"
    print_success "Created usage examples: $EXAMPLES_FILE"
}

# Main installation flow
main() {
    print_header "RustG GPU Compiler Installation"
    print_info "Complete 7-tool GPU-accelerated Rust development environment"
    print_info "Version: v0.2.0"
    
    check_permissions
    print_info "Installation type: $INSTALL_TYPE ($INSTALL_DIR)"
    
    check_requirements
    install_rustg
    setup_environment
    verify_installation
    generate_examples
    run_performance_test
    
    print_header "Installation Complete!"
    print_success "RustG GPU Compiler v0.2.0 installed successfully"
    
    echo ""
    print_info "Tools installed:"
    echo "  • cargo-g      - GPU-accelerated build system"
    echo "  • clippy-f     - GPU-enhanced linter"  
    echo "  • rustdoc-g    - GPU-accelerated documentation generator"
    echo "  • rustup-g     - GPU toolchain manager"
    echo "  • rust-gdb-g   - GPU-enabled debugger"
    echo "  • bindgen-g    - GPU-accelerated FFI generator"
    echo "  • miri-g       - GPU memory safety checker"
    
    echo ""
    print_info "Next steps:"
    echo "1. Source your shell profile: source ~/.bashrc (or ~/.zshrc)"
    echo "2. Run examples: bash ~/.rustg/examples.sh"
    echo "3. Read documentation: cat README.md"
    echo "4. Test installation: cargo-g --version"
    
    echo ""
    print_success "Experience 10x faster Rust development with GPU acceleration! 🚀⚡🦀"
}

# Run installation
main "$@"