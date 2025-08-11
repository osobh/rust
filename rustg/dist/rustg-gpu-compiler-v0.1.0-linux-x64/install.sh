#!/bin/bash

# RustG GPU Compiler Installation Script
# Version 0.1.0 - Linux x64
# Installs cargo-g and clippy-f GPU-accelerated development tools

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

INSTALL_DIR="$HOME/.cargo/bin"
SYSTEM_INSTALL_DIR="/usr/local/bin"

echo -e "${BLUE}🚀 RustG GPU Compiler Installation${NC}"
echo "Version: 0.1.0"
echo "Platform: Linux x64"
echo ""

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    DEFAULT_INSTALL_DIR="$SYSTEM_INSTALL_DIR"
    print_info "Running as root - will install to system directory"
else
    DEFAULT_INSTALL_DIR="$INSTALL_DIR"
    print_info "Running as user - will install to user directory"
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
BIN_DIR="$SCRIPT_DIR/bin"

# Check if binaries exist
if [[ ! -f "$BIN_DIR/cargo-g" ]] || [[ ! -f "$BIN_DIR/clippy-f" ]]; then
    print_error "Binary files not found in $BIN_DIR"
    print_error "Please ensure cargo-g and clippy-f are in the bin/ directory"
    exit 1
fi

# Check for CUDA installation
check_cuda() {
    print_info "Checking CUDA installation..."
    
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        print_status "CUDA compiler found: version $CUDA_VERSION"
        
        if command -v nvidia-smi &> /dev/null; then
            GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
            print_status "GPU detected: $GPU_INFO"
        else
            print_warning "nvidia-smi not found - GPU status unknown"
        fi
    else
        print_warning "CUDA compiler (nvcc) not found"
        print_warning "RustG tools will work but GPU acceleration may be limited"
    fi
}

# Install function
install_binaries() {
    local target_dir="$1"
    
    print_info "Installing to: $target_dir"
    
    # Create target directory if it doesn't exist
    if [[ ! -d "$target_dir" ]]; then
        print_info "Creating directory: $target_dir"
        mkdir -p "$target_dir" || {
            print_error "Failed to create directory: $target_dir"
            return 1
        }
    fi
    
    # Copy binaries
    print_info "Installing cargo-g..."
    cp "$BIN_DIR/cargo-g" "$target_dir/" || {
        print_error "Failed to copy cargo-g"
        return 1
    }
    
    print_info "Installing clippy-f..."
    cp "$BIN_DIR/clippy-f" "$target_dir/" || {
        print_error "Failed to copy clippy-f"
        return 1
    }
    
    # Ensure binaries are executable
    chmod +x "$target_dir/cargo-g" "$target_dir/clippy-f" || {
        print_error "Failed to set execute permissions"
        return 1
    }
    
    print_status "Binaries installed successfully"
    return 0
}

# Verify installation
verify_installation() {
    local target_dir="$1"
    
    print_info "Verifying installation..."
    
    # Check if binaries exist and are executable
    if [[ -x "$target_dir/cargo-g" ]] && [[ -x "$target_dir/clippy-f" ]]; then
        print_status "Installation verification passed"
        
        # Test basic functionality
        print_info "Testing cargo-g..."
        if "$target_dir/cargo-g" --version &> /dev/null; then
            print_status "cargo-g is working"
        else
            print_warning "cargo-g may have issues (check dependencies)"
        fi
        
        print_info "Testing clippy-f..."
        if "$target_dir/clippy-f" --version &> /dev/null; then
            print_status "clippy-f is working"
        else
            print_warning "clippy-f may have issues (check dependencies)"
        fi
    else
        print_error "Installation verification failed"
        return 1
    fi
}

# Main installation process
main() {
    echo ""
    print_info "Starting installation process..."
    
    # Check CUDA
    check_cuda
    echo ""
    
    # Install binaries
    if install_binaries "$DEFAULT_INSTALL_DIR"; then
        echo ""
        verify_installation "$DEFAULT_INSTALL_DIR"
        echo ""
        
        # Success message
        print_status "Installation completed successfully!"
        echo ""
        echo -e "${GREEN}Next steps:${NC}"
        echo "1. Add $DEFAULT_INSTALL_DIR to your PATH if not already there"
        echo "2. Restart your shell or run: source ~/.bashrc"
        echo "3. Test with: cargo-g --help"
        echo "4. Test with: clippy-f --help"
        echo ""
        echo -e "${BLUE}Usage:${NC}"
        echo "• Replace 'cargo build' with 'cargo-g build' for GPU acceleration"
        echo "• Replace 'cargo clippy' with 'clippy-f' for GPU linting"
        echo ""
        echo -e "${YELLOW}Note:${NC} Ensure CUDA toolkit is installed for full GPU acceleration"
        
    else
        print_error "Installation failed"
        echo ""
        echo "Troubleshooting:"
        echo "• Check if you have write permissions to $DEFAULT_INSTALL_DIR"
        echo "• Try running with sudo for system installation"
        echo "• Ensure CUDA toolkit is properly installed"
        exit 1
    fi
}

# Handle command line arguments
case "${1:-install}" in
    "install")
        main
        ;;
    "uninstall")
        print_info "Uninstalling RustG GPU Compiler..."
        rm -f "$DEFAULT_INSTALL_DIR/cargo-g" "$DEFAULT_INSTALL_DIR/clippy-f"
        print_status "Uninstallation completed"
        ;;
    "check")
        check_cuda
        verify_installation "$DEFAULT_INSTALL_DIR"
        ;;
    *)
        echo "Usage: $0 [install|uninstall|check]"
        echo "  install   - Install RustG GPU Compiler (default)"
        echo "  uninstall - Remove installed binaries"
        echo "  check     - Check installation and requirements"
        exit 1
        ;;
esac