#!/bin/bash
# RustG System-Wide Uninstallation Script
# Removes all RustG tools from the system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

print_header() {
    echo -e "\n${BLUE}${BOLD}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✅${NC} $1"
}

print_info() {
    echo -e "${CYAN}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_banner() {
    echo -e "${RED}${BOLD}"
    echo "██████╗ ██╗   ██╗███████╗████████╗ ██████╗ "
    echo "██╔══██╗██║   ██║██╔════╝╚══██╔══╝██╔════╝ "
    echo "██████╔╝██║   ██║███████╗   ██║   ██║  ███╗"
    echo "██╔══██╗██║   ██║╚════██║   ██║   ██║   ██║"
    echo "██║  ██║╚██████╔╝███████║   ██║   ╚██████╔╝"
    echo "╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝    ╚═════╝ "
    echo "GPU-Accelerated Rust Development Environment"
    echo "System-Wide Uninstallation"
    echo -e "${NC}"
}

# Determine sudo requirement
check_permissions() {
    if [[ $EUID -eq 0 ]]; then
        NEED_SUDO=""
        INSTALL_DIR="/usr/local/bin"
    else
        NEED_SUDO="sudo"
        INSTALL_DIR="/usr/local/bin"
        
        if ! command -v sudo >/dev/null 2>&1; then
            print_error "sudo not available. Please run as root."
            exit 1
        fi
    fi
}

# Remove RustG binaries
remove_binaries() {
    print_header "Removing RustG Binaries"
    
    local tools=("rustg" "cargo-g" "clippy-f" "rustfmt-g" "rustdoc-g" "rustup-g" "rust-analyzer-g" "rust-gdb-g" "bindgen-g" "miri-g")
    local removed_count=0
    
    for tool in "${tools[@]}"; do
        if [[ -f "$INSTALL_DIR/$tool" ]]; then
            print_info "Removing $tool..."
            $NEED_SUDO rm -f "$INSTALL_DIR/$tool"
            
            if [[ ! -f "$INSTALL_DIR/$tool" ]]; then
                print_success "$tool removed"
                ((removed_count++))
            else
                print_error "Failed to remove $tool"
            fi
        else
            print_info "$tool not found (already removed)"
        fi
    done
    
    print_success "Removed $removed_count tools"
}

# Remove environment files
remove_environment() {
    print_header "Removing Environment Files"
    
    local files=(
        "$INSTALL_DIR/rustg-env.sh"
        "$INSTALL_DIR/rustg-usage.txt"
        "/etc/profile.d/rustg.sh"
    )
    
    for file in "${files[@]}"; do
        if [[ -f "$file" ]]; then
            print_info "Removing $file..."
            $NEED_SUDO rm -f "$file"
            print_success "$(basename "$file") removed"
        fi
    done
}

# Clean user cache directories
clean_user_cache() {
    print_header "Cleaning User Cache (Optional)"
    
    local cache_dirs=(
        "$HOME/.rustg"
        "$HOME/.cache/rustg"
    )
    
    for cache_dir in "${cache_dirs[@]}"; do
        if [[ -d "$cache_dir" ]]; then
            print_warning "Found RustG cache directory: $cache_dir"
            read -p "Remove cache directory? [y/N]: " -n 1 -r
            echo
            
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                rm -rf "$cache_dir"
                print_success "Cache directory removed"
            else
                print_info "Cache directory preserved"
            fi
        fi
    done
}

# Verify uninstallation
verify_removal() {
    print_header "Verifying Removal"
    
    local tools=("cargo-g" "clippy-f" "rustfmt-g" "rustdoc-g" "rustup-g" "rust-analyzer-g" "rust-gdb-g" "bindgen-g" "miri-g")
    local remaining_tools=()
    
    for tool in "${tools[@]}"; do
        if command -v "$tool" >/dev/null 2>&1; then
            remaining_tools+=("$tool")
        fi
    done
    
    if [[ ${#remaining_tools[@]} -eq 0 ]]; then
        print_success "All RustG tools successfully removed"
    else
        print_warning "Some tools may still be accessible: ${remaining_tools[*]}"
        print_info "These might be from other installations or PATH entries"
    fi
}

# Main uninstallation function
main() {
    print_banner
    
    print_warning "This will remove RustG GPU-accelerated tools from your system"
    echo ""
    read -p "Are you sure you want to continue? [y/N]: " -n 1 -r
    echo ""
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Uninstallation cancelled"
        exit 0
    fi
    
    check_permissions
    remove_binaries
    remove_environment
    clean_user_cache
    verify_removal
    
    print_header "Uninstallation Complete!"
    
    echo -e "${GREEN}${BOLD}RustG has been removed from your system.${NC}"
    echo ""
    echo "To complete the removal:"
    echo "  • Restart your shell or terminal"
    echo "  • Remove any manual PATH modifications"
    echo "  • Clear any remaining environment variables"
    echo ""
    echo "Standard Rust tools (cargo, clippy, rustfmt, etc.) remain unchanged."
    echo ""
    echo "Thank you for using RustG! 🚀"
}

# Execute main function
main "$@"