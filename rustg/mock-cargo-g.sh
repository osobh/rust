#!/bin/bash
# Mock cargo-g rustc wrapper for integration testing
# This demonstrates the cargo-g integration concept

# If called as rustc wrapper, pass through to rustc
if [[ "$1" == *"rustc"* ]] || [[ "$1" == "-vV" ]]; then
    exec "$@"
fi

# Otherwise, it's being used as cargo-g
echo "🔥 cargo-g: GPU-Accelerated Cargo Build System v0.1.0 (MOCK MODE)"
echo "   Detecting GPU-intensive crates..."
echo "   Using rustg for GPU compilation where applicable..."
echo ""

# Pass through to regular cargo for now
exec cargo "$@"