# RustG GPU Compiler Integration Guide

## Overview
This guide covers integrating RustG GPU compiler tools into existing development workflows, CI/CD pipelines, and build systems.

## IDE Integration

### Visual Studio Code
1. **Extension Configuration**:
```json
// .vscode/settings.json
{
  "rust-analyzer.cargo.buildScripts.enable": true,
  "rust-analyzer.checkOnSave.command": "clippy-f",
  "rust-analyzer.cargo.runner": "cargo-g",
  "rust-analyzer.runnables.command": "cargo-g"
}
```

2. **Task Configuration**:
```json
// .vscode/tasks.json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "cargo-g build",
      "type": "shell",
      "command": "cargo-g",
      "args": ["build", "--release"],
      "group": "build"
    },
    {
      "label": "clippy-f lint",
      "type": "shell", 
      "command": "clippy-f",
      "args": ["--workspace", "--output-format", "json"],
      "group": "test"
    }
  ]
}
```

### JetBrains IDEs (CLion, IntelliJ IDEA)
1. **Build Configuration**:
   - File → Settings → Build → Rust
   - Set Build Tool: `cargo-g`
   - Set Clippy: `clippy-f`

2. **External Tools**:
```xml
<!-- Settings → Tools → External Tools -->
<tool name="cargo-g build" 
      program="cargo-g" 
      arguments="build --release"
      workingDirectory="$ProjectFileDir$" />

<tool name="clippy-f analyze"
      program="clippy-f"
      arguments="--workspace --gpu-analysis"
      workingDirectory="$ProjectFileDir$" />
```

### Neovim with LSP
```lua
-- init.lua or lsp config
local lspconfig = require('lspconfig')

lspconfig.rust_analyzer.setup({
  settings = {
    ['rust-analyzer'] = {
      cargo = {
        buildScripts = { enable = true },
        runner = "cargo-g",
      },
      checkOnSave = {
        command = "clippy-f",
        extraArgs = { "--gpu-analysis" }
      }
    }
  }
})
```

## CI/CD Integration

### GitHub Actions
```yaml
name: RustG GPU Build

on: [push, pull_request]

jobs:
  build-gpu:
    runs-on: [self-hosted, gpu, cuda]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Install RustG
      run: |
        wget https://releases.com/rustg-gpu-compiler-v0.1.0-linux-x64.tar.gz
        tar -xzf rustg-gpu-compiler-v0.1.0-linux-x64.tar.gz
        cd rustg-gpu-compiler-v0.1.0-linux-x64
        ./install.sh
        echo "$HOME/.cargo/bin" >> $GITHUB_PATH
    
    - name: Validate GPU Setup
      run: |
        nvidia-smi
        cargo-g --version
        clippy-f --version
    
    - name: Build with GPU Acceleration
      run: |
        cargo-g build --workspace --release
        
    - name: Test with GPU Acceleration  
      run: |
        cargo-g test --workspace
        
    - name: GPU Linting Analysis
      run: |
        clippy-f --workspace --gpu-analysis --output-format json > clippy-report.json
        
    - name: Upload Clippy Report
      uses: actions/upload-artifact@v4
      with:
        name: clippy-gpu-report
        path: clippy-report.json
```

### GitLab CI
```yaml
# .gitlab-ci.yml
stages:
  - build
  - test
  - analyze

variables:
  RUSTG_GPU_ENABLED: "1"
  CUDA_DEVICE_ID: "0"

gpu-build:
  stage: build
  tags:
    - gpu
    - cuda-13
  before_script:
    - curl -L https://releases.com/rustg-gpu-compiler-v0.1.0-linux-x64.tar.gz | tar -xz
    - cd rustg-gpu-compiler-v0.1.0-linux-x64 && ./install.sh && cd ..
    - export PATH="$HOME/.cargo/bin:$PATH"
  script:
    - cargo-g build --workspace --release
  artifacts:
    paths:
      - target/release/
    expire_in: 1 hour

gpu-test:
  stage: test  
  tags:
    - gpu
    - cuda-13
  dependencies:
    - gpu-build
  script:
    - export PATH="$HOME/.cargo/bin:$PATH"
    - cargo-g test --workspace

gpu-analyze:
  stage: analyze
  tags:
    - gpu  
    - cuda-13
  script:
    - export PATH="$HOME/.cargo/bin:$PATH"
    - clippy-f --workspace --gpu-analysis --output-format json > clippy-report.json
  artifacts:
    reports:
      junit: clippy-report.json
```

### Jenkins Pipeline
```groovy
pipeline {
    agent {
        label 'gpu-node'
    }
    
    environment {
        RUSTG_GPU_ENABLED = '1'
        PATH = "${env.HOME}/.cargo/bin:${env.PATH}"
    }
    
    stages {
        stage('Setup RustG') {
            steps {
                sh '''
                    wget https://releases.com/rustg-gpu-compiler-v0.1.0-linux-x64.tar.gz
                    tar -xzf rustg-gpu-compiler-v0.1.0-linux-x64.tar.gz
                    cd rustg-gpu-compiler-v0.1.0-linux-x64
                    ./install.sh
                '''
            }
        }
        
        stage('GPU Validation') {
            steps {
                sh '''
                    nvidia-smi
                    cargo-g --version
                    clippy-f --version
                '''
            }
        }
        
        stage('Build') {
            steps {
                sh 'cargo-g build --workspace --release'
            }
        }
        
        stage('Test') {
            steps {
                sh 'cargo-g test --workspace'
            }
        }
        
        stage('GPU Analysis') {
            steps {
                sh '''
                    clippy-f --workspace --gpu-analysis \
                        --output-format json > clippy-report.json
                '''
            }
            post {
                always {
                    archiveArtifacts artifacts: 'clippy-report.json'
                }
            }
        }
    }
}
```

## Build System Integration

### Cargo.toml Configuration
```toml
[package]
name = "my-gpu-project"
version = "0.1.0"
edition = "2021"

# Optimize for GPU compilation
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"

# GPU-specific features
[features]
default = ["gpu"]
gpu = []
cpu-fallback = []

# Build script for GPU detection
[build-dependencies] 
cc = "1.0"

# RustG-specific metadata
[package.metadata.rustg]
gpu-target = "sm_110"  # Blackwell architecture
cuda-version = "13.0"
prefer-gpu = true
```

### Build Script Integration
```rust
// build.rs
fn main() {
    // Detect GPU capabilities
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=compute_cap")
        .arg("--format=csv,noheader")
        .output()
    {
        let compute_cap = String::from_utf8_lossy(&output.stdout);
        println!("cargo:rustc-env=GPU_COMPUTE_CAP={}", compute_cap.trim());
        
        // Enable GPU features if available
        if compute_cap.trim().parse::<f32>().unwrap_or(0.0) >= 6.0 {
            println!("cargo:rustc-cfg=feature=\"gpu\"");
        }
    }
    
    // Use cargo-g for CUDA compilation if available
    if std::process::Command::new("cargo-g").arg("--version").status().is_ok() {
        println!("cargo:rustc-env=RUSTG_AVAILABLE=1");
    }
}
```

### Makefile Integration
```makefile
# Makefile
RUSTG_AVAILABLE := $(shell command -v cargo-g 2> /dev/null)
CLIPPY_F_AVAILABLE := $(shell command -v clippy-f 2> /dev/null)

# Use GPU tools if available, fallback to standard tools
ifdef RUSTG_AVAILABLE
    CARGO := cargo-g
else
    CARGO := cargo
endif

ifdef CLIPPY_F_AVAILABLE
    CLIPPY := clippy-f
else  
    CLIPPY := cargo clippy
endif

.PHONY: build test lint clean

build:
	$(CARGO) build --release

test:
	$(CARGO) test --workspace

lint:
	$(CLIPPY) --workspace

gpu-lint:
	$(CLIPPY) --workspace --gpu-analysis

clean:
	$(CARGO) clean

install-rustg:
	@echo "Installing RustG GPU Compiler..."
	wget https://releases.com/rustg-gpu-compiler-v0.1.0-linux-x64.tar.gz
	tar -xzf rustg-gpu-compiler-v0.1.0-linux-x64.tar.gz
	cd rustg-gpu-compiler-v0.1.0-linux-x64 && ./install.sh
	@echo "RustG installed successfully"

validate-gpu:
	@echo "GPU Validation:"
	@nvidia-smi --query-gpu=name,compute_cap --format=csv
	@$(CARGO) --version
	@$(CLIPPY) --version
```

## Docker Integration

### Dockerfile
```dockerfile
# Multi-stage build with GPU support
FROM nvidia/cuda:13.0-devel-ubuntu22.04 AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install RustG GPU Compiler
COPY rustg-gpu-compiler-v0.1.0-linux-x64.tar.gz /tmp/
RUN cd /tmp && tar -xzf rustg-gpu-compiler-v0.1.0-linux-x64.tar.gz \
    && cd rustg-gpu-compiler-v0.1.0-linux-x64 \
    && ./install.sh

# Runtime stage
FROM nvidia/cuda:13.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy RustG binaries
COPY --from=builder /root/.cargo/bin/cargo-g /usr/local/bin/
COPY --from=builder /root/.cargo/bin/clippy-f /usr/local/bin/

# Validate GPU setup
RUN cargo-g --version && clippy-f --version

WORKDIR /workspace
```

### docker-compose.yml
```yaml
version: '3.8'

services:
  rustg-dev:
    build:
      context: .
      dockerfile: Dockerfile
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - RUSTG_GPU_ENABLED=1
    volumes:
      - ./src:/workspace/src:ro
      - ./Cargo.toml:/workspace/Cargo.toml:ro
      - build-cache:/workspace/target
    working_dir: /workspace
    command: |
      sh -c "
        cargo-g build --release &&
        cargo-g test --workspace &&
        clippy-f --workspace --gpu-analysis
      "

volumes:
  build-cache:
```

## Performance Monitoring

### GPU Utilization Tracking
```bash
#!/bin/bash
# monitor-gpu-build.sh

# Start GPU monitoring in background
nvidia-smi dmon -s pucvmet -d 1 > gpu-metrics.log &
MONITOR_PID=$!

# Run build with timing
time cargo-g build --workspace --release

# Stop monitoring
kill $MONITOR_PID

# Analyze results
echo "GPU Utilization Summary:"
awk '/^[0-9]/ {gpu+=$3; mem+=$4; count++} END {print "Avg GPU:", gpu/count"%, Avg Memory:", mem/count"%"}' gpu-metrics.log
```

### Build Performance Analysis
```bash
#!/bin/bash  
# benchmark-rustg.sh

echo "Benchmarking RustG vs Standard Cargo"

# Clean state
cargo clean

# Benchmark standard cargo
echo "Testing standard cargo..."
time cargo build --release > cargo-std.log 2>&1
CARGO_TIME=$?

# Clean state  
cargo clean

# Benchmark cargo-g
echo "Testing cargo-g..."
time cargo-g build --release > cargo-g.log 2>&1
CARGO_G_TIME=$?

# Compare results
echo "Results:"
echo "Standard cargo: $(grep 'real' cargo-std.log)"
echo "cargo-g:       $(grep 'real' cargo-g.log)"

# Calculate speedup
SPEEDUP=$(python3 -c "
import sys
std = float('$(grep 'real' cargo-std.log | cut -d'm' -f2 | cut -d's' -f1)')
gpu = float('$(grep 'real' cargo-g.log | cut -d'm' -f2 | cut -d's' -f1)')
print(f'{std/gpu:.2f}x speedup')
")
echo "Speedup: $SPEEDUP"
```

## Troubleshooting Integration Issues

### Common Problems
1. **PATH Issues**: Ensure `~/.cargo/bin` is in PATH
2. **CUDA Not Found**: Verify `nvcc` and `nvidia-smi` availability  
3. **Permission Errors**: Check file permissions and user access
4. **Version Conflicts**: Ensure compatible CUDA and driver versions

### Debugging Commands
```bash
# Check installation
which cargo-g clippy-f
cargo-g --version
clippy-f --version

# Validate GPU
nvidia-smi
nvcc --version

# Test functionality
cargo-g check
clippy-f --help

# Enable verbose logging
RUSTG_VERBOSE_GPU=1 cargo-g build
```

This integration guide ensures seamless adoption of RustG GPU compiler tools across development environments and CI/CD pipelines.