# rustg GPU Compiler Makefile
# Orchestrates hybrid Rust/CUDA build

# Configuration
CARGO := cargo
CMAKE := cmake
MAKE := make
NVCC := nvcc
CUDA_MEMCHECK := cuda-memcheck
NVPROF := nvprof
NSIGHT := nsight-compute

# Directories
BUILD_DIR := target
CUDA_BUILD_DIR := $(BUILD_DIR)/cuda_build
BENCHMARK_DIR := benchmarks
TEST_DIR := tests

# Build profiles
PROFILE ?= release
ifeq ($(PROFILE),debug)
	CARGO_FLAGS := 
	CMAKE_BUILD_TYPE := Debug
else
	CARGO_FLAGS := --release
	CMAKE_BUILD_TYPE := Release
endif

# Number of parallel jobs
JOBS := $(shell nproc 2>/dev/null || echo 4)

# Default target
.PHONY: all
all: build

# Build the complete project (Rust + CUDA)
.PHONY: build
build: cuda-build
	@echo "Building rustg with cargo..."
	$(CARGO) build $(CARGO_FLAGS)
	@echo "Build complete!"

# Build only CUDA components
.PHONY: cuda-build
cuda-build: $(CUDA_BUILD_DIR)/Makefile
	@echo "Building CUDA kernels..."
	$(MAKE) -C $(CUDA_BUILD_DIR) -j$(JOBS)

# Configure CUDA build with CMake
$(CUDA_BUILD_DIR)/Makefile: CMakeLists.txt
	@echo "Configuring CUDA build with CMake..."
	@mkdir -p $(CUDA_BUILD_DIR)
	cd $(CUDA_BUILD_DIR) && $(CMAKE) ../.. \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Run all tests
.PHONY: test
test: unit-test integration-test gpu-test

# Run unit tests
.PHONY: unit-test
unit-test: build
	@echo "Running unit tests..."
	$(CARGO) test $(CARGO_FLAGS) --lib

# Run integration tests
.PHONY: integration-test
integration-test: build
	@echo "Running integration tests..."
	$(CARGO) test $(CARGO_FLAGS) --test '*'

# Run GPU kernel tests
.PHONY: gpu-test
gpu-test: cuda-build
	@echo "Running GPU kernel tests..."
	@if [ -f $(CUDA_BUILD_DIR)/kernel_tests ]; then \
		$(CUDA_BUILD_DIR)/kernel_tests; \
	else \
		echo "GPU kernel tests not built yet"; \
	fi

# Run benchmarks
.PHONY: bench
bench: build
	@echo "Running benchmarks..."
	$(CARGO) bench $(CARGO_FLAGS)
	@if [ -f $(CUDA_BUILD_DIR)/benchmarks ]; then \
		$(CUDA_BUILD_DIR)/benchmarks; \
	fi

# Memory checking with cuda-memcheck
.PHONY: memcheck
memcheck: gpu-test
	@echo "Running cuda-memcheck..."
	@if [ -f $(CUDA_BUILD_DIR)/kernel_tests ]; then \
		$(CUDA_MEMCHECK) --tool memcheck $(CUDA_BUILD_DIR)/kernel_tests; \
		$(CUDA_MEMCHECK) --tool racecheck $(CUDA_BUILD_DIR)/kernel_tests; \
		$(CUDA_MEMCHECK) --tool synccheck $(CUDA_BUILD_DIR)/kernel_tests; \
	else \
		echo "GPU kernel tests not built yet"; \
	fi

# Performance profiling with nvprof
.PHONY: profile
profile: build
	@echo "Profiling with nvprof..."
	@if [ -f $(CUDA_BUILD_DIR)/benchmarks ]; then \
		$(NVPROF) --print-gpu-trace $(CUDA_BUILD_DIR)/benchmarks; \
	else \
		echo "Benchmarks not built yet"; \
	fi

# Performance profiling with Nsight Compute
.PHONY: nsight
nsight: build
	@echo "Profiling with Nsight Compute..."
	@if [ -f $(CUDA_BUILD_DIR)/benchmarks ]; then \
		$(NSIGHT) --set full $(CUDA_BUILD_DIR)/benchmarks; \
	else \
		echo "Benchmarks not built yet"; \
	fi

# Format code
.PHONY: format
format:
	@echo "Formatting Rust code..."
	$(CARGO) fmt
	@echo "Formatting CUDA code..."
	@find src -name "*.cu" -o -name "*.cuh" | xargs clang-format -i
	@find include -name "*.h" -o -name "*.hpp" | xargs clang-format -i

# Lint code
.PHONY: lint
lint:
	@echo "Linting Rust code..."
	$(CARGO) clippy $(CARGO_FLAGS) -- -D warnings
	@echo "Checking CUDA code..."
	@if command -v clang-tidy > /dev/null; then \
		find src -name "*.cu" | xargs clang-tidy; \
	fi

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	$(CARGO) clean
	rm -rf $(CUDA_BUILD_DIR)
	rm -rf $(BUILD_DIR)

# Deep clean (including dependencies)
.PHONY: distclean
distclean: clean
	@echo "Deep cleaning..."
	rm -rf Cargo.lock
	rm -rf target

# Documentation
.PHONY: doc
doc:
	@echo "Building documentation..."
	$(CARGO) doc --no-deps --open

# Check CUDA installation
.PHONY: check-cuda
check-cuda:
	@echo "Checking CUDA installation..."
	@which nvcc > /dev/null && echo "✓ nvcc found: $$(nvcc --version | head -n 1)" || echo "✗ nvcc not found"
	@which $(CUDA_MEMCHECK) > /dev/null && echo "✓ cuda-memcheck found" || echo "✗ cuda-memcheck not found"
	@which $(NVPROF) > /dev/null && echo "✓ nvprof found" || echo "✗ nvprof not found"
	@echo "CUDA device info:"
	@nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || echo "nvidia-smi not available"

# Development setup
.PHONY: setup
setup:
	@echo "Setting up development environment..."
	rustup component add rustfmt clippy
	$(CARGO) install cargo-watch cargo-expand cargo-criterion
	@echo "Checking CUDA..."
	@$(MAKE) check-cuda

# Watch for changes and rebuild
.PHONY: watch
watch:
	cargo watch -x build -x test

# Run the compiler
.PHONY: run
run: build
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make run FILE=path/to/source.rs"; \
	else \
		$(CARGO) run $(CARGO_FLAGS) -- $(FILE); \
	fi

# Help
.PHONY: help
help:
	@echo "rustg GPU Compiler Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build everything (default)"
	@echo "  build        - Build Rust and CUDA components"
	@echo "  cuda-build   - Build only CUDA kernels"
	@echo "  test         - Run all tests"
	@echo "  unit-test    - Run Rust unit tests"
	@echo "  gpu-test     - Run GPU kernel tests"
	@echo "  bench        - Run benchmarks"
	@echo "  memcheck     - Run cuda-memcheck on GPU tests"
	@echo "  profile      - Profile with nvprof"
	@echo "  nsight       - Profile with Nsight Compute"
	@echo "  format       - Format all code"
	@echo "  lint         - Lint all code"
	@echo "  clean        - Clean build artifacts"
	@echo "  distclean    - Deep clean"
	@echo "  doc          - Build and open documentation"
	@echo "  check-cuda   - Check CUDA installation"
	@echo "  setup        - Setup development environment"
	@echo "  watch        - Watch for changes and rebuild"
	@echo "  run FILE=... - Run the compiler on a file"
	@echo ""
	@echo "Variables:"
	@echo "  PROFILE=debug|release - Build profile (default: release)"
	@echo ""
	@echo "Examples:"
	@echo "  make build PROFILE=debug"
	@echo "  make test"
	@echo "  make run FILE=examples/hello.rs"

.DEFAULT_GOAL := all