#ifndef CUDA13_FEATURES_H
#define CUDA13_FEATURES_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cusparse.h>
#include <nvrtc.h>
#include <nvJitLink.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// CUDA 13.0 Memory Pool Features
// ============================================================================

// Async memory allocation with stream-ordered semantics
void* cuda_malloc_async(size_t size, cudaStream_t stream);
int cuda_free_async(void* ptr, cudaStream_t stream);

// Host-pinned memory with CU_MEM_LOCATION_TYPE_HOST support
void* cuda_malloc_host_async(size_t size);
int cuda_free_host(void* ptr);

// Async memory copy operations
int cuda_memcpy_host_to_device_async(void* dst, const void* src, size_t size, cudaStream_t stream);
int cuda_memcpy_device_to_host_async(void* dst, const void* src, size_t size, cudaStream_t stream);

// ============================================================================
// cuBLAS 13.0 Autotune Features
// ============================================================================

// Enable GEMM autotune for optimal performance on RTX 5090
typedef struct {
    cublasHandle_t handle;
    cublasLtHandle_t lt_handle;
    int autotune_enabled;
    int cache_enabled;
} CublasContext;

// Initialize cuBLAS with autotune support
CublasContext* cublas_create_with_autotune();
void cublas_destroy_context(CublasContext* ctx);

// Set autotune mode (CUBLAS_GEMM_AUTOTUNE environment variable)
void cublas_set_autotune(CublasContext* ctx, int enable);

// Perform autotuned GEMM operation
int cublas_gemm_autotuned(
    CublasContext* ctx,
    int m, int n, int k,
    const float* alpha,
    const float* A, int lda,
    const float* B, int ldb,
    const float* beta,
    float* C, int ldc
);

// ============================================================================
// cuSPARSE 13.0 64-bit Index Support
// ============================================================================

// 64-bit sparse matrix operations for large graphs
typedef struct {
    cusparseHandle_t handle;
    int use_64bit_indices;
} CusparseContext;

// Create cuSPARSE context with 64-bit index support
CusparseContext* cusparse_create_64bit();
void cusparse_destroy_context(CusparseContext* ctx);

// SpGEMM with 64-bit indices for large matrices
int cusparse_spgemm_64bit(
    CusparseContext* ctx,
    int64_t m, int64_t n, int64_t k,
    int64_t nnzA, const int64_t* csrRowPtrA, const int64_t* csrColIndA, const float* csrValA,
    int64_t nnzB, const int64_t* csrRowPtrB, const int64_t* csrColIndB, const float* csrValB,
    int64_t* nnzC, int64_t** csrRowPtrC, int64_t** csrColIndC, float** csrValC
);

// ============================================================================
// nvJitLink 13.0 Features
// ============================================================================

// JIT compilation context for dynamic kernel generation
typedef struct {
    nvrtcProgram prog;
    nvJitLinkHandle link_handle;
    char* ptx_code;
    size_t ptx_size;
} JitContext;

// Create JIT compilation context
JitContext* jit_create_context(const char* kernel_name);
void jit_destroy_context(JitContext* ctx);

// Compile CUDA source to PTX with nvrtc
int jit_compile_kernel(
    JitContext* ctx,
    const char* cuda_source,
    const char** headers,
    int num_headers,
    const char** compile_options,
    int num_options
);

// Link PTX modules with nvJitLink (CUDA 13.0)
int jit_link_modules(
    JitContext* ctx,
    void** modules,
    size_t* module_sizes,
    int num_modules,
    const char** link_options,
    int num_options
);

// Load and execute JIT-compiled kernel
int jit_launch_kernel(
    JitContext* ctx,
    const char* kernel_name,
    dim3 grid,
    dim3 block,
    void** args,
    size_t shared_mem,
    cudaStream_t stream
);

// ============================================================================
// RTX 5090 Blackwell-Specific Features
// ============================================================================

// Check if running on Blackwell GPU
int is_blackwell_gpu();

// Get RTX 5090 specific capabilities
typedef struct {
    int compute_capability;  // Should be 110 for RTX 5090
    size_t total_memory;      // 32GB for RTX 5090
    int tensor_core_version;  // 5th gen Tensor Cores
    int fp8_support;          // FP8 training support
    float memory_bandwidth;   // 1.5+ TB/s
    int max_threads_per_sm;   // Blackwell threading model
} BlackwellCapabilities;

BlackwellCapabilities get_blackwell_capabilities();

// Enable Blackwell-specific optimizations
void enable_blackwell_optimizations();

// ============================================================================
// Compile Time Advisor Integration
// ============================================================================

// Profile CUDA compilation time
void ctadvisor_start_profiling();
void ctadvisor_stop_profiling();
void ctadvisor_print_report();

#ifdef __cplusplus
}
#endif

#endif // CUDA13_FEATURES_H