#ifndef NSIGHT_PROFILING_H
#define NSIGHT_PROFILING_H

#include <cuda_runtime.h>
#include <cupti.h>
#include <nvtx3/nvtx3.hpp>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Nsight Systems Integration for CUDA 13.0
// Replaces deprecated nvprof with modern profiling APIs
// ============================================================================

// Initialize Nsight Systems profiling
int nsight_init();
void nsight_cleanup();

// ============================================================================
// NVTX Range Annotations (Nsight Systems)
// ============================================================================

// Push/Pop ranges for marking code sections
void nsight_range_push(const char* name);
void nsight_range_pop();

// Scoped range helper
#ifdef __cplusplus
class NsightRange {
public:
    NsightRange(const char* name) { nsight_range_push(name); }
    ~NsightRange() { nsight_range_pop(); }
};
#define NSIGHT_RANGE(name) NsightRange _nsight_range(name)
#endif

// Mark instantaneous events
void nsight_mark(const char* name);

// Thread naming for better visualization
void nsight_name_thread(uint32_t tid, const char* name);

// ============================================================================
// CUPTI Range Profiling API (CUDA 13.0)
// Replaces deprecated Event/Metric APIs
// ============================================================================

typedef struct {
    CUpti_ProfilerRange range;
    CUpti_ProfilerReplayMode replay_mode;
    size_t range_data_size;
    void* range_data;
} CuptiRangeConfig;

// Initialize CUPTI Range Profiling
int cupti_range_init(CuptiRangeConfig* config);
void cupti_range_cleanup(CuptiRangeConfig* config);

// Start/Stop profiling ranges
int cupti_range_start(CuptiRangeConfig* config, const char* range_name);
int cupti_range_stop(CuptiRangeConfig* config);

// Collect range metrics
typedef struct {
    const char* metric_name;
    double value;
    const char* unit;
} RangeMetric;

int cupti_range_get_metrics(
    CuptiRangeConfig* config,
    RangeMetric** metrics,
    size_t* num_metrics
);

// ============================================================================
// RTX 5090 Blackwell-Specific Metrics
// ============================================================================

typedef struct {
    // Compute metrics
    float sm_efficiency;
    float tensor_core_utilization;
    float fp8_throughput_tflops;
    float fp16_throughput_tflops;
    float fp32_throughput_tflops;
    
    // Memory metrics
    float memory_bandwidth_gbps;
    float l2_cache_hit_rate;
    float memory_utilization;
    
    // Thread block cluster metrics (Blackwell)
    float cluster_efficiency;
    float distributed_shared_memory_usage;
    uint32_t active_clusters;
    
    // Power/Thermal
    float power_usage_watts;
    float gpu_temperature_c;
    float memory_temperature_c;
} BlackwellMetrics;

// Collect RTX 5090 specific metrics
int collect_blackwell_metrics(BlackwellMetrics* metrics);

// ============================================================================
// Kernel Performance Profiling
// ============================================================================

typedef struct {
    const char* kernel_name;
    uint64_t start_timestamp;
    uint64_t end_timestamp;
    uint64_t duration_ns;
    
    // Grid/Block configuration
    dim3 grid_size;
    dim3 block_size;
    size_t shared_memory_bytes;
    uint32_t registers_per_thread;
    
    // Performance counters
    uint64_t instructions_executed;
    uint64_t memory_transactions;
    uint64_t branch_divergence;
    float occupancy;
    
    // Memory metrics
    uint64_t global_load_transactions;
    uint64_t global_store_transactions;
    uint64_t shared_load_transactions;
    uint64_t shared_store_transactions;
    
    // Blackwell specific
    uint32_t cluster_dims[3];
    float cluster_utilization;
} KernelProfile;

// Profile kernel execution
int profile_kernel_launch(
    const char* kernel_name,
    dim3 grid,
    dim3 block,
    size_t shared_mem,
    cudaStream_t stream,
    KernelProfile* profile
);

// ============================================================================
// Memory Profiling
// ============================================================================

typedef struct {
    // Allocation metrics
    uint64_t total_allocated;
    uint64_t peak_allocated;
    uint32_t num_allocations;
    uint32_t num_deallocations;
    
    // Async allocation metrics (CUDA 13.0)
    uint64_t async_allocations;
    uint64_t pool_memory_used;
    uint64_t pool_memory_reserved;
    float pool_utilization;
    
    // Transfer metrics
    uint64_t host_to_device_bytes;
    uint64_t device_to_host_bytes;
    uint64_t device_to_device_bytes;
    float transfer_bandwidth_gbps;
    
    // RTX 5090 32GB memory
    uint64_t total_device_memory;
    uint64_t available_device_memory;
    float memory_fragmentation;
} MemoryProfile;

// Collect memory profiling data
int profile_memory_usage(MemoryProfile* profile);

// ============================================================================
// Compile Time Advisor Integration (ctadvisor)
// ============================================================================

typedef struct {
    const char* file_name;
    double compilation_time_ms;
    size_t ptx_size_bytes;
    size_t sass_size_bytes;
    uint32_t num_kernels;
    uint32_t num_device_functions;
    const char* bottleneck;
    const char* suggestion;
} CompileProfile;

// Profile CUDA compilation
int profile_cuda_compilation(
    const char* source_file,
    CompileProfile* profile
);

// Get compilation optimization suggestions
const char* get_compilation_suggestions(CompileProfile* profile);

// ============================================================================
// Nsight Compute Integration
// ============================================================================

// Launch Nsight Compute profiling session
int nsight_compute_start_session(const char* output_file);
int nsight_compute_stop_session();

// Add kernel replay for detailed analysis
int nsight_compute_replay_kernel(
    const char* kernel_name,
    uint32_t num_replays
);

// ============================================================================
// Export & Reporting
// ============================================================================

// Export formats
typedef enum {
    EXPORT_JSON,
    EXPORT_CSV,
    EXPORT_SQLITE,
    EXPORT_NSYS_QDREP,  // Nsight Systems format
    EXPORT_NCU_REPORT   // Nsight Compute format
} ExportFormat;

// Export profiling data
int export_profile_data(
    const char* output_path,
    ExportFormat format,
    KernelProfile* kernels,
    size_t num_kernels,
    MemoryProfile* memory,
    BlackwellMetrics* metrics
);

// Generate HTML report
int generate_profile_report(
    const char* output_path,
    const char* report_title
);

// ============================================================================
// Utility Functions
// ============================================================================

// Check if running under profiler
int is_profiling_enabled();

// Set profiling verbosity
void set_profiling_level(int level);

// Flush profiling buffers
void flush_profiling_data();

// Get last profiling error
const char* get_profiling_error();

#ifdef __cplusplus
}
#endif

#endif // NSIGHT_PROFILING_H