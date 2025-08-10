/**
 * CUDA Safe API Wrapper
 * Provides exception-safe C ABI for CUDA operations
 * Never allows C++ exceptions to cross FFI boundary
 */

#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error codes for CUDA operations
typedef enum {
    RB_OK = 0,
    RB_ERR_NOT_INITIALIZED = 1,
    RB_ERR_CUDA = 2,
    RB_ERR_THRUST = 3,
    RB_ERR_INVALID_ARG = 4,
    RB_ERR_OOM = 5,
    RB_ERR_KERNEL_LAUNCH = 6,
    RB_ERR_DEVICE_NOT_FOUND = 7,
    RB_ERR_UNKNOWN = 255
} rb_status_t;

// Result structure for returning status and timing info
typedef struct {
    int code;           // rb_status_t
    char msg[256];      // Error message (truncated if needed)
    double millis;      // Timing information (optional)
    size_t value;       // Optional return value
} rb_result_t;

// Safe initialization functions
rb_status_t rb_cuda_init(rb_result_t* out) noexcept;
rb_status_t rb_cuda_device_count(rb_result_t* out) noexcept;
rb_status_t rb_cuda_reset(rb_result_t* out) noexcept;

// Safe dataframe test functions
rb_status_t rb_test_dataframe_columnar_scan(
    rb_result_t* out,
    size_t num_rows
) noexcept;

rb_status_t rb_test_dataframe_hash_join(
    rb_result_t* out,
    size_t left_size,
    size_t right_size
) noexcept;

rb_status_t rb_test_dataframe_performance_comprehensive(
    rb_result_t* out
) noexcept;

// Safe graph test functions
rb_status_t rb_test_graph_bfs_performance(
    rb_result_t* out,
    uint32_t num_vertices,
    uint32_t num_edges
) noexcept;

rb_status_t rb_test_graph_pagerank_performance(
    rb_result_t* out,
    uint32_t num_vertices,
    uint32_t num_edges
) noexcept;

rb_status_t rb_test_graph_performance_comprehensive(
    rb_result_t* out
) noexcept;

// Safe search test functions
rb_status_t rb_test_search_boolean_performance(
    rb_result_t* out,
    uint32_t num_documents,
    uint32_t num_queries
) noexcept;

rb_status_t rb_test_search_performance_comprehensive(
    rb_result_t* out
) noexcept;

// Safe SQL test functions
rb_status_t rb_test_sql_table_scan_performance(
    rb_result_t* out,
    uint64_t num_rows,
    uint32_t num_columns
) noexcept;

rb_status_t rb_test_sql_performance_comprehensive(
    rb_result_t* out
) noexcept;

#ifdef __cplusplus
}
#endif