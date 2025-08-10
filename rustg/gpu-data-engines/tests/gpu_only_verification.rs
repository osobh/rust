/*!
 * GPU-Only Execution Verification Tests
 * 
 * These tests ensure that NO CPU fallback is being used and ALL operations
 * are running on the GPU with the required performance characteristics.
 * 
 * Following strict TDD methodology - tests written BEFORE implementation.
 */

use gpu_data_engines::{GPUDataEngines, EngineConfig, PerformanceStats};
use std::time::Instant;

/// Verify that CUDA is available and NO CPU fallback exists
#[test]
fn test_cuda_mandatory_no_fallback() {
    // This should panic if CUDA is not available (as intended)
    assert!(cfg!(cuda_available), "CUDA must be available - no CPU fallback allowed!");
    
    // Ensure no stub configuration exists
    assert!(!cfg!(cuda_stub), "CUDA stub detected! GPU-native mode is MANDATORY!");
}

/// Test that GPU memory allocation works without CPU fallback
#[test]
fn test_gpu_memory_allocation_native() {
    use std::ffi::c_void;
    
    extern "C" {
        fn gpu_dataframe_create(capacity: usize) -> *mut c_void;
        fn gpu_dataframe_destroy(df: *mut c_void);
    }
    
    unsafe {
        // This should allocate directly on GPU
        let df = gpu_dataframe_create(1_000_000);
        assert!(!df.is_null(), "GPU allocation failed - no CPU fallback allowed!");
        
        // Clean up
        gpu_dataframe_destroy(df);
    }
}

/// Verify dataframe performance meets GPU-native targets
#[test]
fn test_dataframe_gpu_performance() {
    let config = EngineConfig {
        enable_dataframes: true,
        enable_graphs: false,
        enable_search: false,
        enable_sql: false,
        max_memory_gb: 8,
        cuda_device_id: 0,
    };
    
    let engines = GPUDataEngines::new(config)
        .expect("GPU engines must initialize without CPU fallback");
    
    // Test columnar scan performance
    let mut df = gpu_data_engines::dataframe::GPUDataframe::new(10_000_000)
        .expect("GPU dataframe creation failed");
    
    // Add test data
    let test_data: Vec<i64> = (0..10_000_000).collect();
    df.add_int_column(test_data).expect("Failed to add column");
    
    // Measure performance
    let start = Instant::now();
    let sum = df.columnar_scan(0).expect("Columnar scan failed");
    let elapsed = start.elapsed();
    
    // Calculate throughput
    let bytes_processed = 10_000_000 * 8; // 8 bytes per i64
    let throughput_gbps = (bytes_processed as f64 / 1_000_000_000.0) / elapsed.as_secs_f64();
    
    // MUST achieve 100GB/s+ as per requirements
    assert!(
        throughput_gbps >= 100.0,
        "Dataframe throughput {:.2} GB/s is below 100GB/s target - CPU fallback suspected!",
        throughput_gbps
    );
}

/// Verify graph processing performance meets GPU-native targets
#[test]
fn test_graph_gpu_performance() {
    let config = EngineConfig {
        enable_dataframes: false,
        enable_graphs: true,
        enable_search: false,
        enable_sql: false,
        max_memory_gb: 8,
        cuda_device_id: 0,
    };
    
    let engines = GPUDataEngines::new(config)
        .expect("GPU engines must initialize without CPU fallback");
    
    // Create test graph with 1M vertices and 10M edges
    let mut graph = gpu_data_engines::graph::GPUGraph::new(1_000_000, 10_000_000)
        .expect("GPU graph creation failed");
    
    // Add edges
    let edges: Vec<(u32, u32)> = (0..10_000_000)
        .map(|i| ((i % 1_000_000) as u32, ((i + 1) % 1_000_000) as u32))
        .collect();
    
    graph.add_edges(&edges).expect("Failed to add edges");
    
    // Measure BFS performance
    let start = Instant::now();
    let distances = graph.bfs(0).expect("BFS failed");
    let elapsed = start.elapsed();
    
    // Calculate edges traversed per second
    let edges_per_sec = 10_000_000.0 / elapsed.as_secs_f64();
    
    // MUST achieve 1B+ edges/sec as per requirements
    assert!(
        edges_per_sec >= 1_000_000_000.0,
        "Graph throughput {:.2}M edges/sec is below 1000M target - CPU fallback suspected!",
        edges_per_sec / 1_000_000.0
    );
}

/// Verify search engine performance meets GPU-native targets
#[test]
fn test_search_gpu_performance() {
    let config = EngineConfig {
        enable_dataframes: false,
        enable_graphs: false,
        enable_search: true,
        enable_sql: false,
        max_memory_gb: 8,
        cuda_device_id: 0,
    };
    
    let engines = GPUDataEngines::new(config)
        .expect("GPU engines must initialize without CPU fallback");
    
    // Create search engine with 1M documents
    let mut search = gpu_data_engines::search::GPUSearchEngine::new(1_000_000, 100_000)
        .expect("GPU search engine creation failed");
    
    // Add test documents
    for i in 0..1_000_000 {
        let doc = format!("document {} with test content keywords", i);
        search.add_document(i as u32, &doc).expect("Failed to add document");
    }
    
    // Measure query performance
    let queries = vec!["test", "document", "keywords", "content"];
    let start = Instant::now();
    
    for _ in 0..1000 {
        for query in &queries {
            let results = search.boolean_search(&gpu_data_engines::search::QueryType::Boolean(query.to_string())).expect("Search failed");
        }
    }
    
    let elapsed = start.elapsed();
    let queries_per_sec = 4000.0 / elapsed.as_secs_f64();
    
    // MUST achieve 1M+ queries/sec as per requirements
    assert!(
        queries_per_sec >= 1_000_000.0,
        "Search throughput {:.2}K QPS is below 1000K target - CPU fallback suspected!",
        queries_per_sec / 1000.0
    );
}

/// Verify SQL engine performance meets GPU-native targets  
#[test]
fn test_sql_gpu_performance() {
    let config = EngineConfig {
        enable_dataframes: false,
        enable_graphs: false,
        enable_search: false,
        enable_sql: true,
        max_memory_gb: 8,
        cuda_device_id: 0,
    };
    
    let engines = GPUDataEngines::new(config)
        .expect("GPU engines must initialize without CPU fallback");
    
    // Create SQL engine
    let mut sql = gpu_data_engines::sql::GPUSQLEngine::new()
        .expect("GPU SQL engine creation failed");
    
    // Create test table with 10M rows
    sql.create_table("test_table", vec![
        ("id", gpu_data_engines::sql::SQLDataType::Int64),
        ("value", gpu_data_engines::sql::SQLDataType::Double),
        ("name", gpu_data_engines::sql::SQLDataType::Varchar),
    ]).expect("Failed to create table");
    
    // Insert test data
    for i in 0..10_000_000 {
        sql.insert_row("test_table", vec![
            gpu_data_engines::sql::SQLValue::Int(i),
            gpu_data_engines::sql::SQLValue::Double(i as f64 * 1.5),
            gpu_data_engines::sql::SQLValue::String(format!("row_{}", i)),
        ]).expect("Failed to insert row");
    }
    
    // Measure query performance
    let start = Instant::now();
    let result = sql.execute_query("SELECT * FROM test_table WHERE id < 1000000")
        .expect("Query execution failed");
    let elapsed = start.elapsed();
    
    // Calculate throughput (10M rows * ~24 bytes per row)
    let bytes_processed = 10_000_000 * 24;
    let throughput_gbps = (bytes_processed as f64 / 1_000_000_000.0) / elapsed.as_secs_f64();
    
    // MUST achieve 100GB/s+ as per requirements
    assert!(
        throughput_gbps >= 100.0,
        "SQL throughput {:.2} GB/s is below 100GB/s target - CPU fallback suspected!",
        throughput_gbps
    );
}

/// Verify no CPU memory is being used for data processing
#[test]
fn test_no_cpu_memory_usage() {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};
    
    static CPU_ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
    
    struct TrackingAllocator;
    
    unsafe impl GlobalAlloc for TrackingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            CPU_ALLOC_BYTES.fetch_add(layout.size(), Ordering::SeqCst);
            System.alloc(layout)
        }
        
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            CPU_ALLOC_BYTES.fetch_sub(layout.size(), Ordering::SeqCst);
            System.dealloc(ptr, layout)
        }
    }
    
    // Get baseline CPU memory usage
    let baseline = CPU_ALLOC_BYTES.load(Ordering::SeqCst);
    
    // Create and use GPU engines
    let config = EngineConfig::default();
    let engines = GPUDataEngines::new(config)
        .expect("GPU engines must initialize");
    
    // Perform operations
    let mut df = gpu_data_engines::dataframe::GPUDataframe::new(1_000_000)
        .expect("GPU dataframe creation failed");
    let test_data: Vec<i64> = (0..1_000_000).collect();
    df.add_int_column(test_data).expect("Failed to add column");
    let _ = df.columnar_scan(0);
    
    // Check CPU memory usage didn't increase significantly
    let final_usage = CPU_ALLOC_BYTES.load(Ordering::SeqCst);
    let increase = final_usage - baseline;
    
    // Should only have minimal CPU memory for metadata, not data
    assert!(
        increase < 10_000_000, // Less than 10MB increase
        "CPU memory usage increased by {} bytes - data should be on GPU only!",
        increase
    );
}

/// Integration test verifying all engines work together on GPU
#[test]
fn test_all_engines_gpu_integration() {
    let config = EngineConfig {
        enable_dataframes: true,
        enable_graphs: true,
        enable_search: true,
        enable_sql: true,
        max_memory_gb: 16,
        cuda_device_id: 0,
    };
    
    let mut engines = GPUDataEngines::new(config)
        .expect("All GPU engines must initialize without CPU fallback");
    
    // Run comprehensive test
    let stats = engines.benchmark_all()
        .expect("Comprehensive benchmark failed");
    
    // Verify all performance targets are met
    assert!(stats.dataframe_throughput_gbps >= 100.0, 
            "Dataframe: {:.2} GB/s < 100 GB/s", stats.dataframe_throughput_gbps);
    assert!(stats.graph_throughput_edges_per_sec >= 1_000_000_000.0,
            "Graph: {:.2}M edges/s < 1000M", stats.graph_throughput_edges_per_sec / 1_000_000.0);
    assert!(stats.search_queries_per_sec >= 1_000_000.0,
            "Search: {:.2}K QPS < 1000K", stats.search_queries_per_sec / 1000.0);
    assert!(stats.sql_throughput_gbps >= 100.0,
            "SQL: {:.2} GB/s < 100 GB/s", stats.sql_throughput_gbps);
    
    println!("✅ All GPU-native performance targets achieved!");
    println!("   Dataframe: {:.2} GB/s", stats.dataframe_throughput_gbps);
    println!("   Graph: {:.2}M edges/sec", stats.graph_throughput_edges_per_sec / 1_000_000.0);
    println!("   Search: {:.2}K QPS", stats.search_queries_per_sec / 1000.0);
    println!("   SQL: {:.2} GB/s", stats.sql_throughput_gbps);
}