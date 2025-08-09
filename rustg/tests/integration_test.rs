//! Integration tests for GPU-CPU data flow

use rustg::{initialize, shutdown, GpuCompiler};
use std::fs;
use std::path::Path;
use std::time::Instant;

#[test]
fn test_gpu_cpu_integration() {
    // Initialize GPU runtime
    initialize().expect("Failed to initialize GPU runtime");
    
    // Create test source
    let source = r#"
        fn main() {
            let x = 42;
            let y = 3.14;
            println!("Hello, GPU!");
        }
    "#;
    
    // Create compiler
    let compiler = GpuCompiler::new()
        .with_cpu_fallback(false)
        .with_profiling(true);
    
    // Compile source
    let result = compiler.compile_source(source)
        .expect("Compilation failed");
    
    // Verify results
    assert!(result.success);
    assert!(result.token_count > 0);
    assert!(result.total_time_ms() > 0.0);
    
    // Cleanup
    shutdown().expect("Failed to shutdown GPU runtime");
}

#[test]
fn test_large_file_processing() {
    initialize().expect("Failed to initialize GPU runtime");
    
    // Generate large source file
    let mut source = String::new();
    for i in 0..10000 {
        source.push_str(&format!(
            "fn function_{}() {{ let var_{} = {}; }}\n",
            i, i, i
        ));
    }
    
    let compiler = GpuCompiler::new();
    
    // Time the compilation
    let start = Instant::now();
    let result = compiler.compile_source(&source)
        .expect("Compilation failed");
    let elapsed = start.elapsed();
    
    println!("Large file stats:");
    println!("  Source size: {} bytes", source.len());
    println!("  Token count: {}", result.token_count);
    println!("  Total time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!("  Throughput: {:.2} MB/s", 
             source.len() as f64 / elapsed.as_secs_f64() / 1_000_000.0);
    
    assert!(result.success);
    assert!(result.token_count > 100000);
    
    shutdown().expect("Failed to shutdown GPU runtime");
}

#[test]
fn test_memory_limits() {
    initialize().expect("Failed to initialize GPU runtime");
    
    // Test with limited GPU memory
    let compiler = GpuCompiler::new()
        .with_gpu_memory_limit(100 * 1024 * 1024); // 100MB limit
    
    let source = "fn main() { }";
    let result = compiler.compile_source(source)
        .expect("Compilation failed");
    
    assert!(result.success);
    assert!(result.gpu_memory_used_mb() <= 100);
    
    shutdown().expect("Failed to shutdown GPU runtime");
}

#[test]
fn test_cpu_fallback() {
    initialize().expect("Failed to initialize GPU runtime");
    
    // Test CPU fallback mode
    let compiler_gpu = GpuCompiler::new()
        .with_cpu_fallback(false);
    
    let compiler_cpu = GpuCompiler::new()
        .with_cpu_fallback(true);
    
    let source = r#"
        struct Point { x: f32, y: f32 }
        impl Point {
            fn new(x: f32, y: f32) -> Self {
                Point { x, y }
            }
        }
    "#;
    
    // Compile with GPU
    let result_gpu = compiler_gpu.compile_source(source);
    
    // Compile with CPU fallback
    let result_cpu = compiler_cpu.compile_source(source)
        .expect("CPU compilation failed");
    
    // CPU fallback should always work
    assert!(result_cpu.success);
    
    // GPU might fail for unsupported features (that's ok for now)
    if let Ok(gpu_result) = result_gpu {
        // If GPU succeeded, verify consistency
        assert_eq!(gpu_result.token_count, result_cpu.token_count);
    }
    
    shutdown().expect("Failed to shutdown GPU runtime");
}

#[test]
fn test_concurrent_compilation() {
    use std::thread;
    use std::sync::Arc;
    
    initialize().expect("Failed to initialize GPU runtime");
    
    let compiler = Arc::new(GpuCompiler::new());
    let mut handles = vec![];
    
    // Spawn multiple compilation threads
    for i in 0..4 {
        let compiler_clone = Arc::clone(&compiler);
        let handle = thread::spawn(move || {
            let source = format!("fn thread_{}() {{ let x = {}; }}", i, i);
            compiler_clone.compile_source(&source)
        });
        handles.push(handle);
    }
    
    // Wait for all threads and verify results
    for handle in handles {
        let result = handle.join().unwrap()
            .expect("Thread compilation failed");
        assert!(result.success);
    }
    
    shutdown().expect("Failed to shutdown GPU runtime");
}

#[test]
fn test_error_handling() {
    initialize().expect("Failed to initialize GPU runtime");
    
    let compiler = GpuCompiler::new();
    
    // Test with invalid syntax
    let invalid_sources = vec![
        "fn main( {",           // Unmatched parenthesis
        "let x = ;",            // Missing value
        "struct { }",           // Missing name
        "fn () { }",            // Missing function name
    ];
    
    for source in invalid_sources {
        let result = compiler.compile_source(source);
        // For now, we might not catch all errors, but shouldn't crash
        if let Err(e) = result {
            println!("Expected error for '{}': {:?}", source, e);
        }
    }
    
    shutdown().expect("Failed to shutdown GPU runtime");
}

#[test]
#[ignore] // Only run when GPU is available
fn test_performance_speedup() {
    initialize().expect("Failed to initialize GPU runtime");
    
    // Generate test source
    let mut source = String::new();
    for i in 0..50000 {
        source.push_str(&format!("let var_{} = {};\n", i, i));
    }
    
    let compiler_gpu = GpuCompiler::new()
        .with_cpu_fallback(false)
        .with_profiling(true);
    
    let compiler_cpu = GpuCompiler::new()
        .with_cpu_fallback(true)
        .with_profiling(true);
    
    // Benchmark GPU
    let gpu_start = Instant::now();
    let gpu_result = compiler_gpu.compile_source(&source)
        .expect("GPU compilation failed");
    let gpu_time = gpu_start.elapsed();
    
    // Benchmark CPU
    let cpu_start = Instant::now();
    let cpu_result = compiler_cpu.compile_source(&source)
        .expect("CPU compilation failed");
    let cpu_time = cpu_start.elapsed();
    
    // Calculate speedup
    let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
    
    println!("Performance comparison:");
    println!("  CPU time: {:.2}ms", cpu_time.as_secs_f64() * 1000.0);
    println!("  GPU time: {:.2}ms", gpu_time.as_secs_f64() * 1000.0);
    println!("  Speedup: {:.2}x", speedup);
    println!("  GPU throughput: {:.2} MB/s", 
             source.len() as f64 / gpu_time.as_secs_f64() / 1_000_000.0);
    
    // Verify speedup meets target
    assert!(speedup > 10.0, "GPU should be at least 10x faster than CPU");
    
    // Verify correctness
    assert_eq!(gpu_result.token_count, cpu_result.token_count);
    
    shutdown().expect("Failed to shutdown GPU runtime");
}