// Test harness for CUDA tests - NO MOCKS, REAL GPU EXECUTION

use std::ffi::CStr;
use std::os::raw::c_void;

// External CUDA test functions
extern "C" {
    fn run_gpu_detection_tests();
    fn run_build_config_tests();
    fn run_cache_tests();
    fn run_performance_tests();
}

#[test]
fn test_gpu_detection() {
    println!("\n=== Running GPU Detection Tests ===");
    
    // Check if CUDA is available
    if !cuda_available() {
        panic!("CUDA not available - these are REAL GPU tests, no mocks!");
    }
    
    // Run real GPU detection tests
    unsafe {
        run_gpu_detection_tests();
    }
}

#[test]
fn test_build_configuration() {
    println!("\n=== Running Build Configuration Tests ===");
    
    if !cuda_available() {
        panic!("CUDA not available - these are REAL GPU tests!");
    }
    
    unsafe {
        run_build_config_tests();
    }
}

#[test]
fn test_artifact_caching() {
    println!("\n=== Running Artifact Caching Tests ===");
    
    if !cuda_available() {
        panic!("CUDA not available - these are REAL GPU tests!");
    }
    
    unsafe {
        run_cache_tests();
    }
}

#[test]
fn test_performance_benchmarks() {
    println!("\n=== Running Performance Benchmark Tests ===");
    
    if !cuda_available() {
        panic!("CUDA not available - these are REAL GPU tests!");
    }
    
    unsafe {
        run_performance_tests();
    }
}

// Helper function to check CUDA availability
fn cuda_available() -> bool {
    use rustacuda::device::Device;
    
    match Device::num_devices() {
        Ok(count) => {
            if count > 0 {
                println!("Found {} CUDA device(s)", count);
                true
            } else {
                false
            }
        }
        Err(e) => {
            eprintln!("Failed to query CUDA devices: {:?}", e);
            false
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_full_compilation_pipeline() {
        println!("\n=== Running Full Compilation Pipeline Test ===");
        
        if !cuda_available() {
            panic!("CUDA required for pipeline test");
        }
        
        // Run all tests in sequence
        unsafe {
            run_gpu_detection_tests();
            run_build_config_tests();
            run_cache_tests();
            run_performance_tests();
        }
        
        println!("✓ Full pipeline test completed successfully");
    }
}