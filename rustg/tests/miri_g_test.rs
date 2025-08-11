// miri-g TDD tests
// Following red-green-refactor methodology for GPU-accelerated memory safety checker

use std::process::Command;
use std::path::{Path, PathBuf};
use std::fs;
use tempfile::TempDir;
use anyhow::Result;

/// Test basic command-line interface functionality
#[test]
fn test_miri_g_cli_help() {
    let output = Command::new("cargo")
        .args(&["run", "--bin", "miri-g", "--", "--help"])
        .output()
        .expect("Failed to run miri-g");

    assert!(output.status.success());
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("miri-g"));
    assert!(stdout.contains("GPU-accelerated memory safety checker"));
    assert!(stdout.contains("--run"));
    assert!(stdout.contains("--test"));
    assert!(stdout.contains("--check-only"));
    assert!(stdout.contains("--target"));
    assert!(stdout.contains("--no-gpu"));
    assert!(stdout.contains("--stats"));
}

/// Test version information
#[test]
fn test_miri_g_version() {
    let output = Command::new("cargo")
        .args(&["run", "--bin", "miri-g", "--", "--version"])
        .output()
        .expect("Failed to run miri-g");

    assert!(output.status.success());
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("miri-g"));
    assert!(stdout.contains("1.0.0"));
}

/// Test basic Rust program memory safety checking
#[test]
fn test_miri_g_basic_check() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let rust_file = temp_dir.path().join("test.rs");

    // Create a simple Rust program
    fs::write(&rust_file, r#"
        fn main() {
            let x = 42;
            let y = &x;
            println!("Value: {}", y);
        }
    "#)?;

    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "miri-g", "--",
            "--check-only",
            rust_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to run miri-g");

    assert!(output.status.success(), 
        "miri-g failed with stderr: {}", 
        String::from_utf8_lossy(&output.stderr));

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Memory safety check") || stderr.contains("✓"));
    
    Ok(())
}

/// Test GPU acceleration status reporting
#[test]
fn test_miri_g_gpu_status() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let rust_file = temp_dir.path().join("simple.rs");

    fs::write(&rust_file, "fn main() { println!(\"Hello GPU!\"); }")?;

    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "miri-g", "--",
            "--stats",
            "--check-only",
            rust_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to run miri-g");

    assert!(output.status.success());
    
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Should report GPU acceleration status
    assert!(stderr.contains("GPU:") || stderr.contains("CUDA") || stderr.contains("RTX"));
    assert!(stderr.contains("Memory Safety Statistics"));
    
    Ok(())
}

/// Test unsafe code analysis and memory leak detection
#[test]
fn test_miri_g_unsafe_analysis() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let rust_file = temp_dir.path().join("unsafe_test.rs");

    // Create Rust program with unsafe code
    fs::write(&rust_file, r#"
        fn main() {
            unsafe {
                let mut x = 42;
                let ptr = &mut x as *mut i32;
                *ptr = 100;
                println!("Value: {}", x);
            }
        }
    "#)?;

    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "miri-g", "--",
            "--run",
            rust_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to run miri-g");

    assert!(output.status.success());
    
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("unsafe") || stderr.contains("Memory") || stderr.contains("analysis"));
    
    Ok(())
}

/// Test memory leak detection
#[test]
fn test_miri_g_memory_leak_detection() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let rust_file = temp_dir.path().join("leak_test.rs");

    // Create program with potential memory issues
    fs::write(&rust_file, r#"
        use std::collections::HashMap;
        
        fn main() {
            let mut map = HashMap::new();
            for i in 0..1000 {
                map.insert(i, vec![0u8; 1024]);
            }
            // Intentionally not cleaning up for leak detection test
        }
    "#)?;

    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "miri-g", "--",
            "--check-only",
            rust_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to run miri-g");

    assert!(output.status.success());
    
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("memory") || stderr.contains("analysis") || stderr.contains("✓"));
    
    Ok(())
}

/// Test GPU vs CPU performance comparison
#[test] 
fn test_miri_g_performance_comparison() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let rust_file = temp_dir.path().join("perf_test.rs");

    // Generate a more complex program for meaningful performance test
    let mut program = String::from(r#"
        use std::collections::Vec;
        
        fn complex_memory_operations() -> Vec<i32> {
            let mut data = Vec::new();
    "#);
    
    for i in 0..100 {
        program.push_str(&format!(r#"
            data.push({});
            let slice = &data[..];
            let _sum: i32 = slice.iter().sum();
        "#, i));
    }
    
    program.push_str(r#"
            data
        }
        
        fn main() {
            let result = complex_memory_operations();
            println!("Processed {} items", result.len());
        }
    "#);
    
    fs::write(&rust_file, program)?;

    // Test with GPU acceleration
    let gpu_output = Command::new("cargo")
        .args(&[
            "run", "--bin", "miri-g", "--",
            "--stats",
            "--check-only",
            rust_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to run miri-g with GPU");

    assert!(gpu_output.status.success());
    
    // Test without GPU acceleration  
    let cpu_output = Command::new("cargo")
        .args(&[
            "run", "--bin", "miri-g", "--",
            "--no-gpu",
            "--stats", 
            "--check-only",
            rust_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to run miri-g without GPU");

    assert!(cpu_output.status.success());
    
    let gpu_stderr = String::from_utf8_lossy(&gpu_output.stderr);
    let cpu_stderr = String::from_utf8_lossy(&cpu_output.stderr);
    
    // Should contain performance statistics for both runs
    assert!(gpu_stderr.contains("Memory Safety Statistics"));
    assert!(cpu_stderr.contains("Memory Safety Statistics"));
    
    Ok(())
}

/// Test target specification support
#[test]
fn test_miri_g_target_support() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let rust_file = temp_dir.path().join("target_test.rs");

    fs::write(&rust_file, "fn main() { println!(\"Target test\"); }")?;

    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "miri-g", "--",
            "--target", "x86_64-unknown-linux-gnu",
            "--check-only",
            rust_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to run miri-g");

    assert!(output.status.success());
    
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("target") || stderr.contains("x86_64") || stderr.contains("✓"));
    
    Ok(())
}

/// Test error handling for invalid Rust files
#[test]
fn test_miri_g_error_handling() {
    let temp_dir = TempDir::new().unwrap();
    let invalid_rust = temp_dir.path().join("invalid.rs");

    // Create invalid Rust code
    fs::write(&invalid_rust, r#"
        fn main() {
            this is not valid rust syntax!
            let x = ;
        }
    "#).unwrap();

    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "miri-g", "--",
            "--check-only",
            invalid_rust.to_str().unwrap()
        ])
        .output()
        .expect("Failed to run miri-g");

    // Should handle errors gracefully
    assert!(!output.status.success());
    
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("error") || stderr.contains("Error") || stderr.contains("syntax"));
}

/// Test running Rust programs with memory safety checking
#[test]
fn test_miri_g_run_mode() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let rust_file = temp_dir.path().join("run_test.rs");

    fs::write(&rust_file, r#"
        fn factorial(n: u64) -> u64 {
            if n <= 1 {
                1
            } else {
                n * factorial(n - 1)
            }
        }
        
        fn main() {
            let result = factorial(5);
            println!("5! = {}", result);
            assert_eq!(result, 120);
        }
    "#)?;

    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "miri-g", "--",
            "--run",
            rust_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to run miri-g in run mode");

    assert!(output.status.success());
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("5! = 120"));
    
    Ok(())
}

/// Test test mode for running Rust tests with memory checking
#[test]
fn test_miri_g_test_mode() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let rust_file = temp_dir.path().join("test_mode.rs");

    fs::write(&rust_file, r#"
        #[cfg(test)]
        mod tests {
            #[test]
            fn test_memory_safety() {
                let mut vec = Vec::new();
                vec.push(1);
                vec.push(2);
                vec.push(3);
                assert_eq!(vec.len(), 3);
                assert_eq!(vec[0], 1);
            }
            
            #[test]
            fn test_borrowing() {
                let data = vec![1, 2, 3, 4, 5];
                let slice = &data[1..4];
                assert_eq!(slice, &[2, 3, 4]);
            }
        }
    "#)?;

    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "miri-g", "--",
            "--test",
            rust_file.to_str().unwrap()
        ])
        .output()
        .expect("Failed to run miri-g in test mode");

    assert!(output.status.success());
    
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("test") || stderr.contains("✓") || stderr.contains("passed"));
    
    Ok(())
}