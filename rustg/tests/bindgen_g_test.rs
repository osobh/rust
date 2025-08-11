// bindgen-g TDD tests
// Following red-green-refactor methodology for GPU-accelerated FFI bindings generator

use std::process::Command;
use std::path::{Path, PathBuf};
use std::fs;
use tempfile::TempDir;
use anyhow::Result;

/// Test basic command-line interface functionality
#[test]
fn test_bindgen_g_cli_help() {
    let output = Command::new("cargo")
        .args(&["run", "--bin", "bindgen-g", "--", "--help"])
        .output()
        .expect("Failed to run bindgen-g");

    assert!(output.status.success());
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("bindgen-g"));
    assert!(stdout.contains("GPU-accelerated FFI bindings generator"));
    assert!(stdout.contains("--output"));
    assert!(stdout.contains("--whitelist"));
    assert!(stdout.contains("--blacklist"));
    assert!(stdout.contains("--generate-comments"));
    assert!(stdout.contains("--no-gpu"));
    assert!(stdout.contains("--stats"));
}

/// Test version information
#[test]
fn test_bindgen_g_version() {
    let output = Command::new("cargo")
        .args(&["run", "--bin", "bindgen-g", "--", "--version"])
        .output()
        .expect("Failed to run bindgen-g");

    assert!(output.status.success());
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("bindgen-g"));
    assert!(stdout.contains("1.0.0"));
}

/// Test basic header file processing
#[test]
fn test_bindgen_g_basic_header() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let header_path = temp_dir.path().join("test.h");
    let output_path = temp_dir.path().join("bindings.rs");

    // Create a simple C header file
    fs::write(&header_path, r#"
        #include <stdint.h>
        
        typedef struct {
            uint32_t id;
            float value;
        } TestStruct;
        
        int test_function(TestStruct* data);
    "#)?;

    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "bindgen-g", "--",
            "--output", output_path.to_str().unwrap(),
            header_path.to_str().unwrap()
        ])
        .output()
        .expect("Failed to run bindgen-g");

    assert!(output.status.success(), 
        "bindgen-g failed with stderr: {}", 
        String::from_utf8_lossy(&output.stderr));

    // Check that bindings file was generated
    assert!(output_path.exists(), "Output bindings file should exist");
    
    let bindings = fs::read_to_string(&output_path)?;
    assert!(bindings.contains("TestStruct"));
    assert!(bindings.contains("test_function"));
    
    Ok(())
}

/// Test GPU acceleration status reporting
#[test]
fn test_bindgen_g_gpu_status() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let header_path = temp_dir.path().join("simple.h");
    let output_path = temp_dir.path().join("bindings.rs");

    fs::write(&header_path, "#include <stdint.h>\nint simple_func(void);")?;

    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "bindgen-g", "--",
            "--stats",
            "--output", output_path.to_str().unwrap(),
            header_path.to_str().unwrap()
        ])
        .output()
        .expect("Failed to run bindgen-g");

    assert!(output.status.success());
    
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Should report GPU acceleration status
    assert!(stderr.contains("GPU:") || stderr.contains("CUDA") || stderr.contains("RTX"));
    assert!(stderr.contains("Performance Statistics"));
    
    Ok(())
}

/// Test multiple header files processing  
#[test]
fn test_bindgen_g_multiple_headers() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let header1 = temp_dir.path().join("header1.h");
    let header2 = temp_dir.path().join("header2.h");
    let output_path = temp_dir.path().join("bindings.rs");

    fs::write(&header1, r#"
        typedef struct { int x; } Struct1;
        void func1(void);
    "#)?;
    
    fs::write(&header2, r#"
        typedef struct { float y; } Struct2;
        int func2(int arg);
    "#)?;

    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "bindgen-g", "--",
            "--output", output_path.to_str().unwrap(),
            header1.to_str().unwrap(),
            header2.to_str().unwrap()
        ])
        .output()
        .expect("Failed to run bindgen-g");

    assert!(output.status.success());
    
    let bindings = fs::read_to_string(&output_path)?;
    assert!(bindings.contains("Struct1"));
    assert!(bindings.contains("Struct2"));
    assert!(bindings.contains("func1"));
    assert!(bindings.contains("func2"));
    
    Ok(())
}

/// Test whitelist functionality
#[test]
fn test_bindgen_g_whitelist() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let header_path = temp_dir.path().join("test.h");
    let output_path = temp_dir.path().join("bindings.rs");

    fs::write(&header_path, r#"
        void wanted_function(void);
        void unwanted_function(void);
        
        typedef struct { int value; } WantedStruct;
        typedef struct { float data; } UnwantedStruct;
    "#)?;

    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "bindgen-g", "--",
            "--whitelist-function", "wanted_.*",
            "--whitelist-type", "Wanted.*",
            "--output", output_path.to_str().unwrap(),
            header_path.to_str().unwrap()
        ])
        .output()
        .expect("Failed to run bindgen-g");

    assert!(output.status.success());
    
    let bindings = fs::read_to_string(&output_path)?;
    assert!(bindings.contains("wanted_function"));
    assert!(bindings.contains("WantedStruct"));
    assert!(!bindings.contains("unwanted_function"));
    assert!(!bindings.contains("UnwantedStruct"));
    
    Ok(())
}

/// Test blacklist functionality
#[test]
fn test_bindgen_g_blacklist() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let header_path = temp_dir.path().join("test.h");
    let output_path = temp_dir.path().join("bindings.rs");

    fs::write(&header_path, r#"
        void good_function(void);
        void bad_function(void);
        
        typedef struct { int value; } GoodStruct;
        typedef struct { float data; } BadStruct;
    "#)?;

    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "bindgen-g", "--",
            "--blacklist-function", "bad_.*",
            "--blacklist-type", "Bad.*",
            "--output", output_path.to_str().unwrap(),
            header_path.to_str().unwrap()
        ])
        .output()
        .expect("Failed to run bindgen-g");

    assert!(output.status.success());
    
    let bindings = fs::read_to_string(&output_path)?;
    assert!(bindings.contains("good_function"));
    assert!(bindings.contains("GoodStruct"));
    assert!(!bindings.contains("bad_function"));
    assert!(!bindings.contains("BadStruct"));
    
    Ok(())
}

/// Test GPU vs CPU performance comparison
#[test] 
fn test_bindgen_g_performance_comparison() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let header_path = temp_dir.path().join("large.h");
    let gpu_output = temp_dir.path().join("gpu_bindings.rs");
    let cpu_output = temp_dir.path().join("cpu_bindings.rs");

    // Generate a larger header file for meaningful performance test
    let mut header_content = String::from("#include <stdint.h>\n");
    for i in 0..100 {
        header_content.push_str(&format!(r#"
            typedef struct {{
                uint32_t field1_{};
                float field2_{};
                double field3_{};
            }} TestStruct{};
            
            int test_function_{}(TestStruct{} *data);
        "#, i, i, i, i, i, i));
    }
    fs::write(&header_path, header_content)?;

    // Test with GPU acceleration
    let gpu_output = Command::new("cargo")
        .args(&[
            "run", "--bin", "bindgen-g", "--",
            "--stats",
            "--output", gpu_output.to_str().unwrap(),
            header_path.to_str().unwrap()
        ])
        .output()
        .expect("Failed to run bindgen-g with GPU");

    assert!(gpu_output.status.success());
    
    // Test without GPU acceleration  
    let cpu_output = Command::new("cargo")
        .args(&[
            "run", "--bin", "bindgen-g", "--",
            "--no-gpu",
            "--stats", 
            "--output", cpu_output.to_str().unwrap(),
            header_path.to_str().unwrap()
        ])
        .output()
        .expect("Failed to run bindgen-g without GPU");

    assert!(cpu_output.status.success());
    
    let gpu_stderr = String::from_utf8_lossy(&gpu_output.stderr);
    let cpu_stderr = String::from_utf8_lossy(&cpu_output.stderr);
    
    // Should contain performance statistics for both runs
    assert!(gpu_stderr.contains("Performance Statistics"));
    assert!(cpu_stderr.contains("Performance Statistics"));
    
    Ok(())
}

/// Test error handling for invalid header files
#[test]
fn test_bindgen_g_error_handling() {
    let temp_dir = TempDir::new().unwrap();
    let invalid_header = temp_dir.path().join("invalid.h");
    let output_path = temp_dir.path().join("bindings.rs");

    // Create invalid C header
    fs::write(&invalid_header, r#"
        this is not valid C code!
        #include <nonexistent.h>
        syntax error here
    "#).unwrap();

    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "bindgen-g", "--",
            "--output", output_path.to_str().unwrap(),
            invalid_header.to_str().unwrap()
        ])
        .output()
        .expect("Failed to run bindgen-g");

    // Should handle errors gracefully
    assert!(!output.status.success());
    
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("error") || stderr.contains("Error"));
}

/// Test comment generation
#[test]
fn test_bindgen_g_generate_comments() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let header_path = temp_dir.path().join("documented.h");
    let output_path = temp_dir.path().join("bindings.rs");

    fs::write(&header_path, r#"
        /// This is a documented function
        /// @param value Input value
        /// @return Result code
        int documented_function(int value);
        
        /// This structure holds data
        typedef struct {
            int data; ///< The data field
        } DocumentedStruct;
    "#)?;

    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "bindgen-g", "--",
            "--generate-comments",
            "--output", output_path.to_str().unwrap(),
            header_path.to_str().unwrap()
        ])
        .output()
        .expect("Failed to run bindgen-g");

    assert!(output.status.success());
    
    let bindings = fs::read_to_string(&output_path)?;
    assert!(bindings.contains("documented_function"));
    assert!(bindings.contains("DocumentedStruct"));
    // Should include generated documentation comments
    assert!(bindings.contains("///") || bindings.contains("//"));
    
    Ok(())
}

/// Test include path management
#[test]
fn test_bindgen_g_include_paths() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let include_dir = temp_dir.path().join("include");
    fs::create_dir(&include_dir)?;
    
    let common_header = include_dir.join("common.h");
    let main_header = temp_dir.path().join("main.h");
    let output_path = temp_dir.path().join("bindings.rs");

    fs::write(&common_header, r#"
        #ifndef COMMON_H
        #define COMMON_H
        
        typedef struct {
            int shared_field;
        } CommonStruct;
        
        #endif
    "#)?;

    fs::write(&main_header, r#"
        #include "common.h"
        
        void use_common(CommonStruct *data);
    "#)?;

    let output = Command::new("cargo")
        .args(&[
            "run", "--bin", "bindgen-g", "--",
            "--include-path", include_dir.to_str().unwrap(),
            "--output", output_path.to_str().unwrap(),
            main_header.to_str().unwrap()
        ])
        .output()
        .expect("Failed to run bindgen-g");

    assert!(output.status.success());
    
    let bindings = fs::read_to_string(&output_path)?;
    assert!(bindings.contains("CommonStruct"));
    assert!(bindings.contains("use_common"));
    
    Ok(())
}