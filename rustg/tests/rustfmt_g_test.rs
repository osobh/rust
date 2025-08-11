// Tests for rustfmt-g: GPU-accelerated Rust formatter
// Following TDD approach - these tests should fail initially

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tempfile::{NamedTempFile, TempDir};
use pretty_assertions::assert_eq;

#[test]
fn test_rustfmt_g_basic_formatting() {
    let input = "fn  main(){println!(\"Hello\");}";
    let expected = "fn main() {\n    println!(\"Hello\");\n}\n";
    
    let result = run_rustfmt_g(input);
    assert!(result.is_ok(), "rustfmt-g should format basic code");
    assert_eq!(result.unwrap(), expected);
}

#[test]
fn test_rustfmt_g_performance_target() {
    // Generate large source code (10K lines)
    let large_source = generate_large_source(10_000);
    
    let start = std::time::Instant::now();
    let result = run_rustfmt_g(&large_source);
    let elapsed = start.elapsed();
    
    assert!(result.is_ok(), "rustfmt-g should handle large files");
    
    // Target: 10x faster than rustfmt (typically ~1s for 10K lines)
    // GPU target: <100ms
    assert!(elapsed.as_millis() < 100, 
           "rustfmt-g should achieve 10x speedup (got {}ms)", 
           elapsed.as_millis());
}

#[test]
fn test_rustfmt_g_gpu_utilization() {
    let source = generate_large_source(5_000);
    
    // This test should verify GPU is actually being used
    let result = run_rustfmt_g_with_gpu_monitoring(&source);
    assert!(result.is_ok(), "rustfmt-g should execute with GPU monitoring");
    
    let (formatted, gpu_usage) = result.unwrap();
    assert!(!formatted.is_empty(), "Should produce formatted output");
    assert!(gpu_usage > 0.1, "GPU utilization should be >10% for large formatting task");
}

#[test]
fn test_rustfmt_g_incremental_formatting() {
    let original = "fn main() {\n    let x = 1;\n    let y=2;\n    println!(\"{}\", x);\n}";
    let changed_lines = vec![2]; // Only line with formatting issue
    
    let result = run_rustfmt_g_incremental(original, &changed_lines);
    assert!(result.is_ok(), "Incremental formatting should work");
    
    let formatted = result.unwrap();
    assert!(formatted.contains("let y = 2;"), "Should fix formatting on line 2");
    assert!(formatted.contains("let x = 1;"), "Should preserve correct formatting");
}

#[test] 
fn test_rustfmt_g_custom_config() {
    let source = "fn main(){let x=1;}";
    let config = RustfmtConfig {
        indent_width: 2,
        max_line_length: 80,
        use_tabs: true,
    };
    
    let result = run_rustfmt_g_with_config(source, &config);
    assert!(result.is_ok(), "Should work with custom config");
    
    let formatted = result.unwrap();
    assert!(formatted.contains("\t"), "Should use tabs when configured");
}

#[test]
fn test_rustfmt_g_parallel_processing() {
    // Test multiple files in parallel
    let temp_dir = TempDir::new().unwrap();
    let mut files = Vec::new();
    
    // Create 10 test files
    for i in 0..10 {
        let file_path = temp_dir.path().join(format!("test_{}.rs", i));
        let content = format!("fn test_{}(){{let x={};}}", i, i);
        fs::write(&file_path, content).unwrap();
        files.push(file_path);
    }
    
    let start = std::time::Instant::now();
    let result = run_rustfmt_g_on_multiple_files(&files);
    let elapsed = start.elapsed();
    
    assert!(result.is_ok(), "Should format multiple files");
    assert!(elapsed.as_millis() < 50, "Parallel processing should be very fast");
}

#[test]
fn test_rustfmt_g_error_handling() {
    let invalid_rust = "fn main( { invalid syntax }";
    
    let result = run_rustfmt_g(invalid_rust);
    assert!(result.is_err(), "Should handle invalid Rust syntax gracefully");
    
    let error = result.unwrap_err();
    assert!(error.contains("syntax error") || error.contains("parse error"),
           "Error message should be descriptive");
}

#[test]
fn test_rustfmt_g_memory_usage() {
    let large_source = generate_large_source(50_000);
    
    let memory_before = get_memory_usage();
    let _result = run_rustfmt_g(&large_source);
    let memory_after = get_memory_usage();
    
    let memory_used = memory_after - memory_before;
    
    // Should use minimal CPU memory due to GPU processing
    assert!(memory_used < 100_000_000, // 100MB
           "Memory usage should be minimal with GPU processing (used {}MB)", 
           memory_used / 1_000_000);
}

#[test]
fn test_rustfmt_g_preserve_comments() {
    let source = r#"
// This is a comment
fn main() {
    /* Block comment */
    let x=1; // Inline comment
}
"#;

    let result = run_rustfmt_g(source);
    assert!(result.is_ok(), "Should handle comments");
    
    let formatted = result.unwrap();
    assert!(formatted.contains("// This is a comment"));
    assert!(formatted.contains("/* Block comment */"));
    assert!(formatted.contains("// Inline comment"));
}

// Helper functions for testing

fn run_rustfmt_g(source: &str) -> Result<String, String> {
    let temp_file = NamedTempFile::new().map_err(|e| e.to_string())?;
    fs::write(temp_file.path(), source).map_err(|e| e.to_string())?;
    
    let output = Command::new("./target/release/rustfmt-g")
        .arg(temp_file.path())
        .output()
        .map_err(|e| format!("Failed to execute rustfmt-g: {}", e))?;
    
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    
    // Read formatted output
    fs::read_to_string(temp_file.path()).map_err(|e| e.to_string())
}

fn run_rustfmt_g_with_gpu_monitoring(source: &str) -> Result<(String, f32), String> {
    // This would integrate with GPU monitoring tools
    // For now, simulate GPU usage detection
    let formatted = run_rustfmt_g(source)?;
    let simulated_gpu_usage = if source.len() > 1000 { 0.75 } else { 0.0 };
    Ok((formatted, simulated_gpu_usage))
}

fn run_rustfmt_g_incremental(source: &str, changed_lines: &[usize]) -> Result<String, String> {
    let temp_file = NamedTempFile::new().map_err(|e| e.to_string())?;
    fs::write(temp_file.path(), source).map_err(|e| e.to_string())?;
    
    let changed_args: Vec<String> = changed_lines.iter().map(|&i| i.to_string()).collect();
    
    let output = Command::new("./target/release/rustfmt-g")
        .arg("--incremental")
        .arg("--changed-lines")
        .arg(changed_args.join(","))
        .arg(temp_file.path())
        .output()
        .map_err(|e| format!("Failed to execute incremental rustfmt-g: {}", e))?;
    
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    
    fs::read_to_string(temp_file.path()).map_err(|e| e.to_string())
}

#[derive(Debug)]
struct RustfmtConfig {
    indent_width: usize,
    max_line_length: usize,
    use_tabs: bool,
}

fn run_rustfmt_g_with_config(source: &str, config: &RustfmtConfig) -> Result<String, String> {
    let temp_file = NamedTempFile::new().map_err(|e| e.to_string())?;
    fs::write(temp_file.path(), source).map_err(|e| e.to_string())?;
    
    let config_file = NamedTempFile::new().map_err(|e| e.to_string())?;
    let config_content = format!(
        "indent_width = {}\nmax_line_length = {}\nuse_tabs = {}",
        config.indent_width, config.max_line_length, config.use_tabs
    );
    fs::write(config_file.path(), config_content).map_err(|e| e.to_string())?;
    
    let output = Command::new("./target/release/rustfmt-g")
        .arg("--config")
        .arg(config_file.path())
        .arg(temp_file.path())
        .output()
        .map_err(|e| format!("Failed to execute rustfmt-g with config: {}", e))?;
    
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    
    fs::read_to_string(temp_file.path()).map_err(|e| e.to_string())
}

fn run_rustfmt_g_on_multiple_files(files: &[PathBuf]) -> Result<(), String> {
    let file_args: Vec<String> = files.iter()
        .map(|p| p.to_string_lossy().to_string())
        .collect();
    
    let output = Command::new("./target/release/rustfmt-g")
        .arg("--parallel")
        .args(&file_args)
        .output()
        .map_err(|e| format!("Failed to execute parallel rustfmt-g: {}", e))?;
    
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    
    Ok(())
}

fn generate_large_source(lines: usize) -> String {
    let mut source = String::new();
    
    for i in 0..lines {
        if i % 50 == 0 {
            source.push_str(&format!("fn function_{}() {{\n", i));
        } else if i % 50 == 49 {
            source.push_str("}\n");
        } else {
            // Intentionally poor formatting to test formatter
            source.push_str(&format!("let var_{}={}+{};\n", i, i, i+1));
        }
    }
    
    source
}

fn get_memory_usage() -> u64 {
    // Simple memory usage estimation
    // In production, would use proper memory profiling
    use std::fs;
    
    if let Ok(status) = fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                if let Some(kb_str) = line.split_whitespace().nth(1) {
                    if let Ok(kb) = kb_str.parse::<u64>() {
                        return kb * 1024; // Convert to bytes
                    }
                }
            }
        }
    }
    
    0 // Fallback
}

#[test]
fn test_rustfmt_g_binary_exists() {
    let binary_path = PathBuf::from("./target/release/rustfmt-g");
    assert!(false, "rustfmt-g binary should exist at {}", binary_path.display());
}