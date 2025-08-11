// Tests for clippy-f GPU-accelerated linting tool
// Following TDD principles - tests written before implementation

use std::process::Command;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

#[test]
fn test_clippy_f_binary_exists() {
    // Check if clippy-f binary is built
    let binary_path = "./target/release/clippy-f";
    assert!(
        Path::new(binary_path).exists(),
        "clippy-f binary should exist at {}",
        binary_path
    );
}

#[test]
fn test_clippy_f_basic_usage() {
    // Test basic invocation
    let output = Command::new("./target/release/clippy-f")
        .arg("--help")
        .output()
        .expect("Failed to execute clippy-f");

    assert!(output.status.success(), "clippy-f --help should succeed");
    let help_text = String::from_utf8_lossy(&output.stdout);
    assert!(help_text.contains("GPU-accelerated Rust linter"));
    assert!(help_text.contains("--fix"));
    assert!(help_text.contains("--allow"));
    assert!(help_text.contains("--deny"));
}

#[test]
fn test_clippy_f_lint_single_file() {
    // Create test file with lint issues
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("test.rs");
    
    fs::write(&test_file, r#"
fn main() {
    let unused_var = 42;  // Should trigger unused variable warning
    let x = 1;
    let y = 2;
    if x == 1 && x == 1 {  // Should trigger redundant comparison
        println!("test");
    }
}
"#).unwrap();

    let output = Command::new("./target/release/clippy-f")
        .arg(&test_file)
        .output()
        .expect("Failed to run clippy-f");

    assert!(output.status.success());
    let output_text = String::from_utf8_lossy(&output.stdout);
    
    // Should detect unused variable
    assert!(output_text.contains("unused_var") || output_text.contains("unused"));
    // Should detect redundant comparison
    assert!(output_text.contains("redundant") || output_text.contains("comparison"));
}

#[test]
fn test_clippy_f_fix_mode() {
    // Test auto-fix functionality
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("fix_test.rs");
    
    let original_code = r#"
fn main() {
    let mut x = 5;
    x = x + 1;  // Should be fixed to x += 1
    
    if true == true {  // Should be simplified
        println!("test");
    }
}
"#;
    
    fs::write(&test_file, original_code).unwrap();

    let output = Command::new("./target/release/clippy-f")
        .arg("--fix")
        .arg(&test_file)
        .output()
        .expect("Failed to run clippy-f --fix");

    assert!(output.status.success());
    
    let fixed_code = fs::read_to_string(&test_file).unwrap();
    assert!(fixed_code.contains("x += 1"), "Should fix to compound assignment");
    assert!(!fixed_code.contains("true == true"), "Should simplify redundant comparison");
}

#[test]
fn test_clippy_f_workspace_mode() {
    // Test linting entire workspace
    let output = Command::new("./target/release/clippy-f")
        .arg("--workspace")
        .output()
        .expect("Failed to run clippy-f --workspace");

    assert!(output.status.success());
    let output_text = String::from_utf8_lossy(&output.stdout);
    
    // Should process multiple files
    assert!(output_text.contains("Checking") || output_text.contains("files"));
}

#[test]
fn test_clippy_f_custom_rules() {
    // Test custom lint rules
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("custom_test.rs");
    let config_file = temp_dir.path().join("clippy.toml");
    
    // Create config with custom rules
    fs::write(&config_file, r#"
[[custom_lints]]
name = "no_todo"
pattern = "TODO"
severity = "warn"
message = "TODO comments should be tracked in issues"

[[custom_lints]]
name = "no_println"
pattern = "println!"
severity = "deny"
message = "Use proper logging instead of println!"
"#).unwrap();

    fs::write(&test_file, r#"
fn main() {
    // TODO: implement this
    println!("Debug output");
}
"#).unwrap();

    let output = Command::new("./target/release/clippy-f")
        .arg("--config")
        .arg(&config_file)
        .arg(&test_file)
        .output()
        .expect("Failed to run clippy-f with custom config");

    let output_text = String::from_utf8_lossy(&output.stderr);
    assert!(output_text.contains("TODO comments should be tracked"));
    assert!(output_text.contains("Use proper logging"));
}

#[test]
fn test_clippy_f_gpu_specific_lints() {
    // Test GPU-specific pattern detection
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("gpu_test.rs");
    
    fs::write(&test_file, r#"
// Simulated GPU kernel code patterns
fn kernel_function() {
    let thread_id = get_thread_id();
    
    // Branch divergence issue
    if thread_id % 2 == 0 {
        expensive_operation();
    }
    
    // Uncoalesced memory access
    let data = global_array[thread_id * 17];
    
    // Shared memory bank conflict
    __shared__ let shared_data: [f32; 32];
    let value = shared_data[thread_id % 32];
}
"#).unwrap();

    let output = Command::new("./target/release/clippy-f")
        .arg("--gpu-analysis")
        .arg(&test_file)
        .output()
        .expect("Failed to run clippy-f GPU analysis");

    let output_text = String::from_utf8_lossy(&output.stdout);
    assert!(output_text.contains("divergence") || output_text.contains("branch"));
    assert!(output_text.contains("coalesced") || output_text.contains("memory access"));
}

#[test]
fn test_clippy_f_performance() {
    // Test performance vs standard clippy
    use std::time::Instant;
    
    // Run standard clippy (if available)
    let clippy_start = Instant::now();
    let _ = Command::new("cargo")
        .arg("clippy")
        .arg("--")
        .arg("-W")
        .arg("clippy::all")
        .output();
    let clippy_duration = clippy_start.elapsed();
    
    // Run clippy-f
    let clippy_f_start = Instant::now();
    let _ = Command::new("./target/release/clippy-f")
        .arg("--workspace")
        .output()
        .expect("Failed to run clippy-f");
    let clippy_f_duration = clippy_f_start.elapsed();
    
    // clippy-f should be significantly faster (target: 10x)
    // Allow some variance in CI environments
    if clippy_duration.as_millis() > 100 {
        let speedup = clippy_duration.as_secs_f64() / clippy_f_duration.as_secs_f64();
        assert!(
            speedup > 5.0,
            "clippy-f should be at least 5x faster than standard clippy (got {}x)",
            speedup
        );
    }
}

#[test]
fn test_clippy_f_json_output() {
    // Test JSON output format for tooling integration
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("json_test.rs");
    
    fs::write(&test_file, r#"
fn main() {
    let unused = 42;
}
"#).unwrap();

    let output = Command::new("./target/release/clippy-f")
        .arg("--output-format")
        .arg("json")
        .arg(&test_file)
        .output()
        .expect("Failed to run clippy-f with JSON output");

    assert!(output.status.success());
    let json_output = String::from_utf8_lossy(&output.stdout);
    
    // Parse JSON to verify structure
    let parsed: serde_json::Value = serde_json::from_str(&json_output)
        .expect("Output should be valid JSON");
    
    assert!(parsed["diagnostics"].is_array());
    assert!(parsed["diagnostics"][0]["file"].is_string());
    assert!(parsed["diagnostics"][0]["line"].is_number());
}

#[test]
fn test_clippy_f_integration_with_cargo_g() {
    // Test integration with cargo-g
    let output = Command::new("./target/release/cargo-g")
        .arg("clippy")
        .output()
        .expect("Failed to run cargo-g clippy");

    assert!(output.status.success(), "cargo-g clippy should invoke clippy-f");
    
    let output_text = String::from_utf8_lossy(&output.stdout);
    assert!(output_text.contains("GPU-accelerated") || output_text.contains("clippy-f"));
}

#[test]
fn test_clippy_f_parallel_file_processing() {
    // Test parallel processing of multiple files
    let temp_dir = TempDir::new().unwrap();
    
    // Create multiple test files
    for i in 0..10 {
        let test_file = temp_dir.path().join(format!("test_{}.rs", i));
        fs::write(&test_file, format!(r#"
fn function_{}() {{
    let unused_{} = {};
}}
"#, i, i, i)).unwrap();
    }

    let output = Command::new("./target/release/clippy-f")
        .arg(temp_dir.path())
        .output()
        .expect("Failed to run clippy-f on directory");

    assert!(output.status.success());
    let output_text = String::from_utf8_lossy(&output.stdout);
    
    // Should process all files
    for i in 0..10 {
        assert!(
            output_text.contains(&format!("test_{}.rs", i)) ||
            output_text.contains(&format!("unused_{}", i)),
            "Should process test_{}.rs",
            i
        );
    }
}

#[test]
fn test_clippy_f_cache_functionality() {
    // Test caching for improved performance
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("cache_test.rs");
    
    fs::write(&test_file, r#"
fn main() {
    let x = 42;
}
"#).unwrap();

    // First run - should build cache
    let first_run = Command::new("./target/release/clippy-f")
        .arg(&test_file)
        .output()
        .expect("Failed first run");
    assert!(first_run.status.success());

    // Second run - should use cache (faster)
    use std::time::Instant;
    let start = Instant::now();
    let second_run = Command::new("./target/release/clippy-f")
        .arg(&test_file)
        .output()
        .expect("Failed second run");
    let cached_duration = start.elapsed();
    
    assert!(second_run.status.success());
    assert!(cached_duration.as_millis() < 50, "Cached run should be very fast");
}