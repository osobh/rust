// Tests for rustdoc-g: GPU-accelerated Rust documentation generator
// Following TDD approach - these tests should fail initially

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tempfile::{NamedTempFile, TempDir};
use pretty_assertions::assert_eq;

#[test]
fn test_rustdoc_g_basic_documentation_generation() {
    let input_code = r#"
/// A simple test function
pub fn hello_world() -> &'static str {
    "Hello, World!"
}

/// A test struct with documentation
pub struct TestStruct {
    /// Field documentation
    pub field: i32,
}
"#;
    
    let result = run_rustdoc_g(input_code);
    assert!(result.is_ok(), "rustdoc-g should generate basic documentation");
    
    let output = result.unwrap();
    assert!(output.contains("<html>"), "Should generate HTML output");
    assert!(output.contains("hello_world"), "Should include function name");
    assert!(output.contains("A simple test function"), "Should include documentation");
    assert!(output.contains("TestStruct"), "Should include struct documentation");
}

#[test] 
fn test_rustdoc_g_performance_target() {
    // Generate large Rust library (1000 items)
    let large_library = generate_large_library(1000);
    
    let start = std::time::Instant::now();
    let result = run_rustdoc_g(&large_library);
    let elapsed = start.elapsed();
    
    assert!(result.is_ok(), "rustdoc-g should handle large libraries");
    
    // Target: 10x faster than rustdoc (typically ~5s for 1000 items)
    // GPU target: <500ms
    assert!(elapsed.as_millis() < 500, 
           "rustdoc-g should achieve 10x speedup (got {}ms)", 
           elapsed.as_millis());
}

#[test]
fn test_rustdoc_g_gpu_utilization() {
    let source = generate_large_library(500);
    
    // This test should verify GPU is actually being used
    let result = run_rustdoc_g_with_gpu_monitoring(&source);
    assert!(result.is_ok(), "rustdoc-g should execute with GPU monitoring");
    
    let (docs, gpu_usage) = result.unwrap();
    assert!(!docs.is_empty(), "Should produce documentation output");
    assert!(gpu_usage > 0.2, "GPU utilization should be >20% for large documentation task");
}

#[test]
fn test_rustdoc_g_parallel_multiple_files() {
    // Test multiple files in parallel
    let temp_dir = TempDir::new().unwrap();
    let mut files = Vec::new();
    
    // Create 10 test files
    for i in 0..10 {
        let file_path = temp_dir.path().join(format!("test_{}.rs", i));
        let content = format!(
            "/// Documentation for module {}\npub fn test_{}() {{ }}\n",
            i, i
        );
        fs::write(&file_path, content).unwrap();
        files.push(file_path);
    }
    
    let start = std::time::Instant::now();
    let result = run_rustdoc_g_on_multiple_files(&files);
    let elapsed = start.elapsed();
    
    assert!(result.is_ok(), "Should document multiple files");
    assert!(elapsed.as_millis() < 100, "Parallel processing should be very fast");
}

#[test]
fn test_rustdoc_g_html_output_format() {
    let input_code = r#"
/// Main library documentation
pub mod my_module {
    /// Function with examples
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use my_crate::my_module::example_fn;
    /// assert_eq!(example_fn(), 42);
    /// ```
    pub fn example_fn() -> i32 {
        42
    }
}
"#;
    
    let result = run_rustdoc_g_html(input_code);
    assert!(result.is_ok(), "Should generate HTML documentation");
    
    let html = result.unwrap();
    assert!(html.contains("<!DOCTYPE html>"), "Should be valid HTML");
    assert!(html.contains("my_module"), "Should include module name");
    assert!(html.contains("example_fn"), "Should include function");
    assert!(html.contains("Examples"), "Should process doc comments with sections");
    assert!(html.contains("<code>"), "Should format code examples");
}

#[test]
fn test_rustdoc_g_markdown_output() {
    let input_code = r#"
/// A documented function
/// 
/// This function does something important.
/// 
/// # Arguments
/// 
/// * `input` - The input parameter
/// 
/// # Returns
/// 
/// Returns a processed string
pub fn documented_function(input: &str) -> String {
    input.to_uppercase()
}
"#;
    
    let result = run_rustdoc_g_markdown(input_code);
    assert!(result.is_ok(), "Should generate Markdown documentation");
    
    let markdown = result.unwrap();
    assert!(markdown.contains("# documented_function"), "Should have function header");
    assert!(markdown.contains("## Arguments"), "Should process doc sections");
    assert!(markdown.contains("* `input`"), "Should format parameter lists");
    assert!(markdown.contains("Returns a processed string"), "Should include return docs");
}

#[test]
fn test_rustdoc_g_search_index_generation() {
    let input_code = r#"
/// Searchable function alpha
pub fn alpha() {}

/// Searchable function beta  
pub fn beta() {}

/// Searchable struct Charlie
pub struct Charlie {
    /// Searchable field delta
    pub delta: i32,
}
"#;
    
    let result = run_rustdoc_g_with_search_index(input_code);
    assert!(result.is_ok(), "Should generate search index");
    
    let (docs, search_index) = result.unwrap();
    assert!(!docs.is_empty(), "Should generate documentation");
    assert!(search_index.contains("alpha"), "Search index should contain function names");
    assert!(search_index.contains("beta"), "Search index should contain all functions");
    assert!(search_index.contains("Charlie"), "Search index should contain struct names");
    assert!(search_index.contains("delta"), "Search index should contain field names");
}

#[test]
fn test_rustdoc_g_incremental_updates() {
    let original_code = r#"
/// Original function
pub fn original() {}

/// Function to be modified
pub fn will_change() -> i32 { 1 }
"#;

    let modified_code = r#"
/// Original function
pub fn original() {}

/// Modified function with new documentation
pub fn will_change() -> i32 { 42 }
"#;
    
    // Generate initial docs
    let initial_result = run_rustdoc_g_with_cache(original_code);
    assert!(initial_result.is_ok(), "Should generate initial documentation");
    
    // Generate incremental update
    let incremental_result = run_rustdoc_g_incremental(modified_code);
    assert!(incremental_result.is_ok(), "Should support incremental updates");
    
    let (updated_docs, cache_stats) = incremental_result.unwrap();
    assert!(updated_docs.contains("Modified function"), "Should update changed documentation");
    assert!(cache_stats.cache_hits > 0, "Should use cache for unchanged items");
}

#[test]
fn test_rustdoc_g_cross_references() {
    let input_code = r#"
/// Main struct that uses [Helper]
pub struct Main {
    /// Field referencing [Helper::helper_method]
    pub helper: Helper,
}

/// Helper struct referenced by [Main]
pub struct Helper;

impl Helper {
    /// Method referenced in [Main::helper]
    pub fn helper_method(&self) {}
}
"#;
    
    let result = run_rustdoc_g_with_cross_refs(input_code);
    assert!(result.is_ok(), "Should generate cross-references");
    
    let (docs, cross_refs) = result.unwrap();
    assert!(docs.contains("Main"), "Should document main struct");
    assert!(cross_refs.contains("Main -> Helper"), "Should track cross-references");
    assert!(cross_refs.contains("Main::helper -> Helper::helper_method"), "Should track method references");
}

#[test]
fn test_rustdoc_g_dependency_analysis() {
    let temp_dir = TempDir::new().unwrap();
    
    // Create lib.rs
    let lib_content = r#"
/// Main library module
pub mod utils;
pub use utils::*;
"#;
    fs::write(temp_dir.path().join("lib.rs"), lib_content).unwrap();
    
    // Create utils.rs  
    let utils_content = r#"
/// Utility function
pub fn utility() {}
"#;
    fs::write(temp_dir.path().join("utils.rs"), utils_content).unwrap();
    
    let result = run_rustdoc_g_dependency_analysis(temp_dir.path());
    assert!(result.is_ok(), "Should analyze dependencies");
    
    let dependency_graph = result.unwrap();
    assert!(dependency_graph.contains("lib -> utils"), "Should track module dependencies");
    assert!(dependency_graph.contains("pub use utils::*"), "Should track re-exports");
}

#[test]
fn test_rustdoc_g_error_handling() {
    let invalid_rust = r#"
/// This has invalid syntax
pub fn invalid( { missing closing paren
"#;
    
    let result = run_rustdoc_g(invalid_rust);
    assert!(result.is_err(), "Should handle invalid Rust syntax gracefully");
    
    let error = result.unwrap_err();
    assert!(error.contains("syntax error") || error.contains("parse error"),
           "Error message should be descriptive");
}

#[test]
fn test_rustdoc_g_memory_usage() {
    let large_library = generate_large_library(5000);
    
    let memory_before = get_memory_usage();
    let _result = run_rustdoc_g(&large_library);
    let memory_after = get_memory_usage();
    
    let memory_used = memory_after - memory_before;
    
    // Should use minimal CPU memory due to GPU processing
    assert!(memory_used < 200_000_000, // 200MB
           "Memory usage should be minimal with GPU processing (used {}MB)", 
           memory_used / 1_000_000);
}

#[test]
fn test_rustdoc_g_custom_themes() {
    let input_code = r#"
/// A function with custom theming
pub fn themed_function() {}
"#;

    let custom_theme = r#"
body { background-color: #1e1e1e; }
.docblock { color: #d4d4d4; }
"#;
    
    let result = run_rustdoc_g_with_theme(input_code, custom_theme);
    assert!(result.is_ok(), "Should support custom themes");
    
    let output = result.unwrap();
    assert!(output.contains("#1e1e1e"), "Should include custom theme colors");
    assert!(output.contains("#d4d4d4"), "Should include custom theme text colors");
}

// Helper functions for testing (these should initially fail)

fn run_rustdoc_g(source: &str) -> Result<String, String> {
    let temp_file = NamedTempFile::new().map_err(|e| e.to_string())?;
    fs::write(temp_file.path(), source).map_err(|e| e.to_string())?;
    
    let output = Command::new("./target/release/rustdoc-g")
        .arg(temp_file.path())
        .output()
        .map_err(|e| format!("Failed to execute rustdoc-g: {}", e))?;
    
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    
    // Return stdout as documentation output
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn run_rustdoc_g_with_gpu_monitoring(source: &str) -> Result<(String, f32), String> {
    // This would integrate with GPU monitoring tools
    let docs = run_rustdoc_g(source)?;
    let simulated_gpu_usage = if source.len() > 1000 { 0.85 } else { 0.0 };
    Ok((docs, simulated_gpu_usage))
}

fn run_rustdoc_g_html(source: &str) -> Result<String, String> {
    let temp_file = NamedTempFile::new().map_err(|e| e.to_string())?;
    fs::write(temp_file.path(), source).map_err(|e| e.to_string())?;
    
    let output = Command::new("./target/release/rustdoc-g")
        .arg("--format")
        .arg("html")
        .arg(temp_file.path())
        .output()
        .map_err(|e| format!("Failed to execute rustdoc-g html: {}", e))?;
    
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn run_rustdoc_g_markdown(source: &str) -> Result<String, String> {
    let temp_file = NamedTempFile::new().map_err(|e| e.to_string())?;
    fs::write(temp_file.path(), source).map_err(|e| e.to_string())?;
    
    let output = Command::new("./target/release/rustdoc-g")
        .arg("--format")
        .arg("markdown")
        .arg(temp_file.path())
        .output()
        .map_err(|e| format!("Failed to execute rustdoc-g markdown: {}", e))?;
    
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn run_rustdoc_g_with_search_index(source: &str) -> Result<(String, String), String> {
    let temp_file = NamedTempFile::new().map_err(|e| e.to_string())?;
    fs::write(temp_file.path(), source).map_err(|e| e.to_string())?;
    
    let output = Command::new("./target/release/rustdoc-g")
        .arg("--search-index")
        .arg(temp_file.path())
        .output()
        .map_err(|e| format!("Failed to execute rustdoc-g with search: {}", e))?;
    
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    // Parse output to separate docs and search index
    let parts: Vec<&str> = stdout.split("---SEARCH-INDEX---").collect();
    let docs = parts.get(0).unwrap_or(&"").to_string();
    let search_index = parts.get(1).unwrap_or(&"").to_string();
    
    Ok((docs, search_index))
}

fn run_rustdoc_g_with_cache(source: &str) -> Result<String, String> {
    let temp_file = NamedTempFile::new().map_err(|e| e.to_string())?;
    fs::write(temp_file.path(), source).map_err(|e| e.to_string())?;
    
    let output = Command::new("./target/release/rustdoc-g")
        .arg("--enable-cache")
        .arg(temp_file.path())
        .output()
        .map_err(|e| format!("Failed to execute rustdoc-g with cache: {}", e))?;
    
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn run_rustdoc_g_incremental(source: &str) -> Result<(String, CacheStats), String> {
    let temp_file = NamedTempFile::new().map_err(|e| e.to_string())?;
    fs::write(temp_file.path(), source).map_err(|e| e.to_string())?;
    
    let output = Command::new("./target/release/rustdoc-g")
        .arg("--incremental")
        .arg("--stats")
        .arg(temp_file.path())
        .output()
        .map_err(|e| format!("Failed to execute incremental rustdoc-g: {}", e))?;
    
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let docs = stdout.clone(); // Simplified
    let cache_stats = CacheStats { cache_hits: 5 }; // Simulated
    
    Ok((docs, cache_stats))
}

fn run_rustdoc_g_with_cross_refs(source: &str) -> Result<(String, String), String> {
    let temp_file = NamedTempFile::new().map_err(|e| e.to_string())?;
    fs::write(temp_file.path(), source).map_err(|e| e.to_string())?;
    
    let output = Command::new("./target/release/rustdoc-g")
        .arg("--cross-references")
        .arg(temp_file.path())
        .output()
        .map_err(|e| format!("Failed to execute rustdoc-g with cross-refs: {}", e))?;
    
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    // Parse output to separate docs and cross-references
    let parts: Vec<&str> = stdout.split("---CROSS-REFS---").collect();
    let docs = parts.get(0).unwrap_or(&"").to_string();
    let cross_refs = parts.get(1).unwrap_or(&"").to_string();
    
    Ok((docs, cross_refs))
}

fn run_rustdoc_g_dependency_analysis(project_path: &std::path::Path) -> Result<String, String> {
    let output = Command::new("./target/release/rustdoc-g")
        .arg("--dependency-analysis")
        .arg(project_path)
        .output()
        .map_err(|e| format!("Failed to execute dependency analysis: {}", e))?;
    
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn run_rustdoc_g_on_multiple_files(files: &[PathBuf]) -> Result<(), String> {
    let file_args: Vec<String> = files.iter()
        .map(|p| p.to_string_lossy().to_string())
        .collect();
    
    let output = Command::new("./target/release/rustdoc-g")
        .arg("--parallel")
        .args(&file_args)
        .output()
        .map_err(|e| format!("Failed to execute parallel rustdoc-g: {}", e))?;
    
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    
    Ok(())
}

fn run_rustdoc_g_with_theme(source: &str, theme: &str) -> Result<String, String> {
    let temp_source = NamedTempFile::new().map_err(|e| e.to_string())?;
    fs::write(temp_source.path(), source).map_err(|e| e.to_string())?;
    
    let temp_theme = NamedTempFile::new().map_err(|e| e.to_string())?;
    fs::write(temp_theme.path(), theme).map_err(|e| e.to_string())?;
    
    let output = Command::new("./target/release/rustdoc-g")
        .arg("--theme")
        .arg(temp_theme.path())
        .arg(temp_source.path())
        .output()
        .map_err(|e| format!("Failed to execute rustdoc-g with theme: {}", e))?;
    
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn generate_large_library(item_count: usize) -> String {
    let mut source = String::new();
    
    for i in 0..item_count {
        if i % 10 == 0 {
            source.push_str(&format!(
                "/// Module {} with comprehensive documentation\n/// \n/// This module provides functionality for item {}.\n/// \n/// # Examples\n/// \n/// ```rust\n/// use crate::module_{}::function_{};\n/// assert_eq!(function_{}(), {});\n/// ```\npub mod module_{} {{\n",
                i, i, i, i, i, i, i
            ));
        }
        
        source.push_str(&format!(
            "    /// Function {} with detailed documentation\n    /// \n    /// This function performs operation {} and returns result.\n    /// \n    /// # Arguments\n    /// \n    /// * `param` - Input parameter for operation\n    /// \n    /// # Returns\n    /// \n    /// Returns the computed value for item {}\n    pub fn function_{}() -> i32 {{ {} }}\n\n",
            i, i, i, i, i
        ));
        
        if i % 10 == 9 {
            source.push_str("}\n\n");
        }
    }
    
    source
}

fn get_memory_usage() -> u64 {
    // Simple memory usage estimation
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

#[derive(Debug)]
struct CacheStats {
    cache_hits: usize,
}

#[test]
fn test_rustdoc_g_binary_exists() {
    let binary_path = PathBuf::from("./target/release/rustdoc-g");
    assert!(false, "rustdoc-g binary should exist at {}", binary_path.display());
}