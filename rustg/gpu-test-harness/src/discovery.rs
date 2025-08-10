// Test Discovery Module - GPU-aware test detection
// Implements test discovery as validated by CUDA tests

use anyhow::{Result, Context};
use walkdir::WalkDir;
use regex::Regex;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use dashmap::DashMap;
use rayon::prelude::*;

// Test metadata matching CUDA structure
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TestMetadata {
    pub name: String,
    pub category: String,
    pub is_benchmark: bool,
    pub requires_multi_gpu: bool,
    pub expected_runtime_ms: u32,
    pub min_compute_capability: u32,
    pub file_path: PathBuf,
    pub line_number: usize,
}

// Test discovery engine
pub struct TestDiscovery {
    test_directory: PathBuf,
    discovered_tests: DashMap<String, TestMetadata>,
    category_index: DashMap<String, Vec<String>>,
}

impl TestDiscovery {
    pub fn new(test_dir: &str) -> Result<Self> {
        let test_directory = PathBuf::from(test_dir);
        if !test_directory.exists() {
            std::fs::create_dir_all(&test_directory)?;
        }
        
        Ok(Self {
            test_directory,
            discovered_tests: DashMap::new(),
            category_index: DashMap::new(),
        })
    }
    
    // Discover all tests in parallel
    pub fn discover_all(&self) -> Result<Vec<TestMetadata>> {
        let test_files: Vec<PathBuf> = WalkDir::new(&self.test_directory)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path().extension()
                    .map_or(false, |ext| ext == "rs" || ext == "cu")
            })
            .map(|e| e.path().to_path_buf())
            .collect();
        
        // Parse files in parallel for test discovery
        let discovered: Vec<Vec<TestMetadata>> = test_files
            .par_iter()
            .map(|file| self.parse_file_for_tests(file))
            .collect::<Result<Vec<_>>>()?;
        
        // Flatten and store
        let all_tests: Vec<TestMetadata> = discovered
            .into_iter()
            .flatten()
            .collect();
        
        // Index by category
        for test in &all_tests {
            self.discovered_tests.insert(test.name.clone(), test.clone());
            
            self.category_index
                .entry(test.category.clone())
                .or_insert_with(Vec::new)
                .push(test.name.clone());
        }
        
        Ok(all_tests)
    }
    
    // Discover tests matching filter
    pub fn discover_filtered(&self, filter: &str) -> Result<Vec<TestMetadata>> {
        let regex = Regex::new(filter)
            .with_context(|| format!("Invalid filter regex: {}", filter))?;
        
        let all_tests = self.discover_all()?;
        
        Ok(all_tests
            .into_iter()
            .filter(|test| regex.is_match(&test.name) || regex.is_match(&test.category))
            .collect())
    }
    
    // Discover tests by category
    pub fn discover_by_category(&self, category: &str) -> Result<Vec<TestMetadata>> {
        if let Some(test_names) = self.category_index.get(category) {
            Ok(test_names
                .iter()
                .filter_map(|name| {
                    self.discovered_tests.get(name).map(|t| t.clone())
                })
                .collect())
        } else {
            // Discover first if not cached
            self.discover_all()?;
            self.discover_by_category(category)
        }
    }
    
    // Discover benchmarks
    pub fn discover_benchmarks(&self) -> Result<Vec<TestMetadata>> {
        let all_tests = self.discover_all()?;
        Ok(all_tests
            .into_iter()
            .filter(|test| test.is_benchmark)
            .collect())
    }
    
    // Discover multi-GPU tests
    pub fn discover_multi_gpu(&self) -> Result<Vec<TestMetadata>> {
        let all_tests = self.discover_all()?;
        Ok(all_tests
            .into_iter()
            .filter(|test| test.requires_multi_gpu)
            .collect())
    }
    
    // Parse a file for test definitions
    fn parse_file_for_tests(&self, file_path: &Path) -> Result<Vec<TestMetadata>> {
        let content = std::fs::read_to_string(file_path)?;
        let mut tests = Vec::new();
        
        // Look for test attributes
        let test_regex = Regex::new(r#"#\[gpu_test(?:\((.*?)\))?\]"#)?;
        let bench_regex = Regex::new(r#"#\[gpu_benchmark(?:\((.*?)\))?\]"#)?;
        let fn_regex = Regex::new(r#"(?:pub\s+)?(?:async\s+)?fn\s+(\w+)"#)?;
        
        let lines: Vec<&str> = content.lines().collect();
        
        for (line_num, line) in lines.iter().enumerate() {
            // Check for test attribute
            if test_regex.is_match(line) || bench_regex.is_match(line) {
                let is_benchmark = bench_regex.is_match(line);
                
                // Parse attributes
                let mut category = "unit".to_string();
                let mut multi_gpu = false;
                let mut min_cc = 30; // Default CC 3.0
                let mut expected_ms = 100;
                
                // Extract attribute parameters
                if let Some(caps) = test_regex.captures(line)
                    .or_else(|| bench_regex.captures(line)) 
                {
                    if let Some(params) = caps.get(1) {
                        // Parse parameters (simplified)
                        if params.as_str().contains("multi_gpu") {
                            multi_gpu = true;
                        }
                        if params.as_str().contains("integration") {
                            category = "integration".to_string();
                        }
                        if params.as_str().contains("performance") {
                            category = "performance".to_string();
                        }
                    }
                }
                
                // Find function name on next line
                if line_num + 1 < lines.len() {
                    if let Some(fn_caps) = fn_regex.captures(lines[line_num + 1]) {
                        let fn_name = fn_caps.get(1).unwrap().as_str().to_string();
                        
                        tests.push(TestMetadata {
                            name: fn_name,
                            category,
                            is_benchmark,
                            requires_multi_gpu: multi_gpu,
                            expected_runtime_ms: expected_ms,
                            min_compute_capability: min_cc,
                            file_path: file_path.to_path_buf(),
                            line_number: line_num + 1,
                        });
                    }
                }
            }
        }
        
        Ok(tests)
    }
    
    // Check compute capability filter
    pub fn filter_by_capability(&self, tests: Vec<TestMetadata>, cc: u32) 
        -> Vec<TestMetadata> 
    {
        tests.into_iter()
            .filter(|test| test.min_compute_capability <= cc)
            .collect()
    }
    
    // Get test count by category
    pub fn get_category_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for entry in self.category_index.iter() {
            counts.insert(entry.key().clone(), entry.value().len());
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;
    
    #[test]
    fn test_discovery_creation() {
        let temp_dir = TempDir::new().unwrap();
        let discovery = TestDiscovery::new(temp_dir.path().to_str().unwrap());
        assert!(discovery.is_ok());
    }
    
    #[test]
    fn test_file_parsing() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.rs");
        
        fs::write(&test_file, r#"
#[gpu_test]
fn test_vector_add() {
    // Test code
}

#[gpu_benchmark]
fn bench_matrix_mul() {
    // Benchmark code
}

#[gpu_test(multi_gpu)]
fn test_multi_gpu_transfer() {
    // Multi-GPU test
}
        "#).unwrap();
        
        let discovery = TestDiscovery::new(temp_dir.path().to_str().unwrap()).unwrap();
        let tests = discovery.parse_file_for_tests(&test_file).unwrap();
        
        assert_eq!(tests.len(), 3);
        assert_eq!(tests[0].name, "test_vector_add");
        assert!(!tests[0].is_benchmark);
        assert_eq!(tests[1].name, "bench_matrix_mul");
        assert!(tests[1].is_benchmark);
        assert_eq!(tests[2].name, "test_multi_gpu_transfer");
        assert!(tests[2].requires_multi_gpu);
    }
}