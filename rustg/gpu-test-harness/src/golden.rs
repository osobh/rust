// Golden Output System - Reference output management
// Implements golden validation as tested by CUDA

use anyhow::{Result, Context, bail};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use sha2::{Sha256, Digest};
use serde::{Serialize, Deserialize};
use dashmap::DashMap;

// Golden metadata matching CUDA structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenMetadata {
    pub test_name: String,
    pub version: String,
    pub output_size: usize,
    pub data_type: DataType,
    pub tolerance: f32,
    pub platform_specific: bool,
    pub compute_capability: u32,
    pub hash: String,
    pub created_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataType {
    Int32,
    Float32,
    Float64,
    Custom(String),
}

// Comparison result matching CUDA structure
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub matched: bool,
    pub total_elements: usize,
    pub mismatched_elements: usize,
    pub first_mismatch_index: Option<usize>,
    pub max_deviation: f32,
    pub avg_deviation: f32,
    pub details: String,
}

// Golden output manager
pub struct GoldenManager {
    golden_dir: PathBuf,
    metadata_cache: DashMap<String, GoldenMetadata>,
    golden_cache: DashMap<String, Vec<u8>>,
}

impl GoldenManager {
    pub fn new(golden_dir: &str) -> Result<Self> {
        let golden_dir = PathBuf::from(golden_dir);
        if !golden_dir.exists() {
            std::fs::create_dir_all(&golden_dir)?;
        }
        
        let manager = Self {
            golden_dir,
            metadata_cache: DashMap::new(),
            golden_cache: DashMap::new(),
        };
        
        // Load existing golden metadata
        manager.load_metadata()?;
        
        Ok(manager)
    }
    
    // Check if golden exists for test
    pub fn has_golden(&self, test_name: &str) -> bool {
        self.metadata_cache.contains_key(test_name)
    }
    
    // Compare output with golden
    pub fn compare(&self, test_name: &str, output: &[u8]) -> Result<ComparisonResult> {
        let golden = self.load_golden(test_name)?;
        let metadata = self.metadata_cache.get(test_name)
            .context("Golden metadata not found")?;
        
        match metadata.data_type {
            DataType::Int32 => self.compare_ints(output, &golden),
            DataType::Float32 => self.compare_floats(output, &golden, metadata.tolerance),
            DataType::Float64 => self.compare_doubles(output, &golden, metadata.tolerance),
            DataType::Custom(_) => self.compare_bytes(output, &golden),
        }
    }
    
    // Compare integer arrays
    fn compare_ints(&self, output: &[u8], golden: &[u8]) -> Result<ComparisonResult> {
        let output_ints = unsafe {
            std::slice::from_raw_parts(
                output.as_ptr() as *const i32,
                output.len() / 4
            )
        };
        
        let golden_ints = unsafe {
            std::slice::from_raw_parts(
                golden.as_ptr() as *const i32,
                golden.len() / 4
            )
        };
        
        if output_ints.len() != golden_ints.len() {
            return Ok(ComparisonResult {
                matched: false,
                total_elements: output_ints.len(),
                mismatched_elements: output_ints.len(),
                first_mismatch_index: Some(0),
                max_deviation: f32::MAX,
                avg_deviation: f32::MAX,
                details: format!("Size mismatch: {} vs {}", 
                               output_ints.len(), golden_ints.len()),
            });
        }
        
        let mut mismatches = 0;
        let mut first_mismatch = None;
        
        for (i, (out, gold)) in output_ints.iter().zip(golden_ints.iter()).enumerate() {
            if out != gold {
                mismatches += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some(i);
                }
            }
        }
        
        Ok(ComparisonResult {
            matched: mismatches == 0,
            total_elements: output_ints.len(),
            mismatched_elements: mismatches,
            first_mismatch_index: first_mismatch,
            max_deviation: 0.0,
            avg_deviation: 0.0,
            details: if mismatches == 0 {
                "Exact match".to_string()
            } else {
                format!("{} mismatches found", mismatches)
            },
        })
    }
    
    // Compare float arrays with tolerance
    fn compare_floats(&self, output: &[u8], golden: &[u8], tolerance: f32) 
        -> Result<ComparisonResult> 
    {
        let output_floats = unsafe {
            std::slice::from_raw_parts(
                output.as_ptr() as *const f32,
                output.len() / 4
            )
        };
        
        let golden_floats = unsafe {
            std::slice::from_raw_parts(
                golden.as_ptr() as *const f32,
                golden.len() / 4
            )
        };
        
        if output_floats.len() != golden_floats.len() {
            return Ok(ComparisonResult {
                matched: false,
                total_elements: output_floats.len(),
                mismatched_elements: output_floats.len(),
                first_mismatch_index: Some(0),
                max_deviation: f32::MAX,
                avg_deviation: f32::MAX,
                details: format!("Size mismatch: {} vs {}", 
                               output_floats.len(), golden_floats.len()),
            });
        }
        
        let mut mismatches = 0;
        let mut first_mismatch = None;
        let mut max_dev = 0.0f32;
        let mut sum_dev = 0.0f32;
        
        for (i, (out, gold)) in output_floats.iter().zip(golden_floats.iter()).enumerate() {
            let deviation = (out - gold).abs();
            sum_dev += deviation;
            max_dev = max_dev.max(deviation);
            
            if deviation > tolerance {
                mismatches += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some(i);
                }
            }
        }
        
        let avg_dev = sum_dev / output_floats.len() as f32;
        
        Ok(ComparisonResult {
            matched: mismatches == 0,
            total_elements: output_floats.len(),
            mismatched_elements: mismatches,
            first_mismatch_index: first_mismatch,
            max_deviation: max_dev,
            avg_deviation: avg_dev,
            details: if mismatches == 0 {
                format!("Within tolerance (max dev: {:.6})", max_dev)
            } else {
                format!("{} values exceed tolerance {}", mismatches, tolerance)
            },
        })
    }
    
    // Compare double arrays
    fn compare_doubles(&self, output: &[u8], golden: &[u8], tolerance: f32) 
        -> Result<ComparisonResult> 
    {
        let output_doubles = unsafe {
            std::slice::from_raw_parts(
                output.as_ptr() as *const f64,
                output.len() / 8
            )
        };
        
        let golden_doubles = unsafe {
            std::slice::from_raw_parts(
                golden.as_ptr() as *const f64,
                golden.len() / 8
            )
        };
        
        let tolerance_f64 = tolerance as f64;
        let mut mismatches = 0;
        let mut first_mismatch = None;
        let mut max_dev = 0.0f64;
        let mut sum_dev = 0.0f64;
        
        for (i, (out, gold)) in output_doubles.iter().zip(golden_doubles.iter()).enumerate() {
            let deviation = (out - gold).abs();
            sum_dev += deviation;
            max_dev = max_dev.max(deviation);
            
            if deviation > tolerance_f64 {
                mismatches += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some(i);
                }
            }
        }
        
        Ok(ComparisonResult {
            matched: mismatches == 0,
            total_elements: output_doubles.len(),
            mismatched_elements: mismatches,
            first_mismatch_index: first_mismatch,
            max_deviation: max_dev as f32,
            avg_deviation: (sum_dev / output_doubles.len() as f64) as f32,
            details: if mismatches == 0 {
                "Within tolerance".to_string()
            } else {
                format!("{} values exceed tolerance", mismatches)
            },
        })
    }
    
    // Byte-wise comparison
    fn compare_bytes(&self, output: &[u8], golden: &[u8]) -> Result<ComparisonResult> {
        let matched = output == golden;
        
        Ok(ComparisonResult {
            matched,
            total_elements: output.len(),
            mismatched_elements: if matched { 0 } else { 1 },
            first_mismatch_index: if matched { None } else { Some(0) },
            max_deviation: 0.0,
            avg_deviation: 0.0,
            details: if matched {
                "Binary match".to_string()
            } else {
                "Binary mismatch".to_string()
            },
        })
    }
    
    // Update golden output
    pub fn update(&mut self, test_name: &str, data: &[u8]) -> Result<()> {
        // Calculate hash
        let mut hasher = Sha256::new();
        hasher.update(data);
        let hash = format!("{:x}", hasher.finalize());
        
        // Create metadata
        let metadata = GoldenMetadata {
            test_name: test_name.to_string(),
            version: "1.0.0".to_string(),
            output_size: data.len(),
            data_type: DataType::Custom("binary".to_string()),
            tolerance: 0.0,
            platform_specific: false,
            compute_capability: 70, // Default to CC 7.0
            hash,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
        };
        
        // Save golden data
        let golden_path = self.golden_dir.join(format!("{}.golden", test_name));
        std::fs::write(&golden_path, data)?;
        
        // Save metadata
        let meta_path = self.golden_dir.join(format!("{}.meta.json", test_name));
        let meta_json = serde_json::to_string_pretty(&metadata)?;
        std::fs::write(&meta_path, meta_json)?;
        
        // Update caches
        self.metadata_cache.insert(test_name.to_string(), metadata);
        self.golden_cache.insert(test_name.to_string(), data.to_vec());
        
        Ok(())
    }
    
    // Load golden data
    fn load_golden(&self, test_name: &str) -> Result<Vec<u8>> {
        // Check cache first
        if let Some(cached) = self.golden_cache.get(test_name) {
            return Ok(cached.clone());
        }
        
        // Load from disk
        let golden_path = self.golden_dir.join(format!("{}.golden", test_name));
        let data = std::fs::read(&golden_path)
            .with_context(|| format!("Failed to load golden for {}", test_name))?;
        
        // Cache it
        self.golden_cache.insert(test_name.to_string(), data.clone());
        
        Ok(data)
    }
    
    // Load all metadata
    fn load_metadata(&self) -> Result<()> {
        for entry in std::fs::read_dir(&self.golden_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                let content = std::fs::read_to_string(&path)?;
                let metadata: GoldenMetadata = serde_json::from_str(&content)?;
                self.metadata_cache.insert(metadata.test_name.clone(), metadata);
            }
        }
        
        Ok(())
    }
    
    // Generate visual diff
    pub fn generate_diff(&self, output: &[f32], golden: &[f32]) -> Vec<f32> {
        output.iter()
            .zip(golden.iter())
            .map(|(o, g)| (o - g).abs())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_golden_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let manager = GoldenManager::new(temp_dir.path().to_str().unwrap());
        assert!(manager.is_ok());
    }
    
    #[test]
    fn test_int_comparison() {
        let temp_dir = TempDir::new().unwrap();
        let manager = GoldenManager::new(temp_dir.path().to_str().unwrap()).unwrap();
        
        let output = vec![1i32, 2, 3, 4];
        let golden = vec![1i32, 2, 3, 4];
        
        let output_bytes = unsafe {
            std::slice::from_raw_parts(
                output.as_ptr() as *const u8,
                output.len() * 4
            )
        };
        
        let golden_bytes = unsafe {
            std::slice::from_raw_parts(
                golden.as_ptr() as *const u8,
                golden.len() * 4
            )
        };
        
        let result = manager.compare_ints(output_bytes, golden_bytes).unwrap();
        assert!(result.matched);
        assert_eq!(result.mismatched_elements, 0);
    }
}