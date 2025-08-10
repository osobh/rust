// GPU Test Harness - GPU-Native Testing Framework
// Part of rustg ProjectB Phase 1

pub mod discovery;
pub mod assertion;
pub mod golden;
pub mod executor;
pub mod cuda;

use anyhow::Result;
use std::path::Path;

// Re-export main types
pub use discovery::{TestDiscovery, TestMetadata};
pub use assertion::{Assertion, AssertionResult};
pub use golden::{GoldenManager, ComparisonResult};
pub use executor::{TestExecutor, ExecutionResult};

// Test harness configuration
#[derive(Debug, Clone)]
pub struct HarnessConfig {
    pub test_directory: String,
    pub golden_directory: String,
    pub output_directory: String,
    pub parallel_tests: usize,
    pub timeout_ms: u32,
    pub multi_gpu: bool,
    pub performance_tracking: bool,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            test_directory: "tests".to_string(),
            golden_directory: "golden".to_string(),
            output_directory: "output".to_string(),
            parallel_tests: 1024,
            timeout_ms: 5000,
            multi_gpu: false,
            performance_tracking: true,
        }
    }
}

// Main test harness
pub struct GpuTestHarness {
    config: HarnessConfig,
    discovery: TestDiscovery,
    executor: TestExecutor,
    golden_manager: GoldenManager,
}

impl GpuTestHarness {
    pub fn new(config: HarnessConfig) -> Result<Self> {
        let discovery = TestDiscovery::new(&config.test_directory)?;
        let executor = TestExecutor::new(config.parallel_tests, config.multi_gpu)?;
        let golden_manager = GoldenManager::new(&config.golden_directory)?;
        
        Ok(Self {
            config,
            discovery,
            executor,
            golden_manager,
        })
    }
    
    // Discover all tests
    pub fn discover_tests(&mut self) -> Result<Vec<TestMetadata>> {
        self.discovery.discover_all()
    }
    
    // Run tests with specified filter
    pub fn run_tests(&mut self, filter: Option<&str>) -> Result<Vec<ExecutionResult>> {
        let tests = if let Some(f) = filter {
            self.discovery.discover_filtered(f)?
        } else {
            self.discovery.discover_all()?
        };
        
        println!("🚀 Running {} tests on GPU", tests.len());
        
        // Execute tests in parallel on GPU
        let results = self.executor.execute_parallel(&tests)?;
        
        // Validate against golden outputs
        for (test, result) in tests.iter().zip(results.iter()) {
            if self.golden_manager.has_golden(&test.name) {
                let comparison = self.golden_manager.compare(
                    &test.name,
                    &result.output_data
                )?;
                
                if !comparison.matched {
                    println!("❌ Golden mismatch for {}: {}",
                            test.name, comparison.details);
                }
            }
        }
        
        Ok(results)
    }
    
    // Run benchmarks
    pub fn run_benchmarks(&mut self) -> Result<Vec<ExecutionResult>> {
        let benchmarks = self.discovery.discover_benchmarks()?;
        println!("⚡ Running {} benchmarks on GPU", benchmarks.len());
        
        self.executor.execute_benchmarks(&benchmarks)
    }
    
    // Update golden outputs
    pub fn update_golden(&mut self, test_name: &str, data: &[u8]) -> Result<()> {
        self.golden_manager.update(test_name, data)
    }
    
    // Get performance metrics
    pub fn get_metrics(&self) -> PerformanceMetrics {
        self.executor.get_metrics()
    }
}

// Performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub tests_per_second: f32,
    pub total_tests_run: usize,
    pub total_time_ms: f32,
    pub gpu_utilization: f32,
    pub memory_used_mb: usize,
}

// Test attribute macros (will be implemented via proc macros)
pub use gpu_test_harness_macros::gpu_test;
pub use gpu_test_harness_macros::gpu_benchmark;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_harness_creation() {
        let config = HarnessConfig::default();
        // Will test once CUDA setup is complete
        // let harness = GpuTestHarness::new(config);
        // assert!(harness.is_ok());
    }
}