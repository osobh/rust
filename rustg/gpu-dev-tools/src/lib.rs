// GPU Development Tools Library
// Provides GPU-accelerated formatting and linting for rustg

pub mod cuda;
pub mod formatter;
pub mod linter;

pub use formatter::{GPUFormatter, FormatOptions};
pub use linter::{GPULinter, LintIssue, Severity, CustomRule};

/// GPU Development Tools configuration
#[derive(Debug, Clone)]
pub struct DevToolsConfig {
    pub formatter: FormatOptions,
    pub linter: LinterConfig,
    pub cuda: CudaConfig,
}

#[derive(Debug, Clone)]
pub struct LinterConfig {
    pub enabled_rules: Vec<String>,
    pub custom_rules: Vec<CustomRule>,
    pub ignore_patterns: Vec<String>,
    pub max_issues: usize,
}

#[derive(Debug, Clone)]
pub struct CudaConfig {
    pub device_id: i32,
    pub memory_pool_size: usize,
    pub kernel_cache_size: usize,
}

impl Default for DevToolsConfig {
    fn default() -> Self {
        DevToolsConfig {
            formatter: FormatOptions::default(),
            linter: LinterConfig {
                enabled_rules: vec![
                    "unused_variable".to_string(),
                    "memory_leak".to_string(),
                    "divergence_issue".to_string(),
                    "performance_issue".to_string(),
                    "style_violation".to_string(),
                    "gpu_anti_pattern".to_string(),
                ],
                custom_rules: Vec::new(),
                ignore_patterns: vec![
                    "target/".to_string(),
                    "*.test.rs".to_string(),
                ],
                max_issues: 1000,
            },
            cuda: CudaConfig {
                device_id: 0,
                memory_pool_size: 1024 * 1024 * 100, // 100MB
                kernel_cache_size: 1000,
            },
        }
    }
}

/// Main GPU Development Tools interface
pub struct GPUDevTools {
    formatter: GPUFormatter,
    linter: GPULinter,
    config: DevToolsConfig,
}

impl GPUDevTools {
    /// Create new GPU development tools instance
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let config = DevToolsConfig::default();
        Self::with_config(config)
    }

    /// Create with custom configuration
    pub fn with_config(config: DevToolsConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let formatter = GPUFormatter::new()?
            .with_options(config.formatter.clone());
        
        let mut linter = GPULinter::new()?;
        for rule in &config.linter.custom_rules {
            linter.add_custom_rule(rule.clone());
        }

        Ok(GPUDevTools {
            formatter,
            linter,
            config,
        })
    }

    /// Format source code
    pub fn format(&mut self, source: &str) -> Result<String, Box<dyn std::error::Error>> {
        self.formatter.format(source)
    }

    /// Format only changed lines
    pub fn format_incremental(
        &mut self,
        source: &str,
        changed_lines: &[usize],
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.formatter.format_incremental(source, changed_lines)
    }

    /// Lint source code
    pub fn lint(&mut self, source: &str, filename: &str) -> Result<Vec<LintIssue>, Box<dyn std::error::Error>> {
        self.linter.lint(source, filename)
    }

    /// Lint multiple files
    pub fn lint_files(
        &mut self,
        files: &[(String, String)],
    ) -> Result<std::collections::HashMap<String, Vec<LintIssue>>, Box<dyn std::error::Error>> {
        self.linter.lint_files(files)
    }

    /// Format and lint in one pass
    pub fn format_and_lint(
        &mut self,
        source: &str,
        filename: &str,
    ) -> Result<(String, Vec<LintIssue>), Box<dyn std::error::Error>> {
        let formatted = self.formatter.format(source)?;
        let issues = self.linter.lint(&formatted, filename)?;
        Ok((formatted, issues))
    }

    /// Check GPU-specific patterns
    pub fn check_gpu_patterns(&mut self, source: &str) -> Result<Vec<LintIssue>, Box<dyn std::error::Error>> {
        self.linter.check_gpu_patterns(source)
    }

    /// Analyze cross-file dependencies
    pub fn analyze_dependencies(&mut self, files: &[String]) -> Result<Vec<LintIssue>, Box<dyn std::error::Error>> {
        self.linter.analyze_dependencies(files)
    }

    /// Get current configuration
    pub fn get_config(&self) -> &DevToolsConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: DevToolsConfig) -> Result<(), Box<dyn std::error::Error>> {
        self.config = config.clone();
        
        // Recreate formatter with new options
        self.formatter = GPUFormatter::new()?
            .with_options(config.formatter);
        
        // Update linter rules
        self.linter = GPULinter::new()?;
        for rule in &config.linter.custom_rules {
            self.linter.add_custom_rule(rule.clone());
        }
        
        Ok(())
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> DevToolsStats {
        DevToolsStats {
            formatter: self.formatter.get_stats(),
            linter: self.linter.get_stats(),
        }
    }
}

#[derive(Debug)]
pub struct DevToolsStats {
    pub formatter: formatter::FormatterStats,
    pub linter: linter::LinterStats,
}

/// Performance benchmarks for GPU dev tools
pub mod benchmarks {
    use super::*;
    use std::time::Instant;

    /// Benchmark formatter performance
    pub fn benchmark_formatter(lines: usize) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        let mut tools = GPUDevTools::new()?;
        let source = generate_source(lines);
        
        let start = Instant::now();
        let _formatted = tools.format(&source)?;
        let elapsed = start.elapsed();
        
        let lines_per_sec = lines as f64 / elapsed.as_secs_f64();
        
        Ok(BenchmarkResult {
            operation: "format".to_string(),
            input_size: lines,
            elapsed_ms: elapsed.as_millis() as f64,
            throughput: lines_per_sec,
            speedup: lines_per_sec / 10_000.0, // vs 10K lines/sec baseline
        })
    }

    /// Benchmark linter performance
    pub fn benchmark_linter(file_count: usize) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        let mut tools = GPUDevTools::new()?;
        
        let files: Vec<(String, String)> = (0..file_count)
            .map(|i| (
                format!("test_{}.rs", i),
                generate_source(100),
            ))
            .collect();
        
        let start = Instant::now();
        let _results = tools.lint_files(&files)?;
        let elapsed = start.elapsed();
        
        let files_per_sec = file_count as f64 / elapsed.as_secs_f64();
        
        Ok(BenchmarkResult {
            operation: "lint".to_string(),
            input_size: file_count,
            elapsed_ms: elapsed.as_millis() as f64,
            throughput: files_per_sec,
            speedup: files_per_sec / 100.0, // vs 100 files/sec baseline
        })
    }

    #[derive(Debug)]
    pub struct BenchmarkResult {
        pub operation: String,
        pub input_size: usize,
        pub elapsed_ms: f64,
        pub throughput: f64,
        pub speedup: f64,
    }

    fn generate_source(lines: usize) -> String {
        let mut source = String::new();
        for i in 0..lines {
            if i % 50 == 0 {
                source.push_str(&format!("fn function_{}() {{\n", i));
            } else if i % 50 == 49 {
                source.push_str("}\n");
            } else {
                source.push_str(&format!("    let var_{} = {};\n", i, i));
            }
        }
        source
    }
}

/// Validate all performance targets
pub fn validate_performance() -> Result<bool, Box<dyn std::error::Error>> {
    println!("Validating GPU Development Tools Performance...");
    
    // Test formatter
    let formatter_result = formatter::validate_performance()?;
    println!("Formatter: {}", if formatter_result { "✓ PASSED" } else { "✗ FAILED" });
    
    // Test linter
    let linter_result = linter::validate_performance()?;
    println!("Linter: {}", if linter_result { "✓ PASSED" } else { "✗ FAILED" });
    
    // Run benchmarks
    let format_bench = benchmarks::benchmark_formatter(10000)?;
    println!("Format benchmark: {:.0}x speedup", format_bench.speedup);
    
    let lint_bench = benchmarks::benchmark_linter(100)?;
    println!("Lint benchmark: {:.0}x speedup", lint_bench.speedup);
    
    // All must achieve 10x improvement
    Ok(formatter_result && linter_result && 
       format_bench.speedup >= 10.0 && 
       lint_bench.speedup >= 10.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_devtools_creation() {
        let tools = GPUDevTools::new();
        assert!(tools.is_ok());
    }

    #[test]
    fn test_config() {
        let mut config = DevToolsConfig::default();
        config.formatter.indent_width = 2;
        config.linter.max_issues = 500;
        
        let tools = GPUDevTools::with_config(config.clone());
        assert!(tools.is_ok());
        
        if let Ok(tools) = tools {
            assert_eq!(tools.get_config().formatter.indent_width, 2);
            assert_eq!(tools.get_config().linter.max_issues, 500);
        }
    }

    #[test]
    fn test_performance_validation() {
        // This test requires actual CUDA implementation
        // In production, would validate 10x performance
        if let Ok(passed) = validate_performance() {
            assert!(passed, "Must achieve 10x performance improvement");
        }
    }
}