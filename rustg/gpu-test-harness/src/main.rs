// GPU Test Harness CLI - Command-line interface
// Part of rustg ProjectB Phase 1

use anyhow::{Result, Context};
use clap::{Parser, Subcommand};
use colored::Colorize;
use gpu_test_harness::{GpuTestHarness, HarnessConfig};
use std::time::Instant;

#[derive(Parser)]
#[clap(name = "gpu-test-harness")]
#[clap(about = "GPU-native testing framework for rustg", version)]
struct Cli {
    #[clap(subcommand)]
    command: Command,
    
    /// Enable verbose output
    #[clap(short, long, global = true)]
    verbose: bool,
    
    /// Use multiple GPUs if available
    #[clap(long, global = true)]
    multi_gpu: bool,
    
    /// Number of parallel tests
    #[clap(short = 'j', long, default_value = "1024", global = true)]
    parallel: usize,
}

#[derive(Subcommand)]
enum Command {
    /// Run tests
    Test {
        /// Filter tests by pattern
        #[clap(short, long)]
        filter: Option<String>,
        
        /// Category to run (unit, integration, benchmark)
        #[clap(short, long)]
        category: Option<String>,
        
        /// Update golden outputs
        #[clap(long)]
        update_golden: bool,
        
        /// Show performance metrics
        #[clap(long)]
        metrics: bool,
    },
    
    /// Run benchmarks
    Bench {
        /// Filter benchmarks by pattern
        #[clap(short, long)]
        filter: Option<String>,
        
        /// Number of iterations
        #[clap(short, long, default_value = "10")]
        iterations: usize,
    },
    
    /// Discover available tests
    Discover {
        /// Show detailed information
        #[clap(short, long)]
        detailed: bool,
        
        /// Filter by category
        #[clap(short, long)]
        category: Option<String>,
    },
    
    /// Show GPU information
    Info,
    
    /// Clean test artifacts
    Clean,
}

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();
    
    // Print header
    println!("{}", "╔══════════════════════════════════════╗".bright_cyan());
    println!("{}", "║     GPU Test Harness for rustg      ║".bright_cyan());
    println!("{}", "║  Target: 1000+ tests/second on GPU  ║".bright_cyan());
    println!("{}", "╚══════════════════════════════════════╝".bright_cyan());
    println!();
    
    match cli.command {
        Command::Test { filter, category, update_golden, metrics } => {
            run_tests(&cli, filter, category, update_golden, metrics)?;
        }
        Command::Bench { filter, iterations } => {
            run_benchmarks(&cli, filter, iterations)?;
        }
        Command::Discover { detailed, category } => {
            discover_tests(&cli, detailed, category)?;
        }
        Command::Info => {
            show_gpu_info()?;
        }
        Command::Clean => {
            clean_artifacts()?;
        }
    }
    
    Ok(())
}

fn run_tests(cli: &Cli, filter: Option<String>, category: Option<String>, 
             update_golden: bool, show_metrics: bool) -> Result<()> {
    println!("{} Initializing GPU test harness...", "🚀".bright_green());
    
    let config = HarnessConfig {
        test_directory: "tests".to_string(),
        golden_directory: "golden".to_string(),
        output_directory: "output".to_string(),
        parallel_tests: cli.parallel,
        timeout_ms: 5000,
        multi_gpu: cli.multi_gpu,
        performance_tracking: true,
    };
    
    let mut harness = GpuTestHarness::new(config)
        .context("Failed to initialize test harness")?;
    
    // Discover tests
    let tests = if let Some(cat) = category {
        println!("📂 Running {} tests...", cat.bright_yellow());
        harness.discover_tests()?
            .into_iter()
            .filter(|t| t.category == cat)
            .collect()
    } else {
        harness.discover_tests()?
    };
    
    println!("📊 Found {} tests", tests.len().to_string().bright_cyan());
    
    if tests.is_empty() {
        println!("{} No tests found matching criteria", "⚠️".bright_yellow());
        return Ok(());
    }
    
    // Run tests
    let start = Instant::now();
    let results = harness.run_tests(filter.as_deref())?;
    let elapsed = start.elapsed();
    
    // Process results
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = results.iter().filter(|r| !r.passed).count();
    
    println!();
    println!("{}", "═══════════════════════════════════════".bright_blue());
    
    // Show failures
    if failed > 0 {
        println!("{} {} tests failed:", "❌".bright_red(), failed);
        for result in results.iter().filter(|r| !r.passed) {
            println!("  {} {}: {}", 
                    "✗".red(),
                    result.test_id.bright_white(),
                    result.failure_message.bright_red());
        }
        println!();
    }
    
    // Show summary
    let tests_per_second = (results.len() as f32) / elapsed.as_secs_f32();
    
    println!("{} Test Results:", "📈".bright_green());
    println!("  Total:    {}", results.len().to_string().bright_white());
    println!("  Passed:   {}", passed.to_string().bright_green());
    println!("  Failed:   {}", failed.to_string().bright_red());
    println!("  Time:     {:.2}s", elapsed.as_secs_f32());
    println!("  Speed:    {} tests/second", 
            format!("{:.0}", tests_per_second).bright_cyan());
    
    // Performance validation
    if tests_per_second >= 1000.0 {
        println!();
        println!("{} {} Achieved target performance!", 
                "✅".bright_green(),
                "PASS:".bright_green());
    } else if results.len() < 100 {
        println!();
        println!("{} Run 100+ tests to measure performance", 
                "ℹ️".bright_blue());
    } else {
        println!();
        println!("{} {} Performance below target (1000 tests/s)", 
                "⚠️".bright_yellow(),
                "WARNING:".bright_yellow());
    }
    
    // Show metrics if requested
    if show_metrics {
        let metrics = harness.get_metrics();
        println!();
        println!("{} Performance Metrics:", "📊".bright_cyan());
        println!("  Tests/second:     {:.0}", metrics.tests_per_second);
        println!("  GPU Utilization:  {:.1}%", metrics.gpu_utilization);
        println!("  Memory Used:      {} MB", metrics.memory_used_mb);
    }
    
    if failed > 0 {
        std::process::exit(1);
    }
    
    Ok(())
}

fn run_benchmarks(cli: &Cli, filter: Option<String>, iterations: usize) -> Result<()> {
    println!("{} Running GPU benchmarks...", "⚡".bright_yellow());
    
    let config = HarnessConfig {
        test_directory: "tests".to_string(),
        golden_directory: "golden".to_string(),
        output_directory: "output".to_string(),
        parallel_tests: cli.parallel,
        timeout_ms: 30000,
        multi_gpu: cli.multi_gpu,
        performance_tracking: true,
    };
    
    let mut harness = GpuTestHarness::new(config)?;
    let results = harness.run_benchmarks()?;
    
    println!();
    println!("{} Benchmark Results:", "📊".bright_cyan());
    for result in &results {
        println!("  {}: {:.2}ms", 
                result.test_id.bright_white(),
                result.execution_time_ms);
    }
    
    Ok(())
}

fn discover_tests(cli: &Cli, detailed: bool, category: Option<String>) -> Result<()> {
    println!("{} Discovering tests...", "🔍".bright_blue());
    
    let config = HarnessConfig::default();
    let mut harness = GpuTestHarness::new(config)?;
    let tests = harness.discover_tests()?;
    
    if let Some(cat) = category {
        let filtered: Vec<_> = tests.iter()
            .filter(|t| t.category == cat)
            .collect();
        
        println!("Found {} {} tests:", filtered.len(), cat);
        for test in filtered {
            if detailed {
                println!("  {} {} ({}:{})", 
                        "•".bright_green(),
                        test.name.bright_white(),
                        test.file_path.display(),
                        test.line_number);
            } else {
                println!("  {} {}", "•".bright_green(), test.name);
            }
        }
    } else {
        // Group by category
        let mut by_category = std::collections::HashMap::new();
        for test in &tests {
            by_category.entry(test.category.clone())
                .or_insert_with(Vec::new)
                .push(test);
        }
        
        for (category, tests) in by_category {
            println!();
            println!("{} ({} tests):", 
                    category.bright_yellow(),
                    tests.len());
            
            for test in tests {
                if detailed {
                    println!("  {} {} [{}{}]", 
                            if test.is_benchmark { "📊" } else { "✓" },
                            test.name.bright_white(),
                            if test.requires_multi_gpu { "multi-gpu, " } else { "" },
                            format!("CC {}+", test.min_compute_capability / 10));
                } else {
                    println!("  {} {}", 
                            if test.is_benchmark { "📊" } else { "✓" },
                            test.name);
                }
            }
        }
    }
    
    println!();
    println!("Total: {} tests discovered", tests.len().to_string().bright_cyan());
    
    Ok(())
}

fn show_gpu_info() -> Result<()> {
    use gpu_test_harness::cuda::GpuDevice;
    
    println!("{} GPU Information:", "🖥️".bright_cyan());
    println!();
    
    let devices = GpuDevice::enumerate()
        .context("Failed to enumerate GPU devices")?;
    
    if devices.is_empty() {
        println!("{} No CUDA-capable GPUs found!", "❌".bright_red());
        return Ok(());
    }
    
    for device in devices {
        println!("GPU {}: {}", 
                device.id.to_string().bright_yellow(),
                device.name.bright_white());
        println!("  Compute Capability:  {}.{}", 
                device.compute_capability_major,
                device.compute_capability_minor);
        println!("  Memory:              {} MB", device.total_memory_mb);
        println!("  Multiprocessors:     {}", device.multiprocessor_count);
        println!("  Max Threads/Block:   {}", device.max_threads_per_block);
        println!("  Warp Size:           {}", device.warp_size);
        println!("  Managed Memory:      {}", 
                if device.supports_managed_memory { "✓" } else { "✗" });
        println!("  Concurrent Kernels:  {}", 
                if device.supports_concurrent_kernels { "✓" } else { "✗" });
        println!();
    }
    
    Ok(())
}

fn clean_artifacts() -> Result<()> {
    println!("{} Cleaning test artifacts...", "🧹".bright_blue());
    
    let paths = ["output", "target/test-results", ".test-cache"];
    
    for path in &paths {
        if std::path::Path::new(path).exists() {
            std::fs::remove_dir_all(path)
                .with_context(|| format!("Failed to remove {}", path))?;
            println!("  {} Removed {}", "✓".bright_green(), path);
        }
    }
    
    println!("{} Clean complete", "✅".bright_green());
    
    Ok(())
}