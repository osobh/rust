//! rustg compiler driver
//!
//! Main entry point for the rustg GPU-native Rust compiler.

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

/// GPU-native Rust compiler
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input Rust source file
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Output file (defaults to input with .o extension)
    #[arg(short, long, value_name = "OUTPUT")]
    output: Option<PathBuf>,

    /// Emit intermediate representation
    #[arg(long)]
    emit_ir: bool,

    /// Emit AST in JSON format
    #[arg(long)]
    emit_ast: bool,

    /// Enable CPU fallback for unsupported features
    #[arg(long)]
    cpu_fallback: bool,

    /// Enable performance profiling
    #[arg(long)]
    profile: bool,

    /// Verbosity level (can be repeated)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Number of GPU threads to use
    #[arg(long, default_value = "auto")]
    gpu_threads: String,

    /// GPU memory limit in MB
    #[arg(long, default_value = "4096")]
    gpu_memory_limit: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    setup_logging(args.verbose)?;

    // Initialize the GPU compiler
    rustg::initialize().context("Failed to initialize GPU compiler")?;

    // Set up cleanup on exit
    let _guard = CleanupGuard;

    // Create compiler instance
    let mut compiler = rustg::GpuCompiler::new()
        .with_cpu_fallback(args.cpu_fallback)
        .with_profiling(args.profile)
        .with_gpu_memory_limit(args.gpu_memory_limit * 1024 * 1024);

    // Configure GPU threads
    if args.gpu_threads != "auto" {
        let threads: usize = args.gpu_threads.parse()
            .context("Invalid GPU thread count")?;
        compiler = compiler.with_gpu_threads(threads);
    }

    // Compile the input file
    tracing::info!("Compiling {:?}", args.input);
    let result = compiler.compile_file(&args.input)?;

    // Determine output path
    let output_path = args.output.unwrap_or_else(|| {
        let mut path = args.input.clone();
        path.set_extension("o");
        path
    });

    // Write output
    if args.emit_ast {
        let ast_path = output_path.with_extension("ast.json");
        result.write_ast(&ast_path)?;
        tracing::info!("AST written to {:?}", ast_path);
    }

    if args.emit_ir {
        let ir_path = output_path.with_extension("ir");
        result.write_ir(&ir_path)?;
        tracing::info!("IR written to {:?}", ir_path);
    }

    result.write_object(&output_path)?;
    tracing::info!("Output written to {:?}", output_path);

    // Print performance statistics if profiling was enabled
    if args.profile {
        print_performance_stats(&result);
    }

    Ok(())
}

fn setup_logging(verbosity: u8) -> Result<()> {
    let filter = match verbosity {
        0 => EnvFilter::new("warn"),
        1 => EnvFilter::new("info"),
        2 => EnvFilter::new("debug"),
        _ => EnvFilter::new("trace"),
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .compact()
        .init();

    Ok(())
}

fn print_performance_stats(result: &rustg::CompilationResult) {
    println!("\n=== Performance Statistics ===");
    println!("Total compilation time: {:.2}ms", result.total_time_ms());
    println!("  Parsing:      {:.2}ms ({:.1}x speedup)", 
             result.parsing_time_ms, result.parsing_speedup());
    println!("  Type checking: {:.2}ms ({:.1}x speedup)",
             result.type_check_time_ms, result.type_check_speedup());
    println!("  Code gen:     {:.2}ms ({:.1}x speedup)",
             result.codegen_time_ms, result.codegen_speedup());
    println!("GPU memory used: {}MB", result.gpu_memory_used_mb());
    println!("GPU utilization: {:.1}%", result.gpu_utilization());
}

/// RAII guard for cleanup on exit
struct CleanupGuard;

impl Drop for CleanupGuard {
    fn drop(&mut self) {
        if let Err(e) = rustg::shutdown() {
            eprintln!("Warning: Failed to shutdown GPU compiler: {}", e);
        }
    }
}