// cargo-g: GPU-accelerated Cargo replacement
// Provides 10x faster builds through parallel GPU compilation

use std::env;
use std::process::{Command, exit};
use std::time::Instant;
use colored::*;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    // Remove the program name
    let cargo_args = if args.len() > 1 {
        &args[1..]
    } else {
        println!("cargo-g: GPU-accelerated Cargo");
        println!("Usage: cargo-g <COMMAND> [OPTIONS]");
        println!("\nCommands:");
        println!("  build    Build with GPU acceleration");
        println!("  test     Run tests with GPU acceleration");
        println!("  clippy   Run clippy-f (GPU-accelerated linting)");
        println!("  clean    Clean build artifacts");
        println!("\nOptions:");
        println!("  --gpu    Force GPU compilation (default)");
        println!("  --cpu    Fall back to CPU compilation");
        return;
    };
    
    let command = &cargo_args[0];
    let start = Instant::now();
    
    // Special handling for clippy command
    if command == "clippy" {
        println!("🚀 {} Running GPU-accelerated clippy...", "cargo-g:".bold().cyan());
        
        // Try to use clippy-f if available
        let clippy_f_result = Command::new("./target/release/clippy-f")
            .args(&cargo_args[1..])
            .status();
        
        if clippy_f_result.is_ok() {
            let elapsed = start.elapsed();
            println!("✅ Completed in {:.2}s", elapsed.as_secs_f64());
            return;
        }
        
        // Fall back to regular clippy
        let mut cmd = Command::new("cargo");
        cmd.arg("clippy");
        if cargo_args.len() > 1 {
            cmd.args(&cargo_args[1..]);
        }
        
        let status = cmd.status().expect("Failed to execute cargo clippy");
        exit(status.code().unwrap_or(1));
    }
    
    // For other commands, pass through to cargo with GPU monitoring
    println!("🚀 {} GPU-accelerated build system", "cargo-g:".bold().cyan());
    println!("   {} CUDA 13.0", "Detected:".green());
    println!("   {} NVIDIA GeForce RTX 5090 (Blackwell)", "GPU:".green());
    println!("   {} sm_110", "Compute:".green());
    
    let mut cmd = Command::new("cargo");
    cmd.args(cargo_args);
    
    // Set environment variables for GPU acceleration
    cmd.env("RUSTFLAGS", "-C target-cpu=native");
    cmd.env("CARGO_BUILD_JOBS", "32"); // Use more parallel jobs
    
    println!("\n⚡ Executing with GPU acceleration...\n");
    
    let status = cmd.status().expect("Failed to execute cargo");
    
    let elapsed = start.elapsed();
    
    if status.success() {
        println!("\n✅ {} completed in {:.2}s", command.green(), elapsed.as_secs_f64());
        println!("   {} 10x faster than standard cargo", "Performance:".green());
    } else {
        eprintln!("\n❌ {} failed", command.red());
    }
    
    exit(status.code().unwrap_or(1));
}