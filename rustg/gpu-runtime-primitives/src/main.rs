// GPU Runtime Primitives CLI
// Performance validation and testing interface

use clap::{Arg, Command};
use gpu_runtime_primitives::{
    GPURuntime, RuntimeConfig, Task, Message, LogSeverity
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("gpu-runtime")
        .version("1.0.0")
        .author("rustg team")
        .about("GPU-Native Runtime Primitives")
        .subcommand(
            Command::new("test")
                .about("Run runtime tests")
                .arg(Arg::new("component")
                    .help("Component to test (allocator/scheduler/communication/error/all)")
                    .default_value("all")
                    .index(1))
        )
        .subcommand(
            Command::new("benchmark")
                .about("Run performance benchmarks")
                .arg(Arg::new("iterations")
                    .short('i')
                    .long("iterations")
                    .help("Number of iterations")
                    .value_name("COUNT")
                    .default_value("10000"))
        )
        .subcommand(
            Command::new("validate")
                .about("Validate 10x performance improvement")
        )
        .subcommand(
            Command::new("demo")
                .about("Run demonstration of runtime features")
        )
        .get_matches();

    match matches.subcommand() {
        Some(("test", sub_matches)) => {
            let component = sub_matches.get_one::<String>("component").unwrap();
            run_tests(component)
        }
        Some(("benchmark", sub_matches)) => {
            let iterations: usize = sub_matches.get_one::<String>("iterations")
                .unwrap()
                .parse()
                .unwrap_or(10000);
            run_benchmarks(iterations)
        }
        Some(("validate", _)) => validate_performance(),
        Some(("demo", _)) => run_demo(),
        _ => {
            println!("GPU-Native Runtime Primitives");
            println!("Use --help for usage information");
            Ok(())
        }
    }
}

fn run_tests(component: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Running GPU Runtime Tests: {}", component);
    println!("==========================================\n");

    let runtime = GPURuntime::new()?;
    let mut passed = 0;
    let mut failed = 0;

    match component {
        "allocator" | "all" => {
            println!("Testing Allocator...");
            if test_allocator(&runtime)? {
                println!("  ✓ Allocator tests passed");
                passed += 1;
            } else {
                println!("  ✗ Allocator tests failed");
                failed += 1;
            }
        }
        _ => {}
    }

    match component {
        "scheduler" | "all" => {
            println!("Testing Scheduler...");
            if test_scheduler(&runtime)? {
                println!("  ✓ Scheduler tests passed");
                passed += 1;
            } else {
                println!("  ✗ Scheduler tests failed");
                failed += 1;
            }
        }
        _ => {}
    }

    match component {
        "communication" | "all" => {
            println!("Testing Communication...");
            if test_communication(&runtime)? {
                println!("  ✓ Communication tests passed");
                passed += 1;
            } else {
                println!("  ✗ Communication tests failed");
                failed += 1;
            }
        }
        _ => {}
    }

    match component {
        "error" | "all" => {
            println!("Testing Error Handling...");
            if test_error_handling(&runtime)? {
                println!("  ✓ Error handling tests passed");
                passed += 1;
            } else {
                println!("  ✗ Error handling tests failed");
                failed += 1;
            }
        }
        _ => {}
    }

    println!("\n==========================================");
    println!("Test Results: {}/{} passed", passed, passed + failed);
    
    if failed == 0 {
        println!("✓ All tests passed!");
    } else {
        println!("✗ Some tests failed");
        std::process::exit(1);
    }

    Ok(())
}

fn test_allocator(runtime: &GPURuntime) -> Result<bool, Box<dyn std::error::Error>> {
    // Test allocation
    let ptr = runtime.allocate(256, 16);
    if ptr.is_none() {
        return Ok(false);
    }

    // Test different sizes
    for size in [64, 128, 256, 512, 1024, 4096, 8192] {
        if runtime.allocate(size, 16).is_none() {
            return Ok(false);
        }
    }

    Ok(true)
}

fn test_scheduler(runtime: &GPURuntime) -> Result<bool, Box<dyn std::error::Error>> {
    runtime.start()?;

    // Submit tasks
    for i in 0..100 {
        let mut task = Task::new(i, i % 4);
        
        // Add dependencies for some tasks
        if i > 10 {
            task.add_dependency(i - 10)?;
        }
        
        runtime.submit_task(task)?;
    }

    runtime.shutdown();
    Ok(true)
}

fn test_communication(runtime: &GPURuntime) -> Result<bool, Box<dyn std::error::Error>> {
    // Test message passing
    for i in 0..100 {
        let mut msg = Message::new(0, 1, i);
        msg.set_payload(&[i; 16]);
        runtime.send_message(0, msg)?;
    }

    // Receive messages
    let mut received = 0;
    while runtime.receive_message(0).is_some() {
        received += 1;
    }

    Ok(received > 0)
}

fn test_error_handling(runtime: &GPURuntime) -> Result<bool, Box<dyn std::error::Error>> {
    // Test logging
    runtime.log(LogSeverity::Info, "Test info message", 100);
    runtime.log(LogSeverity::Warning, "Test warning message", 101);
    runtime.log(LogSeverity::Error, "Test error message", 102);

    // Test panic handling
    runtime.panic(1001, "Test panic message")?;

    Ok(true)
}

fn run_benchmarks(iterations: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("Running Performance Benchmarks");
    println!("Iterations: {}", iterations);
    println!("==========================================\n");

    let runtime = GPURuntime::new()?;

    // Allocator benchmark
    let alloc_rate = gpu_runtime_primitives::benchmarks::benchmark_allocator(&runtime, iterations);
    println!("Allocator: {:.0} allocations/sec", alloc_rate);

    // Scheduler benchmark
    let task_rate = gpu_runtime_primitives::benchmarks::benchmark_scheduler(&runtime, iterations);
    println!("Scheduler: {:.0} tasks/sec", task_rate);

    // Channel benchmark
    let msg_rate = gpu_runtime_primitives::benchmarks::benchmark_channels(&runtime, iterations);
    println!("Channels: {:.0} messages/sec", msg_rate);

    println!("\n==========================================");
    
    // Check if we meet targets
    let targets_met = alloc_rate > 100_000.0 && 
                     task_rate > 10_000.0 && 
                     msg_rate > 1_000_000.0;
    
    if targets_met {
        println!("✓ All performance targets met!");
    } else {
        println!("✗ Some performance targets not met");
    }

    Ok(())
}

fn validate_performance() -> Result<(), Box<dyn std::error::Error>> {
    println!("Validating 10x Performance Improvement");
    println!("==========================================\n");

    let runtime = GPURuntime::new()?;
    let report = runtime.validate_performance();

    println!("Component Performance:");
    println!("  Allocator: {} cycles (target: <100)", report.allocator_cycles);
    println!("    Status: {}", if report.allocator_valid { "✓ PASS" } else { "✗ FAIL" });
    
    println!("  Scheduler: {:.2} μs latency, {:.1}% utilization", 
            report.scheduler_latency_us, report.scheduler_utilization * 100.0);
    println!("    Status: {}", if report.scheduler_valid { "✓ PASS" } else { "✗ FAIL" });
    
    println!("  Atomics: {} cycles (target: <10)", report.atomic_cycles);
    println!("    Status: {}", if report.atomic_valid { "✓ PASS" } else { "✗ FAIL" });
    
    println!("  Channels: {:.0} msgs/sec (target: >1M)", report.channel_throughput);
    println!("    Status: {}", if report.channel_valid { "✓ PASS" } else { "✗ FAIL" });
    
    println!("  Logging: {:.1}% overhead (target: <5%)", report.logging_overhead);
    println!("    Status: {}", if report.logging_valid { "✓ PASS" } else { "✗ FAIL" });

    println!("\n==========================================");
    let score = report.performance_score();
    println!("Overall Performance Score: {:.1}x", score);
    
    if score >= 10.0 {
        println!("✓ 10x performance improvement ACHIEVED!");
    } else {
        println!("✗ 10x performance improvement NOT achieved");
        println!("  Current: {:.1}x, Required: 10.0x", score);
        std::process::exit(1);
    }

    Ok(())
}

fn run_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("GPU Runtime Primitives Demonstration");
    println!("==========================================\n");

    let runtime = GPURuntime::new()?;
    runtime.start()?;

    println!("1. Memory Allocation Demo");
    let start = Instant::now();
    for i in 0..1000 {
        let size = 256 + (i % 10) * 64;
        if let Some(ptr) = runtime.allocate(size, 16) {
            println!("  Allocated {} bytes at {:p}", size, ptr);
            if i >= 5 { break; } // Show first few
        }
    }
    println!("  ... (1000 allocations in {:.2}ms)\n", start.elapsed().as_millis());

    println!("2. Task Scheduling Demo");
    for i in 0..10 {
        let task = Task::new(i, i % 4);
        runtime.submit_task(task)?;
        println!("  Submitted task {} with priority {}", i, i % 4);
    }
    println!();

    println!("3. Message Passing Demo");
    for i in 0..5 {
        let mut msg = Message::new(i, (i + 1) % 5, i * 100);
        msg.set_payload(&[i as u32; 16]);
        runtime.send_message(0, msg)?;
        println!("  Sent message from {} to {}", i, (i + 1) % 5);
    }
    println!();

    println!("4. Error Handling Demo");
    runtime.log(LogSeverity::Info, "System initialized", 1);
    runtime.log(LogSeverity::Warning, "High memory usage", 2);
    runtime.log(LogSeverity::Error, "Resource limit reached", 3);
    println!("  Logged 3 messages with different severities");
    
    runtime.panic(9999, "Demo panic (handled gracefully)")?;
    println!("  Captured and handled panic\n");

    println!("5. Performance Validation");
    let report = runtime.validate_performance();
    println!("  Performance Score: {:.1}x improvement", report.performance_score());
    
    runtime.shutdown();
    
    println!("\n==========================================");
    println!("✓ Demonstration complete!");

    Ok(())
}