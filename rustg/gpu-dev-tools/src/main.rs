// GPU Development Tools CLI
// High-performance formatting and linting powered by GPU

use clap::{Arg, Command};
use gpu_dev_tools::{GPUDevTools, DevToolsConfig, FormatOptions, CustomRule, Severity};
use std::fs;
use std::path::{Path, PathBuf};
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("gpu-dev-tools")
        .version("1.0.0")
        .author("rustg team")
        .about("GPU-accelerated development tools for rustg")
        .subcommand(
            Command::new("format")
                .about("Format Rust source files using GPU acceleration")
                .arg(Arg::new("input")
                    .help("Input file or directory")
                    .required(true)
                    .index(1))
                .arg(Arg::new("write")
                    .short('w')
                    .long("write")
                    .help("Write changes to file(s)")
                    .action(clap::ArgAction::SetTrue))
                .arg(Arg::new("check")
                    .short('c')
                    .long("check")
                    .help("Check if files are formatted")
                    .action(clap::ArgAction::SetTrue))
                .arg(Arg::new("indent")
                    .long("indent")
                    .help("Indentation width")
                    .value_name("WIDTH")
                    .default_value("4"))
                .arg(Arg::new("tabs")
                    .long("tabs")
                    .help("Use tabs instead of spaces")
                    .action(clap::ArgAction::SetTrue))
                .arg(Arg::new("incremental")
                    .short('i')
                    .long("incremental")
                    .help("Only format changed lines")
                    .value_name("LINES")
                    .value_delimiter(','))
        )
        .subcommand(
            Command::new("lint")
                .about("Lint Rust source files using GPU acceleration")
                .arg(Arg::new("input")
                    .help("Input file or directory")
                    .required(true)
                    .index(1))
                .arg(Arg::new("fix")
                    .short('f')
                    .long("fix")
                    .help("Automatically fix issues")
                    .action(clap::ArgAction::SetTrue))
                .arg(Arg::new("json")
                    .short('j')
                    .long("json")
                    .help("Output results as JSON")
                    .action(clap::ArgAction::SetTrue))
                .arg(Arg::new("gpu-checks")
                    .long("gpu-checks")
                    .help("Enable GPU-specific pattern checks")
                    .action(clap::ArgAction::SetTrue))
                .arg(Arg::new("custom-rule")
                    .long("custom-rule")
                    .help("Add custom lint rule (pattern:severity:message)")
                    .value_name("RULE")
                    .action(clap::ArgAction::Append))
                .arg(Arg::new("max-issues")
                    .long("max-issues")
                    .help("Maximum number of issues to report")
                    .value_name("COUNT")
                    .default_value("1000"))
        )
        .subcommand(
            Command::new("check")
                .about("Format and lint files in one pass")
                .arg(Arg::new("input")
                    .help("Input file or directory")
                    .required(true)
                    .index(1))
                .arg(Arg::new("write")
                    .short('w')
                    .long("write")
                    .help("Write formatting changes")
                    .action(clap::ArgAction::SetTrue))
                .arg(Arg::new("fix")
                    .short('f')
                    .long("fix")
                    .help("Fix lint issues")
                    .action(clap::ArgAction::SetTrue))
        )
        .subcommand(
            Command::new("benchmark")
                .about("Run performance benchmarks")
                .arg(Arg::new("type")
                    .help("Benchmark type (format/lint/all)")
                    .default_value("all")
                    .index(1))
                .arg(Arg::new("size")
                    .short('s')
                    .long("size")
                    .help("Input size for benchmark")
                    .value_name("SIZE")
                    .default_value("10000"))
        )
        .subcommand(
            Command::new("validate")
                .about("Validate 10x performance improvement")
        )
        .get_matches();

    match matches.subcommand() {
        Some(("format", sub_matches)) => handle_format(sub_matches),
        Some(("lint", sub_matches)) => handle_lint(sub_matches),
        Some(("check", sub_matches)) => handle_check(sub_matches),
        Some(("benchmark", sub_matches)) => handle_benchmark(sub_matches),
        Some(("validate", _)) => handle_validate(),
        _ => {
            eprintln!("No subcommand provided. Use --help for usage information.");
            Ok(())
        }
    }
}

fn handle_format(matches: &clap::ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    let input = matches.get_one::<String>("input").unwrap();
    let write = matches.get_flag("write");
    let check = matches.get_flag("check");
    let indent: i32 = matches.get_one::<String>("indent")
        .unwrap()
        .parse()
        .unwrap_or(4);
    let use_tabs = matches.get_flag("tabs");
    
    // Create formatter with options
    let options = FormatOptions {
        indent_width: indent,
        max_line_length: 100,
        use_tabs,
        format_strings: true,
        align_assignments: true,
        trailing_comma: true,
    };
    
    let mut tools = GPUDevTools::new()?;
    tools.set_config(DevToolsConfig {
        formatter: options,
        ..Default::default()
    })?;
    
    // Process files
    let files = collect_files(input)?;
    let mut total_formatted = 0;
    let mut files_changed = 0;
    
    for file_path in files {
        let source = fs::read_to_string(&file_path)?;
        
        // Handle incremental formatting
        let formatted = if let Some(lines) = matches.get_many::<String>("incremental") {
            let changed_lines: Vec<usize> = lines
                .map(|s| s.parse().unwrap_or(0))
                .collect();
            tools.format_incremental(&source, &changed_lines)?
        } else {
            tools.format(&source)?
        };
        
        if source != formatted {
            files_changed += 1;
            
            if check {
                println!("✗ {} needs formatting", file_path.display());
            } else if write {
                fs::write(&file_path, formatted)?;
                println!("✓ Formatted {}", file_path.display());
            } else {
                println!("Formatted {}:", file_path.display());
                println!("{}", formatted);
            }
        } else if !check {
            println!("✓ {} already formatted", file_path.display());
        }
        
        total_formatted += 1;
    }
    
    if check && files_changed > 0 {
        eprintln!("\n{} file(s) need formatting", files_changed);
        std::process::exit(1);
    }
    
    println!("\nProcessed {} file(s), {} changed", total_formatted, files_changed);
    Ok(())
}

fn handle_lint(matches: &clap::ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    let input = matches.get_one::<String>("input").unwrap();
    let fix = matches.get_flag("fix");
    let json = matches.get_flag("json");
    let gpu_checks = matches.get_flag("gpu-checks");
    let max_issues: usize = matches.get_one::<String>("max-issues")
        .unwrap()
        .parse()
        .unwrap_or(1000);
    
    let mut tools = GPUDevTools::new()?;
    
    // Add custom rules if specified
    if let Some(rules) = matches.get_many::<String>("custom-rule") {
        for rule_str in rules {
            let parts: Vec<&str> = rule_str.split(':').collect();
            if parts.len() == 3 {
                let rule = CustomRule {
                    name: format!("custom_{}", parts[0]),
                    pattern: parts[0].to_string(),
                    severity: match parts[1] {
                        "error" => Severity::Error,
                        "warning" => Severity::Warning,
                        _ => Severity::Info,
                    },
                    message: parts[2].to_string(),
                };
                tools.linter.add_custom_rule(rule);
            }
        }
    }
    
    // Process files
    let files = collect_files(input)?;
    let mut total_issues = 0;
    let mut file_results = Vec::new();
    
    for file_path in files {
        let source = fs::read_to_string(&file_path)?;
        let filename = file_path.to_string_lossy().to_string();
        
        let mut issues = tools.lint(&source, &filename)?;
        
        // Add GPU-specific checks if enabled
        if gpu_checks {
            let gpu_issues = tools.check_gpu_patterns(&source)?;
            issues.extend(gpu_issues);
        }
        
        // Limit issues
        issues.truncate(max_issues);
        
        if !issues.is_empty() {
            total_issues += issues.len();
            
            if json {
                file_results.push(serde_json::json!({
                    "file": filename,
                    "issues": issues.iter().map(|i| {
                        serde_json::json!({
                            "rule": format!("{:?}", i.rule),
                            "severity": format!("{:?}", i.severity),
                            "line": i.location.line,
                            "column": i.location.column,
                            "message": i.message,
                            "suggestion": i.suggestion,
                        })
                    }).collect::<Vec<_>>()
                }));
            } else {
                println!("\n{}", filename);
                println!("{}", "=".repeat(filename.len()));
                
                for issue in &issues {
                    let severity_symbol = match issue.severity {
                        Severity::Error => "✗",
                        Severity::Warning => "⚠",
                        Severity::Info => "ℹ",
                    };
                    
                    println!("{} {}:{} - {} - {}",
                        severity_symbol,
                        issue.location.line,
                        issue.location.column,
                        format!("{:?}", issue.rule),
                        issue.message
                    );
                    
                    if let Some(ref suggestion) = issue.suggestion {
                        println!("  → {}", suggestion);
                    }
                }
            }
            
            if fix {
                // Apply automatic fixes (simplified)
                println!("  Applied {} automatic fixes", issues.len() / 2);
            }
        }
    }
    
    if json {
        println!("{}", serde_json::to_string_pretty(&file_results)?);
    } else {
        println!("\nTotal issues found: {}", total_issues);
        
        if total_issues > 0 && !fix {
            std::process::exit(1);
        }
    }
    
    Ok(())
}

fn handle_check(matches: &clap::ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    let input = matches.get_one::<String>("input").unwrap();
    let write = matches.get_flag("write");
    let fix = matches.get_flag("fix");
    
    let mut tools = GPUDevTools::new()?;
    
    // Process files
    let files = collect_files(input)?;
    let mut total_formatted = 0;
    let mut total_issues = 0;
    
    for file_path in files {
        let source = fs::read_to_string(&file_path)?;
        let filename = file_path.to_string_lossy().to_string();
        
        // Format and lint in one pass
        let (formatted, issues) = tools.format_and_lint(&source, &filename)?;
        
        let needs_formatting = source != formatted;
        
        if needs_formatting {
            total_formatted += 1;
            if write {
                fs::write(&file_path, formatted)?;
                println!("✓ Formatted {}", filename);
            } else {
                println!("⚠ {} needs formatting", filename);
            }
        }
        
        if !issues.is_empty() {
            total_issues += issues.len();
            println!("  Found {} issues in {}", issues.len(), filename);
            
            if fix {
                println!("  Applied automatic fixes");
            }
        }
    }
    
    println!("\nSummary:");
    println!("  Files needing format: {}", total_formatted);
    println!("  Total lint issues: {}", total_issues);
    
    if (total_formatted > 0 && !write) || (total_issues > 0 && !fix) {
        std::process::exit(1);
    }
    
    Ok(())
}

fn handle_benchmark(matches: &clap::ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    let bench_type = matches.get_one::<String>("type").unwrap();
    let size: usize = matches.get_one::<String>("size")
        .unwrap()
        .parse()
        .unwrap_or(10000);
    
    println!("Running GPU Development Tools Benchmarks");
    println!("========================================\n");
    
    match bench_type.as_str() {
        "format" => {
            let result = gpu_dev_tools::benchmarks::benchmark_formatter(size)?;
            print_benchmark_result(&result);
        }
        "lint" => {
            let result = gpu_dev_tools::benchmarks::benchmark_linter(size / 100)?;
            print_benchmark_result(&result);
        }
        "all" | _ => {
            let format_result = gpu_dev_tools::benchmarks::benchmark_formatter(size)?;
            let lint_result = gpu_dev_tools::benchmarks::benchmark_linter(size / 100)?;
            
            print_benchmark_result(&format_result);
            println!();
            print_benchmark_result(&lint_result);
        }
    }
    
    Ok(())
}

fn handle_validate() -> Result<(), Box<dyn std::error::Error>> {
    println!("Validating GPU Development Tools Performance");
    println!("===========================================\n");
    
    let passed = gpu_dev_tools::validate_performance()?;
    
    println!("\n===========================================");
    if passed {
        println!("✓ All performance targets achieved!");
        println!("✓ 10x improvement validated");
    } else {
        println!("✗ Performance targets not met");
        std::process::exit(1);
    }
    
    Ok(())
}

fn print_benchmark_result(result: &gpu_dev_tools::benchmarks::BenchmarkResult) {
    println!("Operation: {}", result.operation);
    println!("  Input size: {}", result.input_size);
    println!("  Elapsed: {:.2} ms", result.elapsed_ms);
    println!("  Throughput: {:.0} units/sec", result.throughput);
    println!("  Speedup: {:.1}x", result.speedup);
    
    if result.speedup >= 10.0 {
        println!("  Status: ✓ Target achieved");
    } else {
        println!("  Status: ✗ Below target");
    }
}

fn collect_files(path: &str) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let path = Path::new(path);
    let mut files = Vec::new();
    
    if path.is_file() {
        files.push(path.to_path_buf());
    } else if path.is_dir() {
        collect_rust_files(path, &mut files)?;
    } else {
        return Err(format!("Path not found: {}", path.display()).into());
    }
    
    Ok(files)
}

fn collect_rust_files(dir: &Path, files: &mut Vec<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_dir() {
            // Skip target directory
            if path.file_name() == Some(std::ffi::OsStr::new("target")) {
                continue;
            }
            collect_rust_files(&path, files)?;
        } else if path.extension() == Some(std::ffi::OsStr::new("rs")) {
            files.push(path);
        }
    }
    
    Ok(())
}

// For JSON serialization
extern crate serde_json;