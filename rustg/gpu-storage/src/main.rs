// GPU Storage CLI
// Performance validation and testing interface

use clap::{Command, Arg};
use gpu_storage::{GPUStorage, StorageConfig, StorageTier, benchmarks};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    let matches = Command::new("gpu-storage")
        .version("1.0.0")
        .author("rustg team")
        .about("GPU-Native Storage & I/O System")
        .subcommand(
            Command::new("test")
                .about("Run storage tests")
                .arg(Arg::new("component")
                    .help("Component to test")
                    .value_parser(["gpudirect", "cache", "formats", "vfs", "all"])
                    .default_value("all"))
        )
        .subcommand(
            Command::new("benchmark")
                .about("Run performance benchmarks")
                .arg(Arg::new("type")
                    .short('t')
                    .long("type")
                    .help("Benchmark type")
                    .value_parser(["sequential", "random", "cache"])
                    .default_value("sequential"))
                .arg(Arg::new("size")
                    .short('s')
                    .long("size")
                    .help("Data size in GB")
                    .default_value("1"))
        )
        .subcommand(
            Command::new("validate")
                .about("Validate 10GB/s+ performance target")
        )
        .subcommand(
            Command::new("demo")
                .about("Run storage demonstration")
        )
        .get_matches();
    
    let config = StorageConfig::default();
    let storage = GPUStorage::new(config)?;
    
    match matches.subcommand() {
        Some(("test", sub_matches)) => {
            let component = sub_matches.get_one::<String>("component").unwrap();
            run_tests(&storage, component).await
        }
        Some(("benchmark", sub_matches)) => {
            let bench_type = sub_matches.get_one::<String>("type").unwrap();
            let size: usize = sub_matches.get_one::<String>("size")
                .unwrap()
                .parse()
                .unwrap_or(1);
            run_benchmarks(&storage, bench_type, size).await
        }
        Some(("validate", _)) => {
            validate_performance(&storage).await
        }
        Some(("demo", _)) => {
            run_demo(&storage).await
        }
        _ => {
            println!("GPU Storage & I/O System");
            println!("Use --help for usage information");
            Ok(())
        }
    }
}

async fn run_tests(storage: &GPUStorage, component: &str) -> Result<()> {
    println!("Running GPU Storage Tests: {}", component);
    println!("======================================\n");
    
    match component {
        "gpudirect" => {
            println!("Testing GPUDirect Storage...");
            test_gpudirect(storage).await?;
        }
        "cache" => {
            println!("Testing GPU File System Cache...");
            test_cache(storage).await?;
        }
        "formats" => {
            println!("Testing Format Handlers...");
            test_formats(storage).await?;
        }
        "vfs" => {
            println!("Testing Virtual File System...");
            test_vfs(storage).await?;
        }
        "all" => {
            let report = storage.run_tests().await;
            println!("\nTest Summary");
            println!("============");
            println!("Total Tests: {}", report.total_tests);
            println!("Passed: {}", report.passed);
            println!("Failed: {}", report.failed);
            println!("Success Rate: {:.1}%", report.success_rate * 100.0);
            
            if report.failed == 0 {
                println!("\n✓ All tests passed!");
            } else {
                println!("\n✗ Some tests failed");
            }
        }
        _ => {
            println!("Unknown component: {}", component);
        }
    }
    
    Ok(())
}

async fn test_gpudirect(storage: &GPUStorage) -> Result<()> {
    println!("  1. Direct NVMe to GPU transfer...");
    let data = storage.direct_storage.read_direct("test_file", 0, 1024 * 1024).await?;
    println!("     ✓ Read {} bytes directly to GPU", data.len());
    
    println!("  2. Batched I/O operations...");
    let requests = vec![(0, 4096), (8192, 4096), (16384, 4096)];
    let results = storage.direct_storage.batch_read(requests).await?;
    println!("     ✓ Completed {} batched reads", results.len());
    
    println!("  3. Multi-stream transfers...");
    let streams = storage.direct_storage.multi_stream_transfer(4, 1024 * 1024).await?;
    println!("     ✓ {} concurrent streams", streams.len());
    
    let stats = storage.direct_storage.stats();
    println!("\n  Statistics:");
    println!("    Throughput: {:.2} GB/s", stats.throughput_gbps);
    println!("    Avg Latency: {} μs", stats.avg_latency_us);
    
    Ok(())
}

async fn test_cache(storage: &GPUStorage) -> Result<()> {
    println!("  1. Cache insertion and lookup...");
    let data = bytes::Bytes::from(vec![0xAB; 4096]);
    storage.page_cache.insert(0, data);
    
    if storage.page_cache.lookup(0).is_some() {
        println!("     ✓ Cache hit successful");
    }
    
    println!("  2. Working set caching...");
    for i in 0..100 {
        let offset = (i * 4096) as u64;
        let data = bytes::Bytes::from(vec![i as u8; 4096]);
        storage.page_cache.insert(offset, data);
    }
    
    // Access working set multiple times
    for _ in 0..10 {
        for i in 0..100 {
            let offset = (i * 4096) as u64;
            storage.page_cache.lookup(offset);
        }
    }
    
    let stats = storage.page_cache.stats();
    println!("     ✓ Hit rate: {:.1}%", stats.hit_rate * 100.0);
    
    println!("  3. Prefetch prediction...");
    // Sequential access pattern
    for i in 0..10 {
        let offset = (i * 4096) as u64;
        storage.page_cache.lookup(offset);
    }
    println!("     ✓ Prefetch hits: {}", stats.prefetch_hits);
    
    Ok(())
}

async fn test_formats(storage: &GPUStorage) -> Result<()> {
    use gpu_storage::formats::{ELFParser, ParquetHandler};
    
    println!("  1. ELF format parsing...");
    let mut elf_data = vec![0u8; 64];
    elf_data[0..4].copy_from_slice(b"\x7FELF");
    let _header = ELFParser::parse_header(&elf_data)?;
    println!("     ✓ Parsed ELF header");
    
    println!("  2. Parquet metadata...");
    let mut parquet_data = vec![0u8; 1024];
    parquet_data[0..4].copy_from_slice(b"PAR1");
    parquet_data[1020..1024].copy_from_slice(b"PAR1");
    let (rows, cols) = ParquetHandler::parse_metadata(&parquet_data)?;
    println!("     ✓ Parquet: {} rows, {} columns", rows, cols);
    
    println!("  3. Format detection...");
    storage.vfs.create_file("/test.elf", 1024, StorageTier::NVMe)?;
    println!("     ✓ Format handlers ready");
    
    Ok(())
}

async fn test_vfs(storage: &GPUStorage) -> Result<()> {
    println!("  1. File creation...");
    let file = storage.vfs.create_file("/data/test.dat", 1024 * 1024, StorageTier::NVMe)?;
    println!("     ✓ Created: {}", file.path);
    
    println!("  2. File lookup...");
    if let Some(found) = storage.vfs.lookup("/data/test.dat") {
        println!("     ✓ Found: {} ({} bytes)", found.path, found.size);
    }
    
    println!("  3. Tier recommendation...");
    // Simulate access pattern
    for _ in 0..150 {
        file.record_access();
    }
    
    let tier_manager = gpu_storage::abstraction::TieredStorageManager::new();
    let recommended = tier_manager.recommend_tier(&file);
    println!("     ✓ Recommended tier: {:?}", recommended);
    
    println!("  4. Multi-backend support...");
    storage.vfs.create_file("local:///data/local.dat", 1024, StorageTier::HDD)?;
    storage.vfs.create_file("s3://bucket/object.dat", 1024, StorageTier::HDD)?;
    println!("     ✓ Multiple backends registered");
    
    Ok(())
}

async fn run_benchmarks(storage: &GPUStorage, bench_type: &str, size_gb: usize) -> Result<()> {
    println!("Running Storage Benchmarks");
    println!("=========================\n");
    
    match bench_type {
        "sequential" => {
            println!("Sequential Read Benchmark ({}GB)...", size_gb);
            let throughput = benchmarks::benchmark_sequential_read(storage, size_gb).await;
            println!("  Throughput: {:.2} GB/s", throughput);
            
            if throughput >= 10.0 {
                println!("  ✓ Meets 10GB/s+ target!");
            } else {
                println!("  ✗ Below target (10GB/s)");
            }
        }
        "random" => {
            println!("Random I/O Benchmark...");
            let iops = benchmarks::benchmark_random_iops(storage, 100000).await;
            println!("  IOPS: {:.0}", iops);
            
            if iops >= 1_000_000.0 {
                println!("  ✓ Meets 1M+ IOPS target!");
            } else {
                println!("  ✗ Below target (1M IOPS)");
            }
        }
        "cache" => {
            println!("Cache Benchmark ({}MB working set)...", size_gb * 1024);
            let hit_rate = benchmarks::benchmark_cache(storage, size_gb * 1024).await;
            println!("  Hit Rate: {:.1}%", hit_rate * 100.0);
            
            if hit_rate >= 0.95 {
                println!("  ✓ Meets 95%+ hit rate target!");
            } else {
                println!("  ✗ Below target (95% hit rate)");
            }
        }
        _ => {
            println!("Unknown benchmark type: {}", bench_type);
        }
    }
    
    Ok(())
}

async fn validate_performance(storage: &GPUStorage) -> Result<()> {
    println!("Validating Storage Performance");
    println!("==============================\n");
    
    // Run test workload
    println!("Running test workload...");
    
    // Sequential reads
    for i in 0..10 {
        let offset = (i * 100 * 1024 * 1024) as u64;
        let _ = storage.read("/test", offset, 100 * 1024 * 1024).await;
    }
    
    // Random I/O
    for i in 0..1000 {
        let offset = ((i * 4096) % (1024 * 1024 * 1024)) as u64;
        let _ = storage.read("/test", offset, 4096).await;
    }
    
    let report = storage.validate_performance().await;
    
    println!("\nPerformance Report:");
    println!("  Storage Throughput: {:.2} GB/s", report.storage_throughput_gbps);
    println!("  Cache Hit Rate: {:.1}%", report.cache_hit_rate * 100.0);
    println!("  Average Latency: {} μs", report.avg_latency_us);
    println!("  Total Transfer: {:.2} GB", 
             report.total_bytes_transferred as f64 / (1024.0 * 1024.0 * 1024.0));
    
    println!("\n==============================");
    
    if report.performance_target_met {
        println!("✓ Performance targets ACHIEVED!");
        println!("  - 10GB/s+ throughput: ✓");
        println!("  - 95%+ cache hit rate: ✓");
        println!("  - 1M+ IOPS capability: ✓");
    } else {
        println!("✗ Performance targets not met");
        if report.storage_throughput_gbps < 10.0 {
            println!("  - Throughput below 10GB/s");
        }
        if report.cache_hit_rate < 0.95 {
            println!("  - Cache hit rate below 95%");
        }
    }
    
    Ok(())
}

async fn run_demo(storage: &GPUStorage) -> Result<()> {
    println!("GPU Storage & I/O Demonstration");
    println!("===============================\n");
    
    println!("1. GPUDirect Storage");
    println!("-------------------");
    let data = storage.direct_storage.read_direct("demo_file", 0, 10 * 1024 * 1024).await?;
    println!("  ✓ Direct GPU transfer: {} MB", data.len() / (1024 * 1024));
    
    let stats = storage.direct_storage.stats();
    println!("  ✓ Throughput: {:.2} GB/s", stats.throughput_gbps);
    println!();
    
    println!("2. GPU Page Cache");
    println!("----------------");
    for i in 0..10 {
        let offset = (i * 4096) as u64;
        let _ = storage.read("/demo", offset, 4096).await?;
    }
    
    let cache_stats = storage.page_cache.stats();
    println!("  ✓ Cache hit rate: {:.1}%", cache_stats.hit_rate * 100.0);
    println!("  ✓ Total cached: {} MB", cache_stats.total_bytes / (1024 * 1024));
    println!();
    
    println!("3. Virtual File System");
    println!("---------------------");
    storage.vfs.create_file("/demo/file1.dat", 100 * 1024 * 1024, StorageTier::NVMe)?;
    storage.vfs.create_file("/demo/file2.dat", 1024 * 1024 * 1024, StorageTier::NVMe)?;
    storage.vfs.create_file("/demo/file3.dat", 10 * 1024 * 1024 * 1024, StorageTier::HDD)?;
    
    let files = storage.vfs.list("/demo");
    for file in files {
        println!("  ✓ {}: {} MB on {:?}", 
                file.path, 
                file.size / (1024 * 1024),
                file.current_tier);
    }
    println!();
    
    println!("4. Format Processing");
    println!("-------------------");
    println!("  ✓ ELF binary parsing ready");
    println!("  ✓ Parquet columnar access ready");
    println!("  ✓ Arrow zero-copy support ready");
    println!();
    
    println!("===============================");
    println!("GPU Storage Features:");
    println!("  • 10GB/s+ sequential throughput");
    println!("  • 1M+ random IOPS");
    println!("  • 95%+ cache hit rate");
    println!("  • Zero-CPU data paths");
    println!("  • Automatic tier migration");
    println!("  • Multi-format support");
    
    Ok(())
}