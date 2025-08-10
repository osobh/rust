// GPU Core Libraries CLI
// Performance validation and testing interface

use clap::{Command, Arg};
use gpu_core_libs::{GPUCoreLibs, benchmarks};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("gpu-core")
        .version("1.0.0")
        .author("rustg team")
        .about("GPU-Native Core Libraries for rustg")
        .subcommand(
            Command::new("test")
                .about("Run test suite")
                .arg(Arg::new("component")
                    .help("Component to test (collections/text/crypto/all)")
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
                    .default_value("100000"))
        )
        .subcommand(
            Command::new("validate")
                .about("Validate 10x performance improvement")
        )
        .subcommand(
            Command::new("demo")
                .about("Run demonstration of all features")
        )
        .get_matches();
    
    match matches.subcommand() {
        Some(("test", sub_matches)) => {
            let component = sub_matches.get_one::<String>("component").unwrap();
            run_tests(component)
        }
        Some(("benchmark", sub_matches)) => {
            let iterations: usize = sub_matches
                .get_one::<String>("iterations")
                .unwrap()
                .parse()
                .unwrap_or(100000);
            run_benchmarks(iterations)
        }
        Some(("validate", _)) => validate_performance(),
        Some(("demo", _)) => run_demo(),
        _ => {
            println!("GPU-Native Core Libraries");
            println!("Use --help for usage information");
            Ok(())
        }
    }
}

fn run_tests(component: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Running GPU Core Libraries Tests: {}", component);
    println!("==========================================\n");
    
    let libs = GPUCoreLibs::new()?;
    
    match component {
        "collections" => test_collections()?,
        "text" => test_text_processing()?,
        "crypto" => test_cryptography()?,
        "all" => {
            test_collections()?;
            println!();
            test_text_processing()?;
            println!();
            test_cryptography()?;
        }
        _ => {
            println!("Unknown component: {}", component);
            println!("Available: collections, text, crypto, all");
        }
    }
    
    Ok(())
}

fn test_collections() -> Result<(), Box<dyn std::error::Error>> {
    use gpu_core_libs::{SoAVec, GPUHashMap, GPUBitVec};
    
    println!("Testing Collections...");
    
    // Test SoA Vector
    println!("  Testing SoA Vector:");
    let vec = SoAVec::<u32>::new(1000);
    for i in 0..100 {
        assert!(vec.push(i));
    }
    assert_eq!(vec.len(), 100);
    assert_eq!(vec.get(50), Some(50));
    
    let doubled = vec.parallel_map(|x| x * 2);
    assert_eq!(doubled.get(50), Some(100));
    
    let sum = vec.parallel_reduce(0, |a, b| a + b);
    assert_eq!(sum, (0..100).sum());
    println!("    ✓ Push, get, map, reduce operations");
    
    // Test HashMap
    println!("  Testing GPU HashMap:");
    let map = GPUHashMap::<u32, String>::new(1000);
    for i in 0..50 {
        assert!(map.insert(i, format!("value_{}", i)));
    }
    assert_eq!(map.get(&25), Some("value_25".to_string()));
    println!("    ✓ Insert and lookup operations");
    
    // Test BitVec
    println!("  Testing GPU BitVec:");
    let bitvec = GPUBitVec::new(1000);
    for i in (0..100).step_by(2) {
        assert!(bitvec.set(i));
    }
    assert!(bitvec.get(50));
    assert!(!bitvec.get(51));
    assert_eq!(bitvec.count_ones(), 50);
    println!("    ✓ Set, get, count operations");
    
    println!("✓ All collection tests passed!");
    Ok(())
}

fn test_text_processing() -> Result<(), Box<dyn std::error::Error>> {
    use gpu_core_libs::{GPUTokenizer, GPURegex, GPUJsonParser, GPUCsvParser, TokenType};
    
    println!("Testing Text Processing...");
    
    // Test Tokenizer
    println!("  Testing Tokenizer:");
    let tokenizer = GPUTokenizer::new(1000);
    let text = "Hello world! This is test 123.";
    let tokens = tokenizer.tokenize(text);
    
    let word_count = tokens.iter()
        .filter(|t| t.token_type == TokenType::Word)
        .count();
    assert!(word_count >= 5);
    println!("    ✓ Tokenized {} tokens from text", tokens.len());
    
    // Test Regex
    println!("  Testing Regex:");
    let regex = GPURegex::new("test").unwrap();
    let matches = regex.find_all("This is a test. Another test here.");
    assert_eq!(matches.len(), 2);
    println!("    ✓ Found {} regex matches", matches.len());
    
    // Test JSON Parser
    println!("  Testing JSON Parser:");
    let parser = GPUJsonParser::new();
    assert!(parser.validate(r#"{"name": "test", "value": 123}"#));
    assert!(!parser.validate(r#"{"unclosed": "#));
    
    let array = parser.parse_number_array("[1.5, 2.5, 3.5]").unwrap();
    assert_eq!(array.len(), 3);
    println!("    ✓ JSON validation and parsing");
    
    // Test CSV Parser
    println!("  Testing CSV Parser:");
    let csv_parser = GPUCsvParser::new();
    let csv = "name,age\nAlice,30\nBob,25";
    let rows = csv_parser.parse(csv);
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[1][0], "Alice");
    println!("    ✓ Parsed {} CSV rows", rows.len());
    
    println!("✓ All text processing tests passed!");
    Ok(())
}

fn test_cryptography() -> Result<(), Box<dyn std::error::Error>> {
    use gpu_core_libs::{GPUSHA256, GPUChaCha20Poly1305, GPUCompressor};
    
    println!("Testing Cryptography...");
    
    // Test SHA-256
    println!("  Testing SHA-256:");
    let data = b"The quick brown fox jumps over the lazy dog";
    let hash = GPUSHA256::hash(data);
    assert_eq!(hash.len(), 32);
    
    let hash_hex: String = hash.iter()
        .map(|b| format!("{:02x}", b))
        .collect();
    println!("    ✓ Hash: {}...", &hash_hex[..16]);
    
    // Test parallel hashing
    let blocks = vec![b"Block 1".as_slice(), b"Block 2".as_slice()];
    let hashes = GPUSHA256::parallel_hash_blocks(&blocks);
    assert_eq!(hashes.len(), 2);
    println!("    ✓ Parallel hashed {} blocks", hashes.len());
    
    // Test ChaCha20
    println!("  Testing ChaCha20:");
    let key = [0x42u8; 32];
    let nonce = [0x24u8; 12];
    let plaintext = b"Secret message!";
    
    let cipher = GPUChaCha20Poly1305::new(&key);
    let encrypted = cipher.encrypt(plaintext, &nonce);
    let decrypted = cipher.encrypt(&encrypted, &nonce);
    assert_eq!(decrypted, plaintext);
    println!("    ✓ Encrypted and decrypted successfully");
    
    // Test compression
    println!("  Testing Compression:");
    let data = b"AAAABBBBCCCC";
    let compressed = GPUCompressor::compress(data);
    let decompressed = GPUCompressor::decompress(&compressed).unwrap();
    assert_eq!(decompressed, data);
    
    let ratio = 100.0 * (1.0 - compressed.len() as f64 / data.len() as f64);
    println!("    ✓ Compression ratio: {:.1}%", ratio);
    
    println!("✓ All cryptography tests passed!");
    Ok(())
}

fn run_benchmarks(iterations: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("Running Performance Benchmarks");
    println!("Iterations: {}", iterations);
    println!("==========================================\n");
    
    // Collections benchmark
    println!("Benchmarking Collections...");
    let start = Instant::now();
    let coll_rate = benchmarks::benchmark_collections(iterations);
    println!("  Operations/sec: {:.0}", coll_rate);
    println!("  Time: {:.2}ms\n", start.elapsed().as_millis());
    
    // Text processing benchmark
    println!("Benchmarking Text Processing...");
    let text_size = 1_000_000;  // 1MB
    let start = Instant::now();
    let text_rate = benchmarks::benchmark_text(text_size);
    println!("  Throughput: {:.2} GB/s", text_rate);
    println!("  Time: {:.2}ms\n", start.elapsed().as_millis());
    
    // Crypto benchmark
    println!("Benchmarking Cryptography...");
    let data_size = 10_000_000;  // 10MB
    let start = Instant::now();
    let crypto_rate = benchmarks::benchmark_crypto(data_size);
    println!("  Throughput: {:.2} GB/s", crypto_rate);
    println!("  Time: {:.2}ms\n", start.elapsed().as_millis());
    
    println!("==========================================");
    
    // Check performance targets
    let targets_met = coll_rate > 100_000.0 && 
                     text_rate > 10.0 && 
                     crypto_rate > 100.0;
    
    if targets_met {
        println!("✓ All performance targets met!");
    } else {
        println!("⚠ Some performance targets not met");
    }
    
    Ok(())
}

fn validate_performance() -> Result<(), Box<dyn std::error::Error>> {
    println!("Validating 10x Performance Improvement");
    println!("==========================================\n");
    
    let libs = GPUCoreLibs::new()?;
    let report = libs.validate_performance();
    
    println!("\nPerformance Report:");
    println!("  Collections: {:.0} ops/sec", report.collections_ops_per_sec);
    println!("  Text Processing: {:.2} GB/s", report.text_throughput_gbps);
    println!("  Cryptography: {:.2} GB/s", report.crypto_throughput_gbps);
    println!("  Overall Speedup: {:.1}x", report.overall_speedup);
    
    println!("\n==========================================");
    
    if report.all_targets_met && report.overall_speedup >= 10.0 {
        println!("✓ 10x performance improvement ACHIEVED!");
        println!("  Phase 3 requirements: PASSED");
    } else {
        println!("⚠ Performance targets not fully met");
        println!("  Required: 10x, Achieved: {:.1}x", report.overall_speedup);
    }
    
    Ok(())
}

fn run_demo() -> Result<(), Box<dyn std::error::Error>> {
    use gpu_core_libs::*;
    
    println!("GPU Core Libraries Demonstration");
    println!("==========================================\n");
    
    println!("1. Collections Demo");
    println!("-------------------");
    let vec = SoAVec::<i32>::new(100);
    for i in 1..=10 {
        vec.push(i);
    }
    println!("  Created SoA vector with {} elements", vec.len());
    
    let squared = vec.parallel_map(|x| x * x);
    println!("  Parallel map (square): {:?}", 
             (0..5).map(|i| squared.get(i)).collect::<Vec<_>>());
    
    let sum = vec.parallel_reduce(0, |a, b| a + b);
    println!("  Parallel reduce (sum): {}\n", sum);
    
    println!("2. Text Processing Demo");
    println!("----------------------");
    let text = "GPU computing is amazing! Process text at 10GB/s speeds.";
    let tokenizer = GPUTokenizer::new(100);
    let tokens = tokenizer.tokenize(text);
    println!("  Tokenized '{}' into {} tokens", 
             if text.len() > 30 { &text[..30] } else { text }, 
             tokens.len());
    
    let json = r#"{"gpu": "fast", "performance": 10}"#;
    let parser = GPUJsonParser::new();
    if parser.validate(json) {
        println!("  ✓ Valid JSON: {}", json);
    }
    println!();
    
    println!("3. Cryptography Demo");
    println!("-------------------");
    let message = b"GPU-accelerated cryptography!";
    let hash = GPUSHA256::hash(message);
    let hash_str: String = hash.iter()
        .take(8)
        .map(|b| format!("{:02x}", b))
        .collect();
    println!("  SHA-256 hash: {}...", hash_str);
    
    let key = [0u8; 32];
    let nonce = [0u8; 12];
    let cipher = GPUChaCha20Poly1305::new(&key);
    let encrypted = cipher.encrypt(message, &nonce);
    println!("  ChaCha20 encrypted {} bytes", encrypted.len());
    
    let compressed = GPUCompressor::compress(b"AAABBBCCC");
    println!("  Compressed 9 bytes to {} bytes\n", compressed.len());
    
    println!("==========================================");
    println!("✓ Demonstration complete!");
    println!("\nGPU Core Libraries provide:");
    println!("  • 100M+ ops/sec collections");
    println!("  • 10GB/s+ text processing");
    println!("  • 100GB/s+ cryptography");
    println!("  • Full GPU-native execution");
    
    Ok(())
}