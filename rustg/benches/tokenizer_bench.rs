//! Performance benchmarks for the tokenizer

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use rustg::lexer::{tokenize_cpu, tokenize_gpu};
use std::time::Duration;

fn generate_source_code(size: usize) -> String {
    let mut source = String::new();
    
    // Generate realistic Rust code
    source.push_str("use std::collections::HashMap;\n\n");
    
    for i in 0..size {
        source.push_str(&format!(
            r#"
#[derive(Debug, Clone)]
pub struct Struct_{} {{
    field1: i32,
    field2: String,
    field3: Vec<u8>,
}}

impl Struct_{} {{
    pub fn new(value: i32) -> Self {{
        Self {{
            field1: value,
            field2: format!("value_{{}}", value),
            field3: vec![0; value as usize],
        }}
    }}
    
    pub fn process(&mut self) -> Result<(), String> {{
        if self.field1 < 0 {{
            return Err("Invalid value".to_string());
        }}
        self.field1 *= 2;
        Ok(())
    }}
}}

fn function_{}(x: i32, y: f64) -> f64 {{
    let mut sum = 0.0;
    for i in 0..x {{
        sum += y * i as f64;
    }}
    sum / x as f64
}}
"#,
            i, i, i
        ));
    }
    
    source
}

fn bench_cpu_tokenizer(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_tokenizer");
    group.measurement_time(Duration::from_secs(10));
    
    for size in [10, 100, 1000, 10000].iter() {
        let source = generate_source_code(*size);
        let source_len = source.len();
        
        group.throughput(Throughput::Bytes(source_len as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &source,
            |b, source| {
                b.iter(|| {
                    let tokens = tokenize_cpu(black_box(source))
                        .expect("CPU tokenization failed");
                    black_box(tokens);
                });
            },
        );
    }
    group.finish();
}

fn bench_gpu_tokenizer(c: &mut Criterion) {
    // Initialize GPU runtime once
    rustg::initialize().expect("Failed to initialize GPU");
    
    let mut group = c.benchmark_group("gpu_tokenizer");
    group.measurement_time(Duration::from_secs(10));
    
    for size in [10, 100, 1000, 10000].iter() {
        let source = generate_source_code(*size);
        let source_len = source.len();
        
        group.throughput(Throughput::Bytes(source_len as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &source,
            |b, source| {
                b.iter(|| {
                    let tokens = tokenize_gpu(black_box(source))
                        .expect("GPU tokenization failed");
                    black_box(tokens);
                });
            },
        );
    }
    group.finish();
    
    // Cleanup GPU runtime
    rustg::shutdown().expect("Failed to shutdown GPU");
}

fn bench_comparison(c: &mut Criterion) {
    rustg::initialize().expect("Failed to initialize GPU");
    
    let mut group = c.benchmark_group("tokenizer_comparison");
    group.measurement_time(Duration::from_secs(15));
    
    // Large file for meaningful comparison
    let source = generate_source_code(5000);
    let source_len = source.len();
    
    println!("Benchmark source size: {} bytes", source_len);
    
    group.throughput(Throughput::Bytes(source_len as u64));
    
    group.bench_function("cpu", |b| {
        b.iter(|| {
            let tokens = tokenize_cpu(black_box(&source))
                .expect("CPU tokenization failed");
            black_box(tokens);
        });
    });
    
    group.bench_function("gpu", |b| {
        b.iter(|| {
            let tokens = tokenize_gpu(black_box(&source))
                .expect("GPU tokenization failed");
            black_box(tokens);
        });
    });
    
    group.finish();
    
    rustg::shutdown().expect("Failed to shutdown GPU");
}

fn bench_scalability(c: &mut Criterion) {
    rustg::initialize().expect("Failed to initialize GPU");
    
    let mut group = c.benchmark_group("gpu_scalability");
    group.measurement_time(Duration::from_secs(20));
    
    // Test scalability with different file sizes
    let sizes = vec![
        ("1KB", 1024),
        ("10KB", 10 * 1024),
        ("100KB", 100 * 1024),
        ("1MB", 1024 * 1024),
        ("10MB", 10 * 1024 * 1024),
    ];
    
    for (label, target_size) in sizes {
        // Generate source approximately matching target size
        let iterations = target_size / 500; // Each iteration generates ~500 bytes
        let source = generate_source_code(iterations);
        let actual_size = source.len();
        
        group.throughput(Throughput::Bytes(actual_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &source,
            |b, source| {
                b.iter(|| {
                    let tokens = tokenize_gpu(black_box(source))
                        .expect("GPU tokenization failed");
                    black_box(tokens);
                });
            },
        );
    }
    
    group.finish();
    
    rustg::shutdown().expect("Failed to shutdown GPU");
}

criterion_group!(
    benches, 
    bench_cpu_tokenizer,
    bench_gpu_tokenizer,
    bench_comparison,
    bench_scalability
);
criterion_main!(benches);