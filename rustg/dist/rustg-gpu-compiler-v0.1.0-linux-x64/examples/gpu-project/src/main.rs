// Advanced GPU Project Example
// Demonstrates GPU-optimized patterns that clippy-f can analyze

use rayon::prelude::*;
use std::time::Instant;

fn main() {
    println!("🎯 GPU-Optimized Project Example");
    println!("Demonstrating patterns optimized for GPU compilation");
    
    // Large dataset for GPU processing
    let data: Vec<f64> = (0..1_000_000).map(|x| x as f64).collect();
    
    println!("Processing {} elements...", data.len());
    
    // GPU-friendly parallel operations
    let start = Instant::now();
    let results = gpu_optimized_computation(&data);
    let gpu_time = start.elapsed();
    
    println!("GPU-optimized computation completed in {:?}", gpu_time);
    println!("Result: sum={:.2}, avg={:.2}", results.sum, results.average);
    
    // Memory access patterns that benefit from GPU compilation
    demonstrate_memory_patterns();
    
    // Parallel algorithms optimized for CUDA cores
    demonstrate_parallel_algorithms();
}

struct ComputationResult {
    sum: f64,
    average: f64,
    max: f64,
    min: f64,
}

#[inline]
fn gpu_optimized_computation(data: &[f64]) -> ComputationResult {
    // This computation pattern is optimized for GPU parallel execution
    let sum = data.par_iter().sum::<f64>();
    let max = data.par_iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = data.par_iter().copied().fold(f64::INFINITY, f64::min);
    let average = sum / data.len() as f64;
    
    ComputationResult { sum, average, max, min }
}

fn demonstrate_memory_patterns() {
    println!("\n🧠 Memory Access Pattern Analysis:");
    
    // Coalesced memory access (GPU-friendly)
    let data: Vec<[f32; 4]> = vec![[1.0, 2.0, 3.0, 4.0]; 100_000];
    
    // Sequential access pattern (optimal for GPU)
    let _result: Vec<f32> = data
        .par_iter()
        .map(|chunk| chunk.iter().sum())
        .collect();
    
    println!("✅ Coalesced memory access pattern processed");
    
    // Strided access pattern (clippy-f would flag this)
    let large_array = vec![1.0f32; 1_000_000];
    let _strided_sum: f32 = (0..large_array.len())
        .step_by(16) // Strided access - not optimal for GPU
        .map(|i| large_array[i])
        .sum();
    
    println!("⚠️  Strided access pattern (clippy-f would suggest optimization)");
}

fn demonstrate_parallel_algorithms() {
    println!("\n⚡ Parallel Algorithm Patterns:");
    
    // Reduction pattern (excellent for GPU)
    let numbers: Vec<i64> = (0..100_000).collect();
    let sum = numbers.par_iter().sum::<i64>();
    println!("✅ Parallel reduction: {}", sum);
    
    // Map-reduce pattern (GPU-optimized)
    let squares: Vec<i64> = numbers
        .par_iter()
        .map(|&x| x * x)
        .collect();
    
    let squares_sum = squares.par_iter().sum::<i64>();
    println!("✅ Map-reduce pattern: {}", squares_sum);
    
    // Divergent branching (clippy-f would flag for GPU optimization)
    let _divergent_result: Vec<i64> = numbers
        .par_iter()
        .map(|&x| {
            if x % 2 == 0 {
                expensive_computation(x)
            } else {
                simple_computation(x)  
            }
        })
        .collect();
    
    println!("⚠️  Divergent branching pattern (clippy-f would suggest optimization)");
}

#[inline]
fn expensive_computation(x: i64) -> i64 {
    // Simulate expensive computation
    (0..10).map(|i| x + i).sum()
}

#[inline] 
fn simple_computation(x: i64) -> i64 {
    x * 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_computation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = gpu_optimized_computation(&data);
        
        assert_eq!(result.sum, 15.0);
        assert_eq!(result.average, 3.0);
        assert_eq!(result.max, 5.0);
        assert_eq!(result.min, 1.0);
    }

    #[test] 
    fn test_parallel_algorithms() {
        // Test would be accelerated by cargo-g
        let numbers: Vec<i64> = (0..1000).collect();
        let sum = numbers.iter().sum::<i64>();
        assert_eq!(sum, 499500);
    }
}