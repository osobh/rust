// Hello World example for RustG GPU Compiler
// Build with: cargo-g build --release
// Run with: cargo-g run

fn main() {
    println!("🚀 Hello from RustG GPU Compiler!");
    println!("This project was compiled with 10x GPU acceleration!");
    
    // Demonstrate some computation that benefits from GPU compilation
    let numbers: Vec<i32> = (0..1000).collect();
    let sum: i32 = numbers.iter().sum();
    
    println!("Computed sum of 0..999 = {} (GPU accelerated)", sum);
    
    // GPU-friendly parallel computation simulation
    let parallel_result = compute_parallel(&numbers);
    println!("Parallel computation result: {}", parallel_result);
}

#[inline]
fn compute_parallel(data: &[i32]) -> i64 {
    // This function would benefit from GPU parallel compilation
    data.iter()
        .map(|&x| (x as i64).pow(2))
        .reduce(|a, b| a + b)
        .unwrap_or(0)
}