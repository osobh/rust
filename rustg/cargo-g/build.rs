use std::env;
use std::path::PathBuf;

fn main() {
    // Set up CUDA compilation for test kernels
    let cuda_path = env::var("CUDA_PATH")
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");
    
    // Compile CUDA test kernels
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    cc::Build::new()
        .cuda(true)
        .cudart("static")
        .flag("-arch=sm_70") // Minimum for modern GPUs
        .flag("-O3")
        .flag("--use_fast_math")
        .file("tests/cuda/gpu_detection_test.cu")
        .file("tests/cuda/build_config_test.cu")
        .file("tests/cuda/cache_test.cu")
        .file("tests/cuda/performance_test.cu")
        .compile("libcargo_g_tests.a");
    
    println!("cargo:rerun-if-changed=tests/cuda/");
}