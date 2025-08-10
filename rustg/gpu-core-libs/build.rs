use std::env;
use std::path::PathBuf;

fn main() {
    // Compile CUDA tests
    let cuda_files = vec![
        "tests/cuda/collections_test.cu",
        "tests/cuda/text_processing_test.cu",
        "tests/cuda/crypto_test.cu",
    ];
    
    for cuda_file in &cuda_files {
        println!("cargo:rerun-if-changed={}", cuda_file);
    }
    
    // Use nvcc to compile CUDA files if available
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        
        for cuda_file in cuda_files {
            let file_stem = PathBuf::from(&cuda_file)
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();
            
            let output = out_dir.join(format!("{}", file_stem));
            
            let status = std::process::Command::new("nvcc")
                .args(&[
                    "-O3",
                    "-arch=sm_70",
                    "--std=c++14",
                    "-o",
                    output.to_str().unwrap(),
                    cuda_file,
                ])
                .status();
            
            if let Ok(status) = status {
                if !status.success() {
                    println!("cargo:warning=Failed to compile {}", cuda_file);
                }
            }
        }
    } else {
        println!("cargo:warning=CUDA_PATH not set, skipping CUDA compilation");
    }
    
    // Link CUDA runtime
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
}