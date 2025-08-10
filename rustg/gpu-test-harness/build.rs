use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use walkdir::WalkDir;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=tests/cuda/");
    
    // Get CUDA path
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    
    let cuda_include = format!("{}/include", cuda_path);
    let cuda_lib = format!("{}/lib64", cuda_path);
    
    // Set up CUDA compilation
    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir);
    
    // Find all CUDA test files
    let cuda_tests_dir = "tests/cuda";
    if Path::new(cuda_tests_dir).exists() {
        for entry in WalkDir::new(cuda_tests_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "cu"))
        {
            let cuda_file = entry.path();
            let file_stem = cuda_file.file_stem().unwrap().to_str().unwrap();
            let output_file = out_path.join(format!("{}", file_stem));
            
            println!("cargo:warning=Compiling CUDA test: {:?}", cuda_file);
            
            // Compile CUDA file to executable
            let output = Command::new("nvcc")
                .args(&[
                    "-O3",
                    "-arch=sm_70",  // Volta and newer
                    "-std=c++14",
                    "--expt-relaxed-constexpr",
                    "-I", &cuda_include,
                    cuda_file.to_str().unwrap(),
                    "-o", output_file.to_str().unwrap(),
                    "-lcudart",
                ])
                .output();
            
            match output {
                Ok(result) => {
                    if !result.status.success() {
                        println!("cargo:warning=CUDA compilation failed for {:?}", cuda_file);
                        println!("cargo:warning=stderr: {}", String::from_utf8_lossy(&result.stderr));
                    } else {
                        println!("cargo:warning=Successfully compiled {:?}", cuda_file);
                    }
                }
                Err(e) => {
                    println!("cargo:warning=Failed to run nvcc: {}", e);
                    println!("cargo:warning=Make sure CUDA toolkit is installed");
                }
            }
        }
    }
    
    // Link CUDA libraries
    println!("cargo:rustc-link-search=native={}", cuda_lib);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");
    
    // Generate bindings for CUDA runtime API
    let bindings = bindgen::Builder::default()
        .header_contents(
            "cuda_wrapper.h",
            "#include <cuda_runtime.h>\n#include <device_launch_parameters.h>",
        )
        .clang_arg(format!("-I{}", cuda_include))
        .clang_arg("-x").clang_arg("c++")
        .clang_arg("-std=c++14")
        .allowlist_function("cuda.*")
        .allowlist_type("cuda.*")
        .allowlist_var("cuda.*")
        .derive_default(true)
        .generate();
    
    match bindings {
        Ok(b) => {
            let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
            b.write_to_file(out_path.join("cuda_bindings.rs"))
                .expect("Couldn't write bindings!");
        }
        Err(e) => {
            println!("cargo:warning=Failed to generate CUDA bindings: {:?}", e);
            println!("cargo:warning=Tests will compile but may have limited CUDA support");
        }
    }
}