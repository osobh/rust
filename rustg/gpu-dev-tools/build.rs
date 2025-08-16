use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/");
    
    // Get the output directory
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    
    // Check for CUDA
    let cuda_available = check_cuda_available();
    if !cuda_available {
        panic!("CUDA 13.0+ required for gpu-dev-tools. Please install CUDA toolkit.");
    }
    
    // Create a simple CUDA source file with the needed functions
    let cuda_source = r#"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

extern "C" {

// Device properties function
int cuda_get_device_properties(
    int device_id,
    char* name,
    int* major,
    int* minor,
    size_t* total_mem,
    int* mp_count,
    int* max_threads,
    int* max_blocks,
    int* warp_size) {
  
  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  
  // Copy device name
  if (name != nullptr) {
    strncpy(name, prop.name, 255);
    name[255] = '\0';
  }
  
  // Set compute capability
  if (major != nullptr) *major = prop.major;
  if (minor != nullptr) *minor = prop.minor;
  
  // Set memory info
  if (total_mem != nullptr) *total_mem = prop.totalGlobalMem;
  
  // Set multiprocessor info
  if (mp_count != nullptr) *mp_count = prop.multiProcessorCount;
  if (max_threads != nullptr) *max_threads = prop.maxThreadsPerBlock;
  if (max_blocks != nullptr) *max_blocks = prop.maxBlocksPerMultiProcessor;
  if (warp_size != nullptr) *warp_size = prop.warpSize;
  
  return 0;
}

// Simple memory allocation functions
void* cuda_malloc(size_t size) {
  void* ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, size);
  if (err != cudaSuccess) {
    return nullptr;
  }
  return ptr;
}

int cuda_free(void* ptr) {
  if (ptr == nullptr) {
    return 0;
  }
  cudaError_t err = cudaFree(ptr);
  return static_cast<int>(err);
}

// Simple format functions (minimal implementation)
int cuda_format_lines(
    const char* lines,
    int line_count,
    const int* changed_lines,
    int change_count,
    const void* options,
    char* output) {
    
    if (!lines || !changed_lines || !output || change_count <= 0) {
        return -1;
    }
    
    // Simple implementation: just copy input to output
    size_t lines_len = strlen(lines);
    if (lines_len > 0) {
        strncpy(output, lines, lines_len);
        output[lines_len] = '\0';
    }
    
    return 0;
}

int cuda_format_ast(
    const char* nodes,
    int node_count,
    const void* options,
    char* output,
    int output_size) {
    
    if (!nodes || !output || node_count <= 0 || output_size <= 0) {
        return -1;
    }
    
    // Simple implementation: write formatted output
    snprintf(output, output_size, "formatted_%d_nodes", node_count);
    
    return 0;
}

} // extern "C"
"#;

    // Write the CUDA source file
    let cuda_file = out_dir.join("gpu_dev_tools_cuda.cu");
    std::fs::write(&cuda_file, cuda_source).expect("Failed to write CUDA source");
    
    // Compile the CUDA source
    let nvcc_paths = vec![
        "/usr/local/cuda-13.0/bin/nvcc",
        "/usr/local/cuda/bin/nvcc",
        "nvcc",
    ];
    
    let mut nvcc_found = false;
    for nvcc_path in nvcc_paths {
        let output = Command::new(&nvcc_path)
            .arg("-c")
            .arg(&cuda_file)
            .arg("-o")
            .arg(out_dir.join("gpu_dev_tools_cuda.o"))
            .arg("--compiler-options")
            .arg("-fPIC")
            .arg("-gencode")
            .arg("arch=compute_75,code=sm_75")
            .arg("-gencode")
            .arg("arch=compute_86,code=sm_86")
            .arg("-gencode")
            .arg("arch=compute_89,code=sm_89")
            .arg("-gencode")
            .arg("arch=compute_110,code=sm_110")
            .output();
            
        if let Ok(output) = output {
            if output.status.success() {
                nvcc_found = true;
                break;
            }
        }
    }
    
    if !nvcc_found {
        panic!("Failed to compile CUDA source with nvcc");
    }
    
    // Create static library
    let ar_output = Command::new("ar")
        .arg("rcs")
        .arg(out_dir.join("libgpu_dev_tools_cuda.a"))
        .arg(out_dir.join("gpu_dev_tools_cuda.o"))
        .output()
        .expect("Failed to create static library");
        
    if !ar_output.status.success() {
        panic!("Failed to create static library");
    }
    
    // Link the library
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=gpu_dev_tools_cuda");
    
    // Link CUDA libraries
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let cuda_lib_path = PathBuf::from(cuda_path).join("lib64");
        println!("cargo:rustc-link-search=native={}", cuda_lib_path.display());
    } else {
        // Try common CUDA installation paths
        for cuda_lib in &["/usr/local/cuda/lib64", "/opt/cuda/lib64"] {
            if std::path::Path::new(cuda_lib).exists() {
                println!("cargo:rustc-link-search=native={}", cuda_lib);
                break;
            }
        }
    }
    
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=stdc++");
}

fn check_cuda_available() -> bool {
    let nvcc_paths = vec![
        "/usr/local/cuda-13.0/bin/nvcc",
        "/usr/local/cuda/bin/nvcc",
        "nvcc",
    ];
    
    for nvcc_path in nvcc_paths {
        if let Ok(output) = Command::new(&nvcc_path).arg("--version").output() {
            let version_str = String::from_utf8_lossy(&output.stdout);
            if version_str.contains("release 13.") || version_str.contains("release 14.") {
                return true;
            }
        }
    }
    false
}