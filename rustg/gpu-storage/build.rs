// Build script for GPU Storage with nvidia-fs integration
// Links against NVIDIA GPUDirect Storage (cuFile) library

fn main() {
    println!("cargo:warning=Building GPU Storage with nvidia-fs integration...");
    
    // Check for CUDA installation
    let cuda_path = std::env::var("CUDA_PATH")
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    
    // Check for nvidia-fs/GDS installation
    let gds_path = std::env::var("GDS_PATH")
        .unwrap_or_else(|_| "/usr/local/cuda/gds".to_string());
    
    // Verify nvidia-fs is installed
    let cufile_lib = format!("{}/lib", gds_path);
    let cufile_include = format!("{}/include", gds_path);
    
    if !std::path::Path::new(&cufile_lib).exists() {
        println!("cargo:warning=nvidia-fs library path not found at {}", cufile_lib);
        println!("cargo:warning=Attempting fallback to CUDA lib path");
        
        // Try CUDA lib path as fallback
        let cuda_lib = format!("{}/lib64", cuda_path);
        if std::path::Path::new(&cuda_lib).exists() {
            println!("cargo:rustc-link-search=native={}", cuda_lib);
        }
    } else {
        println!("cargo:rustc-link-search=native={}", cufile_lib);
    }
    
    // Add include paths
    if std::path::Path::new(&cufile_include).exists() {
        println!("cargo:include={}", cufile_include);
    }
    println!("cargo:include={}/include", cuda_path);
    
    // Link CUDA libraries
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");
    
    // Link nvidia-fs/cuFile library
    println!("cargo:rustc-link-lib=cufile");
    
    // Additional CUDA libraries for GPU operations
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cusparse");
    
    // Compile CUDA test files with cuFile support
    let mut build = cc::Build::new();
    build
        .cuda(true)
        .flag("-std=c++14")
        .flag("-O3")
        .flag("-use_fast_math")
        .flag("-Xcompiler=-fPIC")
        .flag("-gencode").flag("arch=compute_70,code=sm_70")
        .flag("-gencode").flag("arch=compute_75,code=sm_75")
        .flag("-gencode").flag("arch=compute_80,code=sm_80")
        .flag("-gencode").flag("arch=compute_86,code=sm_86");
    
    // Add include paths for compilation
    build.include(&cufile_include);
    build.include(format!("{}/include", cuda_path));
    
    // Define macro to enable real nvidia-fs instead of simulation
    build.define("USE_REAL_NVIDIA_FS", "1");
    
    // Add storage tier paths
    build.define("NVME_PATH", "\"/nvme\"");
    build.define("SSD_PATH", "\"/ssd\"");
    build.define("HDD_PATH", "\"/hdd\"");
    
    // Compile CUDA test files
    let cuda_files = vec![
        "tests/cuda/gpudirect_storage_test.cu",
        "tests/cuda/cache_test.cu",
        "tests/cuda/format_handlers_test.cu",
        "tests/cuda/storage_abstraction_test.cu",
    ];
    
    for file in &cuda_files {
        if std::path::Path::new(file).exists() {
            build.file(file);
        } else {
            println!("cargo:warning=CUDA test file not found: {}", file);
        }
    }
    
    // Try to compile
    match build.try_compile("gpu_storage_cuda") {
        Ok(_) => {
            println!("cargo:warning=✅ GPU Storage with nvidia-fs compiled successfully");
            println!("cargo:rustc-cfg=nvidia_fs_available");
        }
        Err(e) => {
            println!("cargo:warning=⚠️ CUDA compilation failed: {}", e);
            println!("cargo:warning=nvidia-fs features will be disabled");
            println!("cargo:rustc-cfg=nvidia_fs_stub");
        }
    }
    
    // Verify storage paths exist
    for (name, path) in &[("NVMe", "/nvme"), ("SSD", "/ssd"), ("HDD", "/hdd")] {
        if std::path::Path::new(path).exists() {
            println!("cargo:warning=✅ {} storage path exists at {}", name, path);
        } else {
            println!("cargo:warning=⚠️ {} storage path not found at {}", name, path);
        }
    }
    
    // Set rerun conditions
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=tests/cuda/");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=GDS_PATH");
}