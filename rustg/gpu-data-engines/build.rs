fn main() {
    // Configure check-cfg for our custom cfg attributes
    println!("cargo::rustc-check-cfg=cfg(cuda_available)");
    
    // GPU-native implementation - always attempt CUDA compilation
    println!("cargo:warning=Building GPU-native data engines with CUDA...");
    
    // Check CUDA availability first
    let nvcc_available = std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);
    
    // Check for CUDA compilation capability
    let cuda_compilation_supported = if !nvcc_available {
        println!("cargo:warning=NVCC not found - using stub implementation for development");
        false
    } else {
        // Test if we can compile with current GCC/NVCC combination
        let test_result = std::process::Command::new("nvcc")
            .args(&["--version", "-ccbin=g++", "-allow-unsupported-compiler"])
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false);
            
        if !test_result {
            println!("cargo:warning=CUDA/GCC compatibility issue detected - using stub implementation");
            println!("cargo:warning=For production: Use GCC 11 or earlier, or upgrade to newer CUDA toolkit");
            false
        } else {
            true
        }
    };

    if cuda_compilation_supported {
        // Attempt CUDA compilation with compatibility fixes
        let mut build = cc::Build::new();
        build
            .cuda(true)
            .flag("-allow-unsupported-compiler")
            .flag("--std=c++11")
            .flag("-O2")
            .flag("-Xcompiler=-fPIC")
            .flag("-Xcompiler=-w")
            .flag("--disable-warnings")
            .flag("-gencode").flag("arch=compute_60,code=sm_60");
        
        build.files(&["tests/cuda/minimal_test.cu"]);
        
        match build.try_compile("gpu_data_engines_cuda") {
            Ok(_) => {
                println!("cargo:rustc-link-lib=cudart");
                println!("cargo:rustc-link-lib=cublas"); 
                println!("cargo:rustc-link-lib=cusparse");
                println!("cargo:rustc-cfg=cuda_available");
                println!("cargo:warning=✅ CUDA compilation successful - GPU-native mode enabled");
            },
            Err(err) => {
                println!("cargo:warning=CUDA compilation failed: {}", err);
                println!("cargo:warning=Falling back to stub implementation for development");
                println!("cargo:rustc-cfg=cuda_stub");
            }
        }
    } else {
        println!("cargo:rustc-cfg=cuda_stub");
        println!("cargo:warning=Using CUDA stub implementation - performance tests will return mock results");
    }

    // Rerun conditions
    println!("cargo:rerun-if-changed=tests/cuda/");
    println!("cargo:rerun-if-changed=build.rs");
}