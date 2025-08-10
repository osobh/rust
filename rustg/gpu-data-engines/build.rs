use std::{env, path::PathBuf, process::Command};

fn nvcc() -> String {
    // Try multiple paths to find NVCC
    let nvcc_paths = vec![
        env::var("CUDA_BIN_PATH")
            .map(|p| format!("{}/nvcc", p))
            .unwrap_or_default(),
        "/usr/local/cuda/bin/nvcc".into(),
        "/usr/local/cuda-13.0/bin/nvcc".into(),
        "nvcc".into(),
    ];
    
    for path in nvcc_paths {
        if PathBuf::from(&path).exists() || 
           Command::new(&path).arg("--version").output().is_ok() {
            return path;
        }
    }
    
    panic!("NVCC not found! Please install CUDA toolkit or set CUDA_BIN_PATH");
}

fn main() {
    println!("cargo:warning=Building GPU-native data engines with CUDA (RDC enabled)...");
    
    let nvcc_path = nvcc();
    
    // Print NVCC version for debugging
    if let Ok(out) = Command::new(&nvcc_path).arg("--version").output() {
        let version = String::from_utf8_lossy(&out.stdout);
        println!("cargo:warning=Using NVCC: {}", version.lines().next().unwrap_or("unknown"));
    }
    
    // Configurable architecture (override with GPU_SM=sm_90 GPU_COMPUTE=compute_90)
    let gpu_sm = env::var("GPU_SM").unwrap_or_else(|_| "sm_90".into());
    let gpu_compute = env::var("GPU_COMPUTE").unwrap_or_else(|_| "compute_90".into());
    
    println!("cargo:warning=Target GPU: {} (PTX: {})", gpu_sm, gpu_compute);
    
    // List of CUDA source files
    let cu_srcs = &[
        "tests/cuda/cuda_safe_api.cu",
        "tests/cuda/cuda_init.cu",
        "tests/cuda/dataframe_test.cu",
        "tests/cuda/graph_test.cu",
        "tests/cuda/search_test.cu",
        "tests/cuda/sql_test.cu",
    ];
    
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    // Step 1: Compile each .cu file to .o with RDC enabled
    let objects: Vec<PathBuf> = cu_srcs.iter().enumerate().map(|(i, src)| {
        let obj = out_dir.join(format!("cuda_{}.o", i));
        
        println!("cargo:warning=Compiling {} -> {}", src, obj.display());
        
        let status = Command::new(&nvcc_path)
            .args([
                "-c", src,
                "--std=c++17",
                "-Xcompiler", "-fPIC",
                "-rdc=true",  // CRITICAL: Enable relocatable device code
                "-gencode", &format!("arch={},code={}", gpu_compute, gpu_sm),
                "-gencode", &format!("arch={},code={}", gpu_compute, gpu_compute), // PTX fallback
                "-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA",  // Define Thrust device system
            ])
            .arg("-lineinfo")  // Add debug info
            .arg("-O3")         // Optimize
            .arg("-use_fast_math")
            .arg("--extended-lambda")  // Support device lambdas
            .arg("-Xcompiler=-Wno-unused-parameter")
            .arg("-Xcompiler=-Wno-unused-variable")
            .arg("-Xcompiler=-w")  // Suppress host compiler warnings
            .arg("--disable-warnings")  // Suppress NVCC warnings
            .arg("-o").arg(&obj)
            .status()
            .expect(&format!("Failed to run nvcc for {}", src));
            
        if !status.success() {
            panic!("NVCC compilation failed for {}", src);
        }
        
        obj
    }).collect();
    
    // Step 2: Device link - create object with all device symbols
    let dlink_obj = out_dir.join("cuda_device_link.o");
    
    println!("cargo:warning=Device linking {} objects -> {}", objects.len(), dlink_obj.display());
    
    let mut dlink_cmd = Command::new(&nvcc_path);
    dlink_cmd.args([
        "-dlink",
        "-rdc=true",
        "-gencode", &format!("arch={},code={}", gpu_compute, gpu_sm),
        "-gencode", &format!("arch={},code={}", gpu_compute, gpu_compute),
        "-o", dlink_obj.to_str().unwrap(),
    ]);
    
    // Add all object files to device link
    for obj in &objects {
        dlink_cmd.arg(obj);
    }
    
    let status = dlink_cmd.status().expect("Failed to run nvcc -dlink");
    if !status.success() {
        panic!("NVCC device linking failed");
    }
    
    // Step 3: Create static library from all objects
    let lib_name = "gpu_data_engines_cuda";
    let lib_path = out_dir.join(format!("lib{}.a", lib_name));
    
    println!("cargo:warning=Creating static library: {}", lib_path.display());
    
    // Use ar to create static library
    let mut ar_cmd = Command::new("ar");
    ar_cmd.arg("crs").arg(&lib_path);
    for obj in &objects {
        ar_cmd.arg(obj);
    }
    ar_cmd.arg(&dlink_obj);  // Include device link object
    
    let status = ar_cmd.status().expect("Failed to create static library");
    if !status.success() {
        panic!("Failed to create static library");
    }
    
    // Step 4: Tell Rust to link everything
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static={}", lib_name);
    
    // CUDA libraries (order matters!)
    println!("cargo:rustc-link-lib=cudadevrt");  // CRITICAL: Required for RDC
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cusparse");
    println!("cargo:rustc-link-lib=cufft");
    println!("cargo:rustc-link-lib=curand");
    
    // C++ standard library for Thrust/CUB symbols
    println!("cargo:rustc-link-lib=stdc++");
    
    // CUDA library search path
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda-13.0/lib64");
    
    // Configure for GPU-only mode
    println!("cargo:rustc-cfg=cuda_available");
    
    // Rebuild triggers
    for src in cu_srcs {
        println!("cargo:rerun-if-changed={}", src);
    }
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=GPU_SM");
    println!("cargo:rerun-if-env-changed=GPU_COMPUTE");
    println!("cargo:rerun-if-env-changed=CUDA_BIN_PATH");
    
    println!("cargo:warning=✅ GPU-native CUDA compilation with RDC successful");
}