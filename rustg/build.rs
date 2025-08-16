use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=CMakeLists.txt");
    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=include/");

    // Get the output directory
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    
    // Build configuration
    let profile = env::var("PROFILE").unwrap();
    let build_type = match profile.as_str() {
        "debug" => "Debug",
        "release" => "Release",
        _ => "Release",
    };

    // Check if CUDA is available and validate environment
    let cuda_available = check_cuda_available();
    
    if !cuda_available {
        panic!("CUDA 13.0+ required for rustg GPU-native compilation. Please install CUDA toolkit.");
    }
    
    // Check driver version and GPU capabilities
    check_driver_and_gpu();
    
    // Check for open kernel modules on Blackwell GPUs
    check_blackwell_kernel_modules();

    // Create build directory
    let build_dir = out_dir.join("cuda_build");
    std::fs::create_dir_all(&build_dir).expect("Failed to create build directory");

    // Configure with CMake (disable tests by default to avoid CUDA compatibility issues)
    let enable_tests = env::var("RUSTG_BUILD_TESTS").unwrap_or("OFF".to_string());
    let cmake_status = Command::new("cmake")
        .current_dir(&build_dir)
        .arg(&manifest_dir)
        .arg(format!("-DCMAKE_BUILD_TYPE={}", build_type))
        .arg(format!("-DBUILD_TESTS={}", enable_tests))
        .arg("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")
        .status()
        .expect("Failed to run cmake");

    if !cmake_status.success() {
        panic!("CMake configuration failed");
    }

    // Build with make
    let make_status = Command::new("make")
        .current_dir(&build_dir)
        .arg("-j")
        .arg(num_cpus::get().to_string())
        .status()
        .expect("Failed to run make");

    if !make_status.success() {
        panic!("Make build failed");
    }

    // Link the built CUDA library
    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=rustg_core");
    
    // Link CUDA runtime libraries
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let cuda_lib_path = PathBuf::from(cuda_path).join("lib64");
        println!("cargo:rustc-link-search=native={}", cuda_lib_path.display());
    } else {
        // Try common CUDA installation paths
        for cuda_lib in &["/usr/local/cuda/lib64", "/opt/cuda/lib64"] {
            if Path::new(cuda_lib).exists() {
                println!("cargo:rustc-link-search=native={}", cuda_lib);
                break;
            }
        }
    }
    
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=stdc++");
    
    // Generate Rust bindings for C++ headers
    generate_bindings(&manifest_dir, &out_dir);
}

fn check_cuda_available() -> bool {
    // Try multiple paths to find nvcc
    let nvcc_paths = vec![
        "/usr/local/cuda-13.0/bin/nvcc",
        "/usr/local/cuda/bin/nvcc",
        "nvcc",
    ];
    
    for nvcc_path in nvcc_paths {
        if let Ok(output) = Command::new(&nvcc_path).arg("--version").output() {
            let version_str = String::from_utf8_lossy(&output.stdout);
            if version_str.contains("release 13.") || version_str.contains("release 14.") {
                println!("cargo:warning=Found CUDA 13.0+ compiler at {}", nvcc_path);
                // Set CUDA_PATH for cmake
                if nvcc_path.contains("cuda-13.0") {
                    env::set_var("CUDA_PATH", "/usr/local/cuda-13.0");
                }
                return true;
            }
            println!("cargo:warning=CUDA version too old at {}, need 13.0+", nvcc_path);
        }
    }
    false
}

fn check_driver_and_gpu() {
    // Check NVIDIA driver version
    if let Ok(output) = Command::new("nvidia-smi")
        .arg("--query-gpu=driver_version,name,compute_cap")
        .arg("--format=csv,noheader")
        .output() 
    {
        let info = String::from_utf8_lossy(&output.stdout);
        println!("cargo:warning=GPU Info: {}", info.trim());
        
        // Parse driver version
        let parts: Vec<&str> = info.split(',').collect();
        if parts.len() >= 3 {
            let driver_version = parts[0].trim();
            let gpu_name = parts[1].trim();
            let compute_cap = parts[2].trim();
            
            // Check for RTX 5090 (Blackwell)
            if gpu_name.contains("RTX 5090") || gpu_name.contains("RTX 50") {
                println!("cargo:warning=Detected RTX 5090 (Blackwell) - using sm_110");
                println!("cargo:rustc-cfg=blackwell_gpu");
                
                // Validate driver version for Blackwell (need 580+)
                if let Some(version) = driver_version.split('.').next() {
                    if let Ok(ver_num) = version.parse::<u32>() {
                        if ver_num < 580 {
                            panic!("RTX 5090 requires driver 580.65.06 or newer, found {}", driver_version);
                        }
                    }
                }
            }
            
            println!("cargo:warning=Driver: {}, GPU: {}, Compute: {}", 
                     driver_version, gpu_name, compute_cap);
        }
    } else {
        println!("cargo:warning=Could not detect GPU, assuming RTX 5090 target");
    }
}

fn check_blackwell_kernel_modules() {
    // Check if using open kernel modules (required for Blackwell)
    if let Ok(output) = Command::new("lsmod").output() {
        let modules = String::from_utf8_lossy(&output.stdout);
        
        if modules.contains("nvidia_drm") && modules.contains("nvidia_modeset") {
            // Check if using open modules
            if let Ok(params) = std::fs::read_to_string("/sys/module/nvidia/version") {
                if params.contains("Open") {
                    println!("cargo:warning=✓ Using NVIDIA open kernel modules (required for Blackwell)");
                    return;
                }
            }
        }
        
        // On Blackwell, warn if not using open modules
        if cfg!(blackwell_gpu) {
            println!("cargo:warning=⚠ RTX 5090 (Blackwell) requires open GPU kernel modules");
            println!("cargo:warning=⚠ Install with: sudo apt install nvidia-driver-580-open");
        }
    }
}

fn generate_bindings(manifest_dir: &Path, out_dir: &Path) {
    // Simple fallback: just create a basic binding for the types we need
    // This avoids complex clang configuration issues
    let bindings = bindgen::Builder::default()
        .header(manifest_dir.join("include/gpu_types.h").to_str().unwrap())
        .clang_arg("-I")
        .clang_arg(manifest_dir.join("include").to_str().unwrap())
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_type("rustg::.*")
        .allowlist_var("rustg::.*")
        .allowlist_function("rustg::.*")
        .derive_default(true)
        .derive_debug(true)
        .derive_copy(true)
        .size_t_is_usize(true)
        .generate()
        .unwrap_or_else(|_| {
            // Fallback: Create minimal bindings manually
            println!("cargo:warning=Bindgen failed, creating minimal bindings");
            bindgen::Builder::default()
                .header_contents("gpu_types_minimal.h", r#"
                    #include <stdint.h>
                    // Minimal type definitions for rustg
                    typedef uint32_t u32;
                    typedef uint8_t u8;
                "#)
                .generate()
                .expect("Failed to generate minimal bindings")
        });

    bindings
        .write_to_file(out_dir.join("cuda_bindings.rs"))
        .expect("Couldn't write bindings");
}