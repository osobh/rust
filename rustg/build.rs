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

    // Check if CUDA is available
    let cuda_available = check_cuda_available();
    
    if !cuda_available {
        println!("cargo:warning=CUDA not found, building CPU-only version");
        println!("cargo:rustc-cfg=feature=\"cpu-fallback\"");
        // For now, we'll skip CUDA compilation if not available
        // In production, we'd build CPU fallback implementations
        return;
    }

    // Create build directory
    let build_dir = out_dir.join("cuda_build");
    std::fs::create_dir_all(&build_dir).expect("Failed to create build directory");

    // Configure with CMake
    let cmake_status = Command::new("cmake")
        .current_dir(&build_dir)
        .arg(&manifest_dir)
        .arg(format!("-DCMAKE_BUILD_TYPE={}", build_type))
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
    println!("cargo:rustc-link-search=native={}", build_dir.join("lib").display());
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
    
    // Generate Rust bindings for C++ headers
    generate_bindings(&manifest_dir, &out_dir);
}

fn check_cuda_available() -> bool {
    // Check if nvcc is available
    Command::new("nvcc")
        .arg("--version")
        .output()
        .is_ok()
}

fn generate_bindings(manifest_dir: &Path, out_dir: &Path) {
    let bindings = bindgen::Builder::default()
        .header(manifest_dir.join("include/gpu_types.h").to_str().unwrap())
        .clang_arg("-I")
        .clang_arg(manifest_dir.join("include").to_str().unwrap())
        .clang_arg("-std=c++17")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate_comments(true)
        .derive_default(true)
        .derive_debug(true)
        .derive_copy(true)
        .size_t_is_usize(true)
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(out_dir.join("cuda_bindings.rs"))
        .expect("Couldn't write bindings");
}