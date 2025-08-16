//! CUDA FFI bindings and wrappers

use crate::error::{CompilerError, Result};
use std::ffi::{c_int, c_void};

// External CUDA functions (these will be implemented in CUDA)
extern "C" {
    fn cuda_initialize() -> c_int;
    fn cuda_cleanup() -> c_int;
    fn cuda_get_device_count() -> c_int;
    fn cuda_get_device_memory_size(device: c_int) -> usize;
    fn cuda_malloc_device(size: usize) -> *mut c_void;
    fn cuda_free_device(ptr: *mut c_void) -> c_int;
    fn cuda_memcpy_host_to_device(dst: *mut c_void, src: *const c_void, size: usize) -> c_int;
    fn cuda_memcpy_device_to_host(dst: *mut c_void, src: *const c_void, size: usize) -> c_int;
    fn cuda_synchronize() -> c_int;
    fn cuda_get_last_error() -> c_int;
    fn cuda_get_error_string(error: c_int) -> *const i8;
    // Additional functions needed by rustfmt-g  
    fn cuda_get_device_properties(device: c_int, properties_out: *mut c_void) -> c_int;
    fn cuda_format_lines(input: *const i8, input_len: usize, output: *mut i8, output_len: *mut usize) -> c_int;
}

/// Initialize CUDA runtime
pub fn initialize_cuda() -> Result<()> {
    let result = unsafe { cuda_initialize() };
    check_cuda_error(result)?;
    
    let device_count = unsafe { cuda_get_device_count() };
    if device_count <= 0 {
        return Err(CompilerError::Cuda("No CUDA devices found".to_string()));
    }
    
    tracing::info!("CUDA initialized with {} device(s)", device_count);
    Ok(())
}

/// Clean up CUDA runtime
pub fn cleanup_cuda() -> Result<()> {
    let result = unsafe { cuda_cleanup() };
    check_cuda_error(result)
}

/// Get available device memory size
pub unsafe fn get_device_memory_size() -> Result<usize> {
    let size = cuda_get_device_memory_size(0); // Use device 0 for now
    if size == 0 {
        return Err(CompilerError::Cuda("Failed to get device memory size".to_string()));
    }
    Ok(size)
}

/// Allocate device memory
pub unsafe fn cuda_malloc(size: usize) -> Result<*mut u8> {
    let ptr = cuda_malloc_device(size);
    if ptr.is_null() {
        let error = cuda_get_last_error();
        check_cuda_error(error)?;
        return Err(CompilerError::GpuMemory(format!(
            "Failed to allocate {} bytes on GPU",
            size
        )));
    }
    Ok(ptr as *mut u8)
}

/// Free device memory
pub unsafe fn cuda_free(ptr: *mut u8) -> Result<()> {
    let result = cuda_free_device(ptr as *mut c_void);
    check_cuda_error(result)
}

/// Free device memory (C-compatible version)
pub unsafe fn cuda_free_c(ptr: *mut c_void) -> c_int {
    cuda_free_device(ptr)
}

/// Copy memory from host to device
pub fn copy_to_device<T>(dst: *mut c_void, src: &[T]) -> Result<()> {
    let size = std::mem::size_of_val(src);
    let result = unsafe {
        cuda_memcpy_host_to_device(
            dst,
            src.as_ptr() as *const c_void,
            size
        )
    };
    check_cuda_error(result)
}

/// Copy memory from device to host
pub fn copy_from_device<T>(dst: &mut [T], src: *const c_void) -> Result<()> {
    let size = std::mem::size_of_val(dst);
    let result = unsafe {
        cuda_memcpy_device_to_host(
            dst.as_mut_ptr() as *mut c_void,
            src,
            size
        )
    };
    check_cuda_error(result)
}

/// Synchronize device
pub fn synchronize_device() -> Result<()> {
    let result = unsafe { cuda_synchronize() };
    check_cuda_error(result)
}

/// Check CUDA error code
fn check_cuda_error(code: c_int) -> Result<()> {
    if code != 0 {
        let error_str = unsafe {
            let ptr = cuda_get_error_string(code);
            if ptr.is_null() {
                format!("Unknown CUDA error: {}", code)
            } else {
                std::ffi::CStr::from_ptr(ptr)
                    .to_string_lossy()
                    .to_string()
            }
        };
        Err(CompilerError::Cuda(error_str))
    } else {
        Ok(())
    }
}

/// Get device properties
pub fn get_device_properties(device: i32) -> Result<DeviceProperties> {
    let mut props = std::mem::MaybeUninit::<[u8; 1024]>::uninit(); // Large enough for cudaDeviceProp
    let result = unsafe { 
        cuda_get_device_properties(device, props.as_mut_ptr() as *mut c_void) 
    };
    
    check_cuda_error(result)?;
    
    // For now, return default properties
    // In a real implementation, we'd parse the returned cudaDeviceProp struct
    Ok(DeviceProperties {
        name: "RTX 5090".to_string(),
        total_memory: 24 * 1024 * 1024 * 1024, // 24GB
        compute_capability_major: 11,
        compute_capability_minor: 0,
        max_threads_per_block: 1024,
        max_blocks_per_multiprocessor: 32,
        multiprocessor_count: 128,
        warp_size: 32,
        max_shared_memory_per_block: 48 * 1024,
    })
}

/// Format lines using GPU acceleration
pub fn format_lines_gpu(input: &str) -> Result<String> {
    let input_len = input.len();
    let mut output_buffer = vec![0u8; input_len * 2]; // Allocate extra space for formatting
    let mut output_len = output_buffer.len();
    
    let result = unsafe {
        cuda_format_lines(
            input.as_ptr() as *const i8,
            input_len,
            output_buffer.as_mut_ptr() as *mut i8,
            &mut output_len as *mut usize,
        )
    };
    
    check_cuda_error(result)?;
    
    // Convert back to string
    output_buffer.truncate(output_len);
    String::from_utf8(output_buffer)
        .map_err(|e| CompilerError::InvalidUtf8(e.to_string()))
}

/// Device properties
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    pub name: String,
    pub total_memory: usize,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
    pub max_threads_per_block: i32,
    pub max_blocks_per_multiprocessor: i32,
    pub multiprocessor_count: i32,
    pub warp_size: i32,
    pub max_shared_memory_per_block: usize,
}