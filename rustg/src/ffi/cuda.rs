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