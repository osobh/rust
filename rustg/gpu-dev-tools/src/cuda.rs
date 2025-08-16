// CUDA Context and Utilities for GPU Development Tools
// Manages GPU resources and kernel execution

use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;

/// CUDA context wrapper for GPU operations
pub struct CudaContext {
    device_id: i32,
    device_props: DeviceProperties,
    memory_pools: Arc<Mutex<MemoryPools>>,
}

#[derive(Debug, Clone)]
pub struct DeviceProperties {
    pub name: String,
    pub compute_capability: (i32, i32),
    pub total_memory: usize,
    pub multiprocessor_count: i32,
    pub max_threads_per_block: i32,
    pub max_blocks_per_multiprocessor: i32,
    pub warp_size: i32,
}

struct MemoryPools {
    small_pool: Vec<*mut u8>,  // <1KB allocations
    medium_pool: Vec<*mut u8>, // 1KB-1MB allocations
    large_pool: Vec<*mut u8>,  // >1MB allocations
}

impl CudaContext {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Initialize CUDA
        let device_id = 0; // Use first GPU
        let device_props = Self::get_device_properties(device_id)?;
        
        Ok(CudaContext {
            device_id,
            device_props,
            memory_pools: Arc::new(Mutex::new(MemoryPools {
                small_pool: Vec::new(),
                medium_pool: Vec::new(),
                large_pool: Vec::new(),
            })),
        })
    }

    pub fn with_device(device_id: i32) -> Result<Self, Box<dyn std::error::Error>> {
        let device_props = Self::get_device_properties(device_id)?;
        
        Ok(CudaContext {
            device_id,
            device_props,
            memory_pools: Arc::new(Mutex::new(MemoryPools {
                small_pool: Vec::new(),
                medium_pool: Vec::new(),
                large_pool: Vec::new(),
            })),
        })
    }

    fn get_device_properties(device_id: i32) -> Result<DeviceProperties, Box<dyn std::error::Error>> {
        // Use the CUDA function from rustg crate
        // We need to declare it here since it's a C function
        extern "C" {
            fn cuda_get_device_properties(
                device_id: i32,
                name: *mut u8,
                major: *mut i32,
                minor: *mut i32,
                total_mem: *mut usize,
                mp_count: *mut i32,
                max_threads: *mut i32,
                max_blocks: *mut i32,
                warp_size: *mut i32,
            ) -> i32;
        }

        let mut name_buf = vec![0u8; 256];
        let mut major = 0i32;
        let mut minor = 0i32;
        let mut total_mem = 0usize;
        let mut mp_count = 0i32;
        let mut max_threads = 0i32;
        let mut max_blocks = 0i32;
        let mut warp_size = 0i32;

        unsafe {
            let result = cuda_get_device_properties(
                device_id,
                name_buf.as_mut_ptr(),
                &mut major,
                &mut minor,
                &mut total_mem,
                &mut mp_count,
                &mut max_threads,
                &mut max_blocks,
                &mut warp_size,
            );

            if result != 0 {
                return Err("Failed to get device properties".into());
            }
        }

        let name = String::from_utf8_lossy(&name_buf)
            .trim_end_matches('\0')
            .to_string();

        Ok(DeviceProperties {
            name,
            compute_capability: (major, minor),
            total_memory: total_mem,
            multiprocessor_count: mp_count,
            max_threads_per_block: max_threads,
            max_blocks_per_multiprocessor: max_blocks,
            warp_size,
        })
    }

    /// Allocate GPU memory from pool
    pub fn allocate(&self, size: usize) -> Result<*mut u8, Box<dyn std::error::Error>> {
        // Use the CUDA function from rustg crate
        extern "C" {
            fn cuda_malloc(size: usize) -> *mut u8;
        }

        let mut pools = self.memory_pools.lock().unwrap();
        
        // Check pools first
        let ptr = if size < 1024 {
            pools.small_pool.pop()
        } else if size < 1024 * 1024 {
            pools.medium_pool.pop()
        } else {
            pools.large_pool.pop()
        };

        if let Some(ptr) = ptr {
            Ok(ptr)
        } else {
            // Allocate new
            let ptr = unsafe { cuda_malloc(size) };
            if ptr.is_null() {
                Err("GPU memory allocation failed".into())
            } else {
                Ok(ptr)
            }
        }
    }

    /// Free GPU memory back to pool
    pub fn free(&self, ptr: *mut u8, size: usize) {
        let mut pools = self.memory_pools.lock().unwrap();
        
        if size < 1024 {
            pools.small_pool.push(ptr);
        } else if size < 1024 * 1024 {
            pools.medium_pool.push(ptr);
        } else {
            pools.large_pool.push(ptr);
        }
    }

    /// Copy data to GPU
    pub fn copy_to_device<T>(&self, data: &[T]) -> Result<*mut T, Box<dyn std::error::Error>> {
        extern "C" {
            fn cuda_memcpy_host_to_device(
                dst: *mut u8,
                src: *const u8,
                size: usize,
            ) -> i32;
        }

        let size = std::mem::size_of_val(data);
        let device_ptr = self.allocate(size)? as *mut T;

        unsafe {
            let result = cuda_memcpy_host_to_device(
                device_ptr as *mut u8,
                data.as_ptr() as *const u8,
                size,
            );

            if result != 0 {
                return Err("Failed to copy data to device".into());
            }
        }

        Ok(device_ptr)
    }

    /// Copy data from GPU
    pub fn copy_from_device<T>(&self, device_ptr: *const T, count: usize) -> Result<Vec<T>, Box<dyn std::error::Error>> 
    where
        T: Clone + Default,
    {
        extern "C" {
            fn cuda_memcpy_device_to_host(
                dst: *mut u8,
                src: *const u8,
                size: usize,
            ) -> i32;
        }

        let mut host_data = vec![T::default(); count];
        let size = count * std::mem::size_of::<T>();

        unsafe {
            let result = cuda_memcpy_device_to_host(
                host_data.as_mut_ptr() as *mut u8,
                device_ptr as *const u8,
                size,
            );

            if result != 0 {
                return Err("Failed to copy data from device".into());
            }
        }

        Ok(host_data)
    }

    /// Launch kernel with configuration
    pub fn launch_kernel(
        &self,
        kernel_name: &str,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        args: &[*mut u8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        extern "C" {
            fn cuda_launch_kernel(
                kernel_name: *const u8,
                grid_x: u32,
                grid_y: u32,
                grid_z: u32,
                block_x: u32,
                block_y: u32,
                block_z: u32,
                args: *const *mut u8,
                arg_count: i32,
            ) -> i32;
        }

        let kernel_cstr = std::ffi::CString::new(kernel_name)?;

        unsafe {
            let result = cuda_launch_kernel(
                kernel_cstr.as_ptr() as *const u8,
                grid_dim.0,
                grid_dim.1,
                grid_dim.2,
                block_dim.0,
                block_dim.1,
                block_dim.2,
                args.as_ptr(),
                args.len() as i32,
            );

            if result != 0 {
                return Err(format!("Failed to launch kernel {}", kernel_name).into());
            }
        }

        Ok(())
    }

    /// Synchronize device
    pub fn synchronize(&self) -> Result<(), Box<dyn std::error::Error>> {
        extern "C" {
            fn cuda_device_synchronize() -> i32;
        }

        unsafe {
            let result = cuda_device_synchronize();
            if result != 0 {
                return Err("Device synchronization failed".into());
            }
        }

        Ok(())
    }

    /// Get last CUDA error
    pub fn get_last_error(&self) -> Option<String> {
        extern "C" {
            fn cuda_get_last_error_msg(msg: *mut u8, max_len: i32) -> i32;
        }

        let mut msg_buf = vec![0u8; 256];
        
        unsafe {
            let result = cuda_get_last_error_msg(msg_buf.as_mut_ptr(), 256);
            if result != 0 {
                let msg = String::from_utf8_lossy(&msg_buf)
                    .trim_end_matches('\0')
                    .to_string();
                Some(msg)
            } else {
                None
            }
        }
    }

    /// Calculate optimal launch configuration
    pub fn calculate_launch_config(&self, total_elements: usize) -> (u32, u32) {
        let threads_per_block = 256;
        let blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        
        // Limit to device capabilities
        let max_blocks = self.device_props.multiprocessor_count as usize * 
                        self.device_props.max_blocks_per_multiprocessor as usize;
        
        let blocks = blocks.min(max_blocks);
        
        (blocks as u32, threads_per_block as u32)
    }

    /// Get device properties
    pub fn get_properties(&self) -> &DeviceProperties {
        &self.device_props
    }

    /// Check if GPU supports required compute capability
    pub fn check_compute_capability(&self, major: i32, minor: i32) -> bool {
        let (dev_major, dev_minor) = self.device_props.compute_capability;
        dev_major > major || (dev_major == major && dev_minor >= minor)
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        // Use the CUDA function from rustg crate
        extern "C" {
            fn cuda_free(ptr: *mut u8);
        }

        // Free pooled memory
        let pools = self.memory_pools.lock().unwrap();
        
        for ptr in &pools.small_pool {
            unsafe { cuda_free(*ptr); }
        }
        for ptr in &pools.medium_pool {
            unsafe { cuda_free(*ptr); }
        }
        for ptr in &pools.large_pool {
            unsafe { cuda_free(*ptr); }
        }
    }
}

/// Performance timer for GPU operations
pub struct GpuTimer {
    start_event: *mut u8,
    stop_event: *mut u8,
}

impl GpuTimer {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        extern "C" {
            fn cuda_create_event() -> *mut u8;
        }

        unsafe {
            let start = cuda_create_event();
            let stop = cuda_create_event();
            
            if start.is_null() || stop.is_null() {
                return Err("Failed to create CUDA events".into());
            }
            
            Ok(GpuTimer {
                start_event: start,
                stop_event: stop,
            })
        }
    }

    pub fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        extern "C" {
            fn cuda_record_event(event: *mut u8) -> i32;
        }

        unsafe {
            let result = cuda_record_event(self.start_event);
            if result != 0 {
                return Err("Failed to record start event".into());
            }
        }
        
        Ok(())
    }

    pub fn stop(&self) -> Result<(), Box<dyn std::error::Error>> {
        extern "C" {
            fn cuda_record_event(event: *mut u8) -> i32;
        }

        unsafe {
            let result = cuda_record_event(self.stop_event);
            if result != 0 {
                return Err("Failed to record stop event".into());
            }
        }
        
        Ok(())
    }

    pub fn elapsed_ms(&self) -> Result<f32, Box<dyn std::error::Error>> {
        extern "C" {
            fn cuda_event_elapsed_time(
                start: *mut u8,
                stop: *mut u8,
                time_ms: *mut f32,
            ) -> i32;
        }

        let mut time_ms = 0.0f32;
        
        unsafe {
            let result = cuda_event_elapsed_time(
                self.start_event,
                self.stop_event,
                &mut time_ms,
            );
            
            if result != 0 {
                return Err("Failed to get elapsed time".into());
            }
        }
        
        Ok(time_ms)
    }
}

impl Drop for GpuTimer {
    fn drop(&mut self) {
        extern "C" {
            fn cuda_destroy_event(event: *mut u8);
        }

        unsafe {
            cuda_destroy_event(self.start_event);
            cuda_destroy_event(self.stop_event);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_context_creation() {
        let context = CudaContext::new();
        assert!(context.is_ok());
    }

    #[test]
    fn test_device_properties() {
        if let Ok(context) = CudaContext::new() {
            let props = context.get_properties();
            assert!(props.warp_size == 32);
            assert!(props.max_threads_per_block >= 512);
        }
    }

    #[test]
    fn test_launch_config() {
        if let Ok(context) = CudaContext::new() {
            let (blocks, threads) = context.calculate_launch_config(10000);
            assert!(blocks > 0);
            assert!(threads > 0);
            assert!(threads <= 1024);
        }
    }
}