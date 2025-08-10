// CUDA bindings and kernel management
// Provides low-level GPU access for test harness

use anyhow::{Result, Context};
use std::ffi::{CString, CStr};
use std::ptr;
use std::os::raw::c_void;

// Include generated CUDA bindings
include!(concat!(env!("OUT_DIR"), "/cuda_bindings.rs"));

// CUDA error handling
pub fn check_cuda_error(err: cudaError_t) -> Result<()> {
    if err != cudaError_cudaSuccess {
        let err_str = unsafe {
            CStr::from_ptr(cudaGetErrorString(err))
                .to_string_lossy()
        };
        anyhow::bail!("CUDA error: {}", err_str);
    }
    Ok(())
}

// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub id: i32,
    pub name: String,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
    pub total_memory_mb: usize,
    pub multiprocessor_count: i32,
    pub max_threads_per_block: i32,
    pub max_threads_per_multiprocessor: i32,
    pub warp_size: i32,
    pub shared_memory_per_block: usize,
    pub supports_managed_memory: bool,
    pub supports_concurrent_kernels: bool,
}

impl GpuDevice {
    pub fn enumerate() -> Result<Vec<Self>> {
        let mut device_count = 0;
        unsafe {
            check_cuda_error(cudaGetDeviceCount(&mut device_count))?;
        }
        
        let mut devices = Vec::new();
        for i in 0..device_count {
            let device = Self::get_device(i)?;
            devices.push(device);
        }
        
        Ok(devices)
    }
    
    pub fn get_device(id: i32) -> Result<Self> {
        let mut prop: cudaDeviceProp = unsafe { std::mem::zeroed() };
        unsafe {
            check_cuda_error(cudaGetDeviceProperties(&mut prop, id))?;
        }
        
        let name = unsafe {
            CStr::from_ptr(prop.name.as_ptr())
                .to_string_lossy()
                .into_owned()
        };
        
        Ok(Self {
            id,
            name,
            compute_capability_major: prop.major,
            compute_capability_minor: prop.minor,
            total_memory_mb: prop.totalGlobalMem / (1024 * 1024),
            multiprocessor_count: prop.multiProcessorCount,
            max_threads_per_block: prop.maxThreadsPerBlock,
            max_threads_per_multiprocessor: prop.maxThreadsPerMultiProcessor,
            warp_size: prop.warpSize,
            shared_memory_per_block: prop.sharedMemPerBlock,
            supports_managed_memory: prop.managedMemory != 0,
            supports_concurrent_kernels: prop.concurrentKernels != 0,
        })
    }
    
    pub fn set_current(&self) -> Result<()> {
        unsafe {
            check_cuda_error(cudaSetDevice(self.id))?;
        }
        Ok(())
    }
    
    pub fn get_compute_capability(&self) -> u32 {
        (self.compute_capability_major * 10 + self.compute_capability_minor) as u32
    }
}

// CUDA memory management
pub struct CudaMemory<T> {
    ptr: *mut T,
    size: usize,
}

impl<T> CudaMemory<T> {
    pub fn allocate(count: usize) -> Result<Self> {
        let size = count * std::mem::size_of::<T>();
        let mut ptr: *mut c_void = ptr::null_mut();
        
        unsafe {
            check_cuda_error(cudaMalloc(&mut ptr, size))?;
        }
        
        Ok(Self {
            ptr: ptr as *mut T,
            size: count,
        })
    }
    
    pub fn copy_from_host(&mut self, data: &[T]) -> Result<()> {
        let size = data.len() * std::mem::size_of::<T>();
        unsafe {
            check_cuda_error(cudaMemcpy(
                self.ptr as *mut c_void,
                data.as_ptr() as *const c_void,
                size,
                cudaMemcpyKind_cudaMemcpyHostToDevice,
            ))?;
        }
        Ok(())
    }
    
    pub fn copy_to_host(&self, data: &mut [T]) -> Result<()> {
        let size = data.len() * std::mem::size_of::<T>();
        unsafe {
            check_cuda_error(cudaMemcpy(
                data.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                size,
                cudaMemcpyKind_cudaMemcpyDeviceToHost,
            ))?;
        }
        Ok(())
    }
    
    pub fn as_ptr(&self) -> *mut T {
        self.ptr
    }
}

impl<T> Drop for CudaMemory<T> {
    fn drop(&mut self) {
        unsafe {
            let _ = cudaFree(self.ptr as *mut c_void);
        }
    }
}

// CUDA stream management
pub struct CudaStream {
    stream: cudaStream_t,
}

impl CudaStream {
    pub fn new() -> Result<Self> {
        let mut stream: cudaStream_t = ptr::null_mut();
        unsafe {
            check_cuda_error(cudaStreamCreate(&mut stream))?;
        }
        Ok(Self { stream })
    }
    
    pub fn synchronize(&self) -> Result<()> {
        unsafe {
            check_cuda_error(cudaStreamSynchronize(self.stream))?;
        }
        Ok(())
    }
    
    pub fn as_ptr(&self) -> cudaStream_t {
        self.stream
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        unsafe {
            let _ = cudaStreamDestroy(self.stream);
        }
    }
}

// CUDA event management
pub struct CudaEvent {
    event: cudaEvent_t,
}

impl CudaEvent {
    pub fn new() -> Result<Self> {
        let mut event: cudaEvent_t = ptr::null_mut();
        unsafe {
            check_cuda_error(cudaEventCreate(&mut event))?;
        }
        Ok(Self { event })
    }
    
    pub fn record(&self, stream: &CudaStream) -> Result<()> {
        unsafe {
            check_cuda_error(cudaEventRecord(self.event, stream.as_ptr()))?;
        }
        Ok(())
    }
    
    pub fn synchronize(&self) -> Result<()> {
        unsafe {
            check_cuda_error(cudaEventSynchronize(self.event))?;
        }
        Ok(())
    }
    
    pub fn elapsed_time(&self, start: &CudaEvent) -> Result<f32> {
        let mut ms = 0.0f32;
        unsafe {
            check_cuda_error(cudaEventElapsedTime(&mut ms, start.event, self.event))?;
        }
        Ok(ms)
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        unsafe {
            let _ = cudaEventDestroy(self.event);
        }
    }
}

// PTX module loading
pub struct PtxModule {
    module: cudaModule_t,
}

impl PtxModule {
    pub fn load_from_file(path: &str) -> Result<Self> {
        let c_path = CString::new(path)?;
        let mut module: cudaModule_t = ptr::null_mut();
        
        unsafe {
            check_cuda_error(cudaModuleLoad(&mut module, c_path.as_ptr()))?;
        }
        
        Ok(Self { module })
    }
    
    pub fn load_from_ptx(ptx: &str) -> Result<Self> {
        let c_ptx = CString::new(ptx)?;
        let mut module: cudaModule_t = ptr::null_mut();
        
        unsafe {
            check_cuda_error(cudaModuleLoadData(&mut module, c_ptx.as_ptr() as *const c_void))?;
        }
        
        Ok(Self { module })
    }
    
    pub fn get_function(&self, name: &str) -> Result<cudaFunction_t> {
        let c_name = CString::new(name)?;
        let mut function: cudaFunction_t = ptr::null_mut();
        
        unsafe {
            check_cuda_error(cudaModuleGetFunction(&mut function, self.module, c_name.as_ptr()))?;
        }
        
        Ok(function)
    }
}

impl Drop for PtxModule {
    fn drop(&mut self) {
        unsafe {
            let _ = cudaModuleUnload(self.module);
        }
    }
}

// Kernel launch configuration
#[derive(Debug, Clone)]
pub struct LaunchConfig {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem: usize,
    pub stream: Option<cudaStream_t>,
}

impl LaunchConfig {
    pub fn new(grid: (u32, u32, u32), block: (u32, u32, u32)) -> Self {
        Self {
            grid_dim: grid,
            block_dim: block,
            shared_mem: 0,
            stream: None,
        }
    }
    
    pub fn with_shared_memory(mut self, bytes: usize) -> Self {
        self.shared_mem = bytes;
        self
    }
    
    pub fn with_stream(mut self, stream: cudaStream_t) -> Self {
        self.stream = Some(stream);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_device_enumeration() {
        // Will test if CUDA is available
        if let Ok(devices) = GpuDevice::enumerate() {
            assert!(!devices.is_empty());
            for device in devices {
                println!("Found GPU: {} (CC {}.{})", 
                        device.name,
                        device.compute_capability_major,
                        device.compute_capability_minor);
            }
        }
    }
}