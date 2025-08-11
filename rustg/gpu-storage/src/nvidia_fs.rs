// NVIDIA GPUDirect Storage (nvidia-fs) FFI Bindings
// Enhanced for CUDA 13.0 with cuFile 1.15.0.42
// Provides Rust bindings for cuFile API for direct GPU-storage transfers
// RTX 5090: Optimized for 1.5TB/s memory bandwidth

use std::os::raw::{c_int, c_void, c_char, c_ulong};
use std::ffi::CString;
use std::path::Path;
use anyhow::{Result, Context};

/// cuFile handle type
#[repr(C)]
pub struct CUfileHandle_t {
    _private: [u8; 0],
}

/// File descriptor type for cuFile
#[repr(C)]
pub struct CUfileDescr_t {
    pub handle: CUfileHandle_u,
    pub handle_type: CUfileHandleType,
}

/// Union for file handle
#[repr(C)]
pub union CUfileHandle_u {
    pub fd: c_int,
    pub handle: *mut c_void,
}

/// Handle type enumeration
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CUfileHandleType {
    CU_FILE_HANDLE_TYPE_OPAQUE_FD = 1,
    CU_FILE_HANDLE_TYPE_OPAQUE_WIN32 = 2,
    CU_FILE_HANDLE_TYPE_OPAQUE_NVFS = 3,
}

/// cuFile error codes (Updated for CUDA 13.0)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CUfileError_t {
    CU_FILE_SUCCESS = 0,
    CU_FILE_DRIVER_NOT_INITIALIZED = -1,
    CU_FILE_DRIVER_ALREADY_CLOSED = -2,
    CU_FILE_DRIVER_VERSION_MISMATCH = -3,
    CU_FILE_INVALID_VALUE = -4,
    CU_FILE_MEMORY_ALLOCATION_FAILED = -5,
    CU_FILE_NOT_SUPPORTED = -6,
    CU_FILE_PERMISSION_DENIED = -7,
    CU_FILE_NVFS_DRIVER_ERROR = -8,
    CU_FILE_CUDA_ERROR = -9,
    CU_FILE_OS_CALL_ERROR = -10,
    CU_FILE_IO_ERROR = -11,
    // CUDA 13.0 additions
    CU_FILE_BUFFER_NOT_REGISTERED = -12,
    CU_FILE_INVALID_MAPPING = -13,
    CU_FILE_BATCH_FULL = -14,
    CU_FILE_ASYNC_NOT_SUPPORTED = -15,
}

/// cuFile driver properties
#[repr(C)]
pub struct CUfileDrvProps_t {
    pub max_direct_io_size: usize,
    pub max_device_cache_size: usize,
    pub max_device_pinned_mem_size: usize,
    pub max_batch_io_size: usize,
    pub max_batch_io_timeout_msecs: c_int,
}

/// Batch I/O parameters
#[repr(C)]
pub struct CUfileBatchHandle_t {
    _private: [u8; 0],
}

/// I/O parameters for batch operations
#[repr(C)]
pub struct CUfileIOParams_t {
    pub mode: c_int,
    pub u: CUfileIOParams_u,
    pub fh: *mut CUfileHandle_t,
    pub opcode: CUfileOpcode_t,
    pub cookie: *mut c_void,
}

#[repr(C)]
pub union CUfileIOParams_u {
    pub buf: CUfileIOBufParams_t,
}

#[repr(C)]
pub struct CUfileIOBufParams_t {
    pub devPtr_base: *mut c_void,
    pub devPtr_offset: usize,
    pub size: usize,
    pub file_offset: i64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CUfileOpcode_t {
    CU_FILE_READ = 0,
    CU_FILE_WRITE = 1,
}

// External C functions from libcufile.so
#[link(name = "cufile")]
extern "C" {
    // Driver management
    pub fn cuFileDriverOpen() -> CUfileError_t;
    pub fn cuFileDriverClose() -> CUfileError_t;
    pub fn cuFileDriverGetProperties(props: *mut CUfileDrvProps_t) -> CUfileError_t;
    
    // File handle management
    pub fn cuFileHandleRegister(
        fh: *mut *mut CUfileHandle_t,
        descr: *const CUfileDescr_t
    ) -> CUfileError_t;
    
    pub fn cuFileHandleDeregister(fh: *mut CUfileHandle_t) -> CUfileError_t;
    
    // Buffer management
    pub fn cuFileBufRegister(
        devPtr_base: *const c_void,
        size: usize,
        flags: c_int
    ) -> CUfileError_t;
    
    pub fn cuFileBufDeregister(devPtr_base: *const c_void) -> CUfileError_t;
    
    // I/O operations
    pub fn cuFileRead(
        fh: *mut CUfileHandle_t,
        devPtr_base: *mut c_void,
        size: usize,
        file_offset: i64,
        devPtr_offset: i64
    ) -> isize;
    
    pub fn cuFileWrite(
        fh: *mut CUfileHandle_t,
        devPtr_base: *const c_void,
        size: usize,
        file_offset: i64,
        devPtr_offset: i64
    ) -> isize;
    
    // Batch I/O operations
    pub fn cuFileBatchIOSetUp(
        batch_handle: *mut *mut CUfileBatchHandle_t,
        max_batch_size: u32
    ) -> CUfileError_t;
    
    pub fn cuFileBatchIOSubmit(
        batch_handle: *mut CUfileBatchHandle_t,
        io_batch_params: *mut CUfileIOParams_t,
        nr: u32,
        flags: c_int
    ) -> CUfileError_t;
    
    pub fn cuFileBatchIOGetStatus(
        batch_handle: *mut CUfileBatchHandle_t,
        io_batch_params: *mut CUfileIOParams_t,
        nr: *mut u32,
        timeout: c_int
    ) -> CUfileError_t;
    
    pub fn cuFileBatchIODestroy(batch_handle: *mut CUfileBatchHandle_t) -> CUfileError_t;
}

/// High-level Rust wrapper for nvidia-fs
pub struct NvidiaFS {
    initialized: bool,
    properties: CUfileDrvProps_t,
}

impl NvidiaFS {
    /// Initialize nvidia-fs driver
    pub fn init() -> Result<Self> {
        unsafe {
            let result = cuFileDriverOpen();
            if result != CUfileError_t::CU_FILE_SUCCESS {
                return Err(anyhow::anyhow!("Failed to initialize nvidia-fs: {:?}", result));
            }
            
            let mut props = std::mem::zeroed();
            let result = cuFileDriverGetProperties(&mut props);
            if result != CUfileError_t::CU_FILE_SUCCESS {
                cuFileDriverClose();
                return Err(anyhow::anyhow!("Failed to get driver properties: {:?}", result));
            }
            
            Ok(Self {
                initialized: true,
                properties: props,
            })
        }
    }
    
    /// Get maximum direct I/O size
    pub fn max_direct_io_size(&self) -> usize {
        self.properties.max_direct_io_size
    }
    
    /// Open file for GPUDirect access
    pub fn open_file(&self, path: &Path, fd: c_int) -> Result<GDSFile> {
        if !self.initialized {
            return Err(anyhow::anyhow!("nvidia-fs not initialized"));
        }
        
        unsafe {
            let mut handle: *mut CUfileHandle_t = std::ptr::null_mut();
            let descr = CUfileDescr_t {
                handle: CUfileHandle_u { fd },
                handle_type: CUfileHandleType::CU_FILE_HANDLE_TYPE_OPAQUE_FD,
            };
            
            let result = cuFileHandleRegister(&mut handle, &descr);
            if result != CUfileError_t::CU_FILE_SUCCESS {
                return Err(anyhow::anyhow!("Failed to register file handle: {:?}", result));
            }
            
            Ok(GDSFile {
                handle,
                fd,
                path: path.to_path_buf(),
            })
        }
    }
    
    /// Register GPU buffer for direct I/O
    pub fn register_buffer(&self, gpu_ptr: *mut c_void, size: usize) -> Result<()> {
        unsafe {
            let result = cuFileBufRegister(gpu_ptr, size, 0);
            if result != CUfileError_t::CU_FILE_SUCCESS {
                return Err(anyhow::anyhow!("Failed to register buffer: {:?}", result));
            }
            Ok(())
        }
    }
    
    /// Deregister GPU buffer
    pub fn deregister_buffer(&self, gpu_ptr: *mut c_void) -> Result<()> {
        unsafe {
            let result = cuFileBufDeregister(gpu_ptr);
            if result != CUfileError_t::CU_FILE_SUCCESS {
                return Err(anyhow::anyhow!("Failed to deregister buffer: {:?}", result));
            }
            Ok(())
        }
    }
}

impl Drop for NvidiaFS {
    fn drop(&mut self) {
        if self.initialized {
            unsafe {
                cuFileDriverClose();
            }
        }
    }
}

/// GPUDirect Storage file handle
pub struct GDSFile {
    handle: *mut CUfileHandle_t,
    fd: c_int,
    path: std::path::PathBuf,
}

impl GDSFile {
    /// Read directly from storage to GPU memory
    pub fn read(&self, gpu_buffer: *mut c_void, size: usize, file_offset: i64) -> Result<usize> {
        unsafe {
            let bytes_read = cuFileRead(
                self.handle,
                gpu_buffer,
                size,
                file_offset,
                0  // device offset
            );
            
            if bytes_read < 0 {
                return Err(anyhow::anyhow!("cuFileRead failed: {}", bytes_read));
            }
            
            Ok(bytes_read as usize)
        }
    }
    
    /// Write directly from GPU memory to storage
    pub fn write(&self, gpu_buffer: *const c_void, size: usize, file_offset: i64) -> Result<usize> {
        unsafe {
            let bytes_written = cuFileWrite(
                self.handle,
                gpu_buffer,
                size,
                file_offset,
                0  // device offset
            );
            
            if bytes_written < 0 {
                return Err(anyhow::anyhow!("cuFileWrite failed: {}", bytes_written));
            }
            
            Ok(bytes_written as usize)
        }
    }
    
    /// Get file descriptor
    pub fn fd(&self) -> c_int {
        self.fd
    }
    
    /// Get file path
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for GDSFile {
    fn drop(&mut self) {
        unsafe {
            cuFileHandleDeregister(self.handle);
        }
    }
}

/// Batch I/O operations
pub struct GDSBatch {
    handle: *mut CUfileBatchHandle_t,
    max_size: u32,
}

impl GDSBatch {
    /// Create new batch I/O handle
    pub fn new(max_batch_size: u32) -> Result<Self> {
        unsafe {
            let mut handle: *mut CUfileBatchHandle_t = std::ptr::null_mut();
            let result = cuFileBatchIOSetUp(&mut handle, max_batch_size);
            
            if result != CUfileError_t::CU_FILE_SUCCESS {
                return Err(anyhow::anyhow!("Failed to setup batch I/O: {:?}", result));
            }
            
            Ok(Self {
                handle,
                max_size: max_batch_size,
            })
        }
    }
    
    /// Submit batch I/O operations
    pub fn submit(&self, operations: &mut [CUfileIOParams_t]) -> Result<()> {
        unsafe {
            let result = cuFileBatchIOSubmit(
                self.handle,
                operations.as_mut_ptr(),
                operations.len() as u32,
                0  // flags
            );
            
            if result != CUfileError_t::CU_FILE_SUCCESS {
                return Err(anyhow::anyhow!("Failed to submit batch I/O: {:?}", result));
            }
            
            Ok(())
        }
    }
    
    /// Get batch I/O status
    pub fn get_status(&self, operations: &mut [CUfileIOParams_t], timeout_ms: c_int) -> Result<u32> {
        unsafe {
            let mut completed = 0u32;
            let result = cuFileBatchIOGetStatus(
                self.handle,
                operations.as_mut_ptr(),
                &mut completed,
                timeout_ms
            );
            
            if result != CUfileError_t::CU_FILE_SUCCESS {
                return Err(anyhow::anyhow!("Failed to get batch I/O status: {:?}", result));
            }
            
            Ok(completed)
        }
    }
}

impl Drop for GDSBatch {
    fn drop(&mut self) {
        unsafe {
            cuFileBatchIODestroy(self.handle);
        }
    }
}

// ============================================================================
// CUDA 13.0 Async I/O Extensions
// ============================================================================

/// Async I/O handle for CUDA 13.0
#[repr(C)]
pub struct CUfileAsyncHandle_t {
    _private: [u8; 0],
}

/// Async I/O parameters
#[repr(C)]
pub struct CUfileAsyncParams_t {
    pub stream: *mut c_void,  // CUDA stream
    pub callback: Option<extern "C" fn(*mut c_void)>,
    pub callback_data: *mut c_void,
    pub flags: u32,
}

/// CUDA 13.0 async I/O operations
#[link(name = "cufile")]
extern "C" {
    pub fn cuFileReadAsync(
        fh: *mut CUfileHandle_t,
        devPtr_base: *mut c_void,
        size: usize,
        file_offset: i64,
        devPtr_offset: i64,
        params: *const CUfileAsyncParams_t
    ) -> isize;
    
    pub fn cuFileWriteAsync(
        fh: *mut CUfileHandle_t,
        devPtr_base: *const c_void,
        size: usize,
        file_offset: i64,
        devPtr_offset: i64,
        params: *const CUfileAsyncParams_t
    ) -> isize;
}

/// Enhanced GDS file handle with CUDA 13.0 async support
pub struct GDSFileAsync {
    pub handle: *mut CUfileHandle_t,
    pub stream: *mut c_void,
}

impl GDSFileAsync {
    /// Create async GDS file handle
    pub fn new(path: &Path, stream: *mut c_void) -> Result<Self> {
        let nvidia_fs = NvidiaFS::new()?;
        let file_handle = nvidia_fs.open_file(path)?;
        
        Ok(Self {
            handle: file_handle.handle,
            stream,
        })
    }
    
    /// Async read with CUDA 13.0
    pub fn read_async(
        &self,
        device_ptr: *mut c_void,
        size: usize,
        file_offset: i64,
        callback: Option<extern "C" fn(*mut c_void)>
    ) -> Result<()> {
        let params = CUfileAsyncParams_t {
            stream: self.stream,
            callback,
            callback_data: std::ptr::null_mut(),
            flags: 0,
        };
        
        unsafe {
            let result = cuFileReadAsync(
                self.handle,
                device_ptr,
                size,
                file_offset,
                0,
                &params
            );
            
            if result < 0 {
                return Err(anyhow::anyhow!("Async read failed: {}", result));
            }
        }
        
        Ok(())
    }
    
    /// Async write with CUDA 13.0
    pub fn write_async(
        &self,
        device_ptr: *const c_void,
        size: usize,
        file_offset: i64,
        callback: Option<extern "C" fn(*mut c_void)>
    ) -> Result<()> {
        let params = CUfileAsyncParams_t {
            stream: self.stream,
            callback,
            callback_data: std::ptr::null_mut(),
            flags: 0,
        };
        
        unsafe {
            let result = cuFileWriteAsync(
                self.handle,
                device_ptr,
                size,
                file_offset,
                0,
                &params
            );
            
            if result < 0 {
                return Err(anyhow::anyhow!("Async write failed: {}", result));
            }
        }
        
        Ok(())
    }
}

// ============================================================================
// RTX 5090 Optimizations
// ============================================================================

/// Configuration optimized for RTX 5090 (32GB, 1.5TB/s)
pub struct RTX5090Config {
    pub buffer_size: usize,      // Optimal: 256MB for RTX 5090
    pub queue_depth: u32,        // Optimal: 64 for Blackwell
    pub alignment: usize,        // 4KB alignment
    pub use_pinned_memory: bool,
    pub enable_p2p: bool,        // Peer-to-peer for multi-GPU
}

impl Default for RTX5090Config {
    fn default() -> Self {
        Self {
            buffer_size: 256 * 1024 * 1024,  // 256MB
            queue_depth: 64,                  // Blackwell optimal
            alignment: 4096,                  // 4KB
            use_pinned_memory: true,
            enable_p2p: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_codes() {
        assert_eq!(CUfileError_t::CU_FILE_SUCCESS as i32, 0);
        assert_eq!(CUfileError_t::CU_FILE_DRIVER_NOT_INITIALIZED as i32, -1);
    }
    
    #[test]
    fn test_handle_types() {
        assert_eq!(CUfileHandleType::CU_FILE_HANDLE_TYPE_OPAQUE_FD as i32, 1);
    }
}