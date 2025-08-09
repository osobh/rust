//! GPU memory management for the rustg compiler

use crate::error::{CompilerError, Result};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

/// GPU memory pool for efficient allocation
pub struct GpuMemoryPool {
    /// Total pool size in bytes
    total_size: usize,
    /// Available memory in bytes
    available: usize,
    /// Base device pointer (from CUDA)
    base_ptr: *mut u8,
    /// Allocation map
    allocations: HashMap<usize, Allocation>,
    /// Next allocation ID
    next_id: usize,
}

/// Individual memory allocation
#[derive(Debug, Clone)]
struct Allocation {
    id: usize,
    offset: usize,
    size: usize,
    in_use: bool,
}

/// Global memory pool instance
static MEMORY_POOL: Mutex<Option<Arc<Mutex<GpuMemoryPool>>>> = Mutex::new(None);

/// Initialize GPU memory pools
pub fn initialize_memory_pools() -> Result<()> {
    let pool_size = get_gpu_memory_size()?;
    let pool = GpuMemoryPool::new(pool_size)?;
    
    let mut global_pool = MEMORY_POOL.lock();
    *global_pool = Some(Arc::new(Mutex::new(pool)));
    
    tracing::info!("Initialized GPU memory pool with {}MB", pool_size / (1024 * 1024));
    Ok(())
}

/// Clean up GPU memory pools
pub fn cleanup_memory_pools() -> Result<()> {
    let mut global_pool = MEMORY_POOL.lock();
    if let Some(pool) = global_pool.take() {
        let mut pool = pool.lock();
        pool.cleanup()?;
    }
    Ok(())
}

/// Get the GPU memory pool
pub fn get_memory_pool() -> Result<Arc<Mutex<GpuMemoryPool>>> {
    let global_pool = MEMORY_POOL.lock();
    global_pool.clone().ok_or_else(|| {
        CompilerError::GpuMemory("Memory pool not initialized".to_string())
    })
}

impl GpuMemoryPool {
    /// Create a new GPU memory pool
    pub fn new(size: usize) -> Result<Self> {
        let base_ptr = unsafe { allocate_gpu_memory(size)? };
        
        Ok(Self {
            total_size: size,
            available: size,
            base_ptr,
            allocations: HashMap::new(),
            next_id: 1,
        })
    }
    
    /// Allocate memory from the pool
    pub fn allocate(&mut self, size: usize, alignment: usize) -> Result<GpuMemoryHandle> {
        // Align size to boundary
        let aligned_size = align_up(size, alignment);
        
        if aligned_size > self.available {
            return Err(CompilerError::GpuMemory(format!(
                "Insufficient GPU memory: requested {}MB, available {}MB",
                aligned_size / (1024 * 1024),
                self.available / (1024 * 1024)
            )));
        }
        
        // Find a free block (simple first-fit for now)
        let offset = self.find_free_block(aligned_size)?;
        
        let id = self.next_id;
        self.next_id += 1;
        
        let allocation = Allocation {
            id,
            offset,
            size: aligned_size,
            in_use: true,
        };
        
        self.allocations.insert(id, allocation.clone());
        self.available -= aligned_size;
        
        let ptr = unsafe { self.base_ptr.add(offset) };
        
        Ok(GpuMemoryHandle {
            id,
            ptr: ptr as *mut std::ffi::c_void,
            size: aligned_size,
        })
    }
    
    /// Free a memory allocation
    pub fn free(&mut self, handle: GpuMemoryHandle) -> Result<()> {
        if let Some(allocation) = self.allocations.get_mut(&handle.id) {
            if !allocation.in_use {
                return Err(CompilerError::GpuMemory(
                    "Attempting to free already freed memory".to_string()
                ));
            }
            allocation.in_use = false;
            self.available += allocation.size;
            Ok(())
        } else {
            Err(CompilerError::GpuMemory(
                "Invalid memory handle".to_string()
            ))
        }
    }
    
    /// Find a free block of the requested size
    fn find_free_block(&self, size: usize) -> Result<usize> {
        // Simple implementation: find first gap
        // TODO: Implement better allocation strategy (buddy system, etc.)
        
        let mut allocations: Vec<_> = self.allocations
            .values()
            .filter(|a| a.in_use)
            .collect();
        allocations.sort_by_key(|a| a.offset);
        
        let mut current_offset = 0;
        
        for allocation in allocations {
            let gap = allocation.offset - current_offset;
            if gap >= size {
                return Ok(current_offset);
            }
            current_offset = allocation.offset + allocation.size;
        }
        
        // Check if there's space at the end
        if self.total_size - current_offset >= size {
            Ok(current_offset)
        } else {
            Err(CompilerError::GpuMemory(
                "Memory fragmentation: unable to find contiguous block".to_string()
            ))
        }
    }
    
    /// Clean up the memory pool
    fn cleanup(&mut self) -> Result<()> {
        unsafe {
            free_gpu_memory(self.base_ptr)?;
        }
        self.base_ptr = std::ptr::null_mut();
        self.allocations.clear();
        Ok(())
    }
}

/// Handle to GPU memory allocation
#[derive(Debug)]
pub struct GpuMemoryHandle {
    id: usize,
    ptr: *mut std::ffi::c_void,
    size: usize,
}

impl GpuMemoryHandle {
    /// Get the device pointer
    pub fn device_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }
    
    /// Get the allocation size
    pub fn size(&self) -> usize {
        self.size
    }
}

/// Align a value up to the specified alignment
fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) / alignment * alignment
}

/// Get available GPU memory size
fn get_gpu_memory_size() -> Result<usize> {
    // This will call into CUDA to get device properties
    unsafe {
        crate::ffi::cuda::get_device_memory_size()
    }
}

/// Allocate GPU memory (calls CUDA)
unsafe fn allocate_gpu_memory(size: usize) -> Result<*mut u8> {
    crate::ffi::cuda::cuda_malloc(size)
}

/// Free GPU memory (calls CUDA)
unsafe fn free_gpu_memory(ptr: *mut u8) -> Result<()> {
    crate::ffi::cuda::cuda_free(ptr)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 8), 0);
        assert_eq!(align_up(1, 8), 8);
        assert_eq!(align_up(8, 8), 8);
        assert_eq!(align_up(9, 8), 16);
        assert_eq!(align_up(100, 32), 128);
    }
}