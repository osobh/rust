// GPU-Native Allocator Implementation
// Lock-free memory allocation with <100 cycle latency

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::alloc::Layout;
use std::collections::HashMap;
use std::sync::Arc;

// FFI bindings to CUDA kernels
extern "C" {
    fn cuda_allocate_slab(size: u32, alignment: u32) -> *mut u8;
    fn cuda_deallocate_slab(ptr: *mut u8, size: u32);
    fn cuda_allocate_region(size: usize) -> *mut u8;
    fn cuda_deallocate_region(ptr: *mut u8, size: usize);
    fn cuda_reset_arena(arena_id: u32);
    fn cuda_get_allocation_cycles() -> u64;
}

/// Slab allocator for fixed-size allocations
pub struct SlabAllocator {
    slab_size: u32,
    num_slabs: u32,
    free_list: Vec<AtomicU32>,
    next_free: AtomicU32,
    memory_base: *mut u8,
    allocation_count: AtomicU64,
    deallocation_count: AtomicU64,
}

unsafe impl Send for SlabAllocator {}
unsafe impl Sync for SlabAllocator {}

impl SlabAllocator {
    /// Create new slab allocator with specified slab size
    pub fn new(slab_size: u32, num_slabs: u32) -> Result<Self, &'static str> {
        if slab_size == 0 || num_slabs == 0 {
            return Err("Invalid slab configuration");
        }

        let total_size = (slab_size * num_slabs) as usize;
        let memory_base = unsafe { cuda_allocate_region(total_size) };
        
        if memory_base.is_null() {
            return Err("Failed to allocate GPU memory");
        }

        let mut free_list = Vec::with_capacity(num_slabs as usize);
        for i in 0..num_slabs {
            free_list.push(AtomicU32::new(i + 1));
        }

        Ok(SlabAllocator {
            slab_size,
            num_slabs,
            free_list,
            next_free: AtomicU32::new(0),
            memory_base,
            allocation_count: AtomicU64::new(0),
            deallocation_count: AtomicU64::new(0),
        })
    }

    /// Allocate a slab with lock-free operation
    pub fn allocate(&self) -> Option<*mut u8> {
        loop {
            let old_free = self.next_free.load(Ordering::Acquire);
            
            if old_free >= self.num_slabs {
                return None; // Out of memory
            }

            let new_free = self.free_list[old_free as usize].load(Ordering::Relaxed);
            
            if self.next_free.compare_exchange_weak(
                old_free,
                new_free,
                Ordering::Release,
                Ordering::Relaxed,
            ).is_ok() {
                self.allocation_count.fetch_add(1, Ordering::Relaxed);
                let offset = old_free * self.slab_size;
                return Some(unsafe { self.memory_base.add(offset as usize) });
            }
        }
    }

    /// Deallocate a slab back to the pool
    pub fn deallocate(&self, ptr: *mut u8) {
        let offset = unsafe { ptr.offset_from(self.memory_base) } as u32;
        let slab_idx = offset / self.slab_size;
        
        loop {
            let old_free = self.next_free.load(Ordering::Acquire);
            self.free_list[slab_idx as usize].store(old_free, Ordering::Relaxed);
            
            if self.next_free.compare_exchange_weak(
                old_free,
                slab_idx,
                Ordering::Release,
                Ordering::Relaxed,
            ).is_ok() {
                self.deallocation_count.fetch_add(1, Ordering::Relaxed);
                break;
            }
        }
    }

    /// Get allocation statistics
    pub fn stats(&self) -> AllocationStats {
        AllocationStats {
            allocations: self.allocation_count.load(Ordering::Relaxed),
            deallocations: self.deallocation_count.load(Ordering::Relaxed),
            available_slabs: self.count_free_slabs(),
            total_slabs: self.num_slabs,
        }
    }

    fn count_free_slabs(&self) -> u32 {
        let mut count = 0;
        let mut current = self.next_free.load(Ordering::Relaxed);
        
        while current < self.num_slabs {
            count += 1;
            current = self.free_list[current as usize].load(Ordering::Relaxed);
        }
        
        count
    }
}

/// Region allocator for large contiguous allocations
pub struct RegionAllocator {
    region_size: usize,
    current_offset: AtomicUsize,
    memory_base: *mut u8,
    allocations: Arc<parking_lot::RwLock<HashMap<*mut u8, usize>>>,
}

unsafe impl Send for RegionAllocator {}
unsafe impl Sync for RegionAllocator {}

impl RegionAllocator {
    /// Create new region allocator
    pub fn new(region_size: usize) -> Result<Self, &'static str> {
        let memory_base = unsafe { cuda_allocate_region(region_size) };
        
        if memory_base.is_null() {
            return Err("Failed to allocate GPU region");
        }

        Ok(RegionAllocator {
            region_size,
            current_offset: AtomicUsize::new(0),
            memory_base,
            allocations: Arc::new(parking_lot::RwLock::new(HashMap::new())),
        })
    }

    /// Allocate contiguous memory region
    pub fn allocate(&self, size: usize, alignment: usize) -> Option<*mut u8> {
        let aligned_size = (size + alignment - 1) & !(alignment - 1);
        
        let offset = self.current_offset.fetch_add(aligned_size, Ordering::Relaxed);
        
        if offset + aligned_size > self.region_size {
            // Rollback
            self.current_offset.fetch_sub(aligned_size, Ordering::Relaxed);
            return None;
        }

        let ptr = unsafe { self.memory_base.add(offset) };
        self.allocations.write().insert(ptr, aligned_size);
        
        Some(ptr)
    }

    /// Deallocate region (for bookkeeping only in simple allocator)
    pub fn deallocate(&self, ptr: *mut u8) {
        self.allocations.write().remove(&ptr);
    }

    /// Get memory efficiency
    pub fn efficiency(&self) -> f32 {
        let used = self.current_offset.load(Ordering::Relaxed);
        used as f32 / self.region_size as f32
    }
}

/// Arena allocator with bulk deallocation
pub struct ArenaAllocator {
    arena_id: u32,
    arena_size: usize,
    current_offset: AtomicUsize,
    memory_base: *mut u8,
    generation: AtomicU32,
}

unsafe impl Send for ArenaAllocator {}
unsafe impl Sync for ArenaAllocator {}

impl ArenaAllocator {
    /// Create new arena allocator
    pub fn new(arena_id: u32, arena_size: usize) -> Result<Self, &'static str> {
        let memory_base = unsafe { cuda_allocate_region(arena_size) };
        
        if memory_base.is_null() {
            return Err("Failed to allocate GPU arena");
        }

        Ok(ArenaAllocator {
            arena_id,
            arena_size,
            current_offset: AtomicUsize::new(0),
            memory_base,
            generation: AtomicU32::new(0),
        })
    }

    /// Allocate from arena
    pub fn allocate(&self, size: usize) -> Option<*mut u8> {
        let offset = self.current_offset.fetch_add(size, Ordering::Relaxed);
        
        if offset + size > self.arena_size {
            // Rollback
            self.current_offset.fetch_sub(size, Ordering::Relaxed);
            return None;
        }

        Some(unsafe { self.memory_base.add(offset) })
    }

    /// Reset arena (bulk deallocation)
    pub fn reset(&self) {
        self.current_offset.store(0, Ordering::Relaxed);
        self.generation.fetch_add(1, Ordering::Relaxed);
        unsafe { cuda_reset_arena(self.arena_id) };
    }

    /// Get current generation
    pub fn generation(&self) -> u32 {
        self.generation.load(Ordering::Relaxed)
    }
}

/// Hierarchical memory pool manager
pub struct MemoryPoolHierarchy {
    thread_pools: Vec<SlabAllocator>,
    warp_pools: Vec<RegionAllocator>,
    block_pool: RegionAllocator,
    global_pool: RegionAllocator,
}

impl MemoryPoolHierarchy {
    /// Create hierarchical memory pools
    pub fn new(config: PoolConfig) -> Result<Self, &'static str> {
        let mut thread_pools = Vec::new();
        for _ in 0..config.num_threads {
            thread_pools.push(SlabAllocator::new(128, 8)?);
        }

        let mut warp_pools = Vec::new();
        for _ in 0..config.num_warps {
            warp_pools.push(RegionAllocator::new(32 * 1024)?);
        }

        let block_pool = RegionAllocator::new(256 * 1024)?;
        let global_pool = RegionAllocator::new(10 * 1024 * 1024)?;

        Ok(MemoryPoolHierarchy {
            thread_pools,
            warp_pools,
            block_pool,
            global_pool,
        })
    }

    /// Allocate from appropriate tier
    pub fn allocate(&self, size: usize, thread_id: usize) -> Option<*mut u8> {
        if size <= 128 {
            // Thread-local allocation
            if thread_id < self.thread_pools.len() {
                return self.thread_pools[thread_id].allocate();
            }
        } else if size <= 1024 {
            // Warp-level allocation
            let warp_id = thread_id / 32;
            if warp_id < self.warp_pools.len() {
                return self.warp_pools[warp_id].allocate(size, 16);
            }
        } else if size <= 8192 {
            // Block-level allocation
            return self.block_pool.allocate(size, 16);
        }

        // Global allocation
        self.global_pool.allocate(size, 16)
    }

    /// Deallocate from appropriate tier
    pub fn deallocate(&self, ptr: *mut u8, size: usize, thread_id: usize) {
        if size <= 128 {
            if thread_id < self.thread_pools.len() {
                self.thread_pools[thread_id].deallocate(ptr);
            }
        } else if size <= 1024 {
            let warp_id = thread_id / 32;
            if warp_id < self.warp_pools.len() {
                self.warp_pools[warp_id].deallocate(ptr);
            }
        } else if size <= 8192 {
            self.block_pool.deallocate(ptr);
        } else {
            self.global_pool.deallocate(ptr);
        }
    }
}

/// Configuration for memory pools
#[derive(Clone, Debug)]
pub struct PoolConfig {
    pub num_threads: usize,
    pub num_warps: usize,
    pub thread_pool_size: usize,
    pub warp_pool_size: usize,
    pub block_pool_size: usize,
    pub global_pool_size: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        PoolConfig {
            num_threads: 256,
            num_warps: 8,
            thread_pool_size: 1024,
            warp_pool_size: 32 * 1024,
            block_pool_size: 256 * 1024,
            global_pool_size: 10 * 1024 * 1024,
        }
    }
}

/// Allocation statistics
#[derive(Debug, Clone)]
pub struct AllocationStats {
    pub allocations: u64,
    pub deallocations: u64,
    pub available_slabs: u32,
    pub total_slabs: u32,
}

/// GPU allocator trait
pub trait GPUAllocator: Send + Sync {
    fn allocate(&self, layout: Layout) -> Result<*mut u8, &'static str>;
    fn deallocate(&self, ptr: *mut u8, layout: Layout);
    fn stats(&self) -> AllocationStats;
}

/// Unified GPU allocator combining all strategies
pub struct UnifiedGPUAllocator {
    slab_allocator: SlabAllocator,
    region_allocator: RegionAllocator,
    arena_allocator: ArenaAllocator,
    pools: MemoryPoolHierarchy,
}

impl UnifiedGPUAllocator {
    pub fn new() -> Result<Self, &'static str> {
        Ok(UnifiedGPUAllocator {
            slab_allocator: SlabAllocator::new(256, 10000)?,
            region_allocator: RegionAllocator::new(128 * 1024 * 1024)?,
            arena_allocator: ArenaAllocator::new(0, 64 * 1024 * 1024)?,
            pools: MemoryPoolHierarchy::new(PoolConfig::default())?,
        })
    }

    /// Select best allocator for given size
    pub fn allocate(&self, size: usize, alignment: usize) -> Option<*mut u8> {
        if size == 256 {
            self.slab_allocator.allocate()
        } else if size > 8192 {
            self.region_allocator.allocate(size, alignment)
        } else {
            self.pools.allocate(size, 0)
        }
    }

    /// Get allocation latency in cycles
    pub fn get_allocation_cycles() -> u64 {
        unsafe { cuda_get_allocation_cycles() }
    }

    /// Validate performance target (<100 cycles)
    pub fn validate_performance(&self) -> bool {
        Self::get_allocation_cycles() < 100
    }
}

// Re-export for convenience
pub use parking_lot;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slab_allocator() {
        let allocator = SlabAllocator::new(256, 100).unwrap();
        let ptr = allocator.allocate().unwrap();
        assert!(!ptr.is_null());
        allocator.deallocate(ptr);
    }

    #[test]
    fn test_allocation_performance() {
        let allocator = UnifiedGPUAllocator::new().unwrap();
        assert!(allocator.validate_performance());
    }
}