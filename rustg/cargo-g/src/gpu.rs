// GPU detection and management module
// Interfaces with CUDA for real GPU operations - NO MOCKS

use anyhow::{Context, Result};
use rustacuda::device::{Device, DeviceAttribute};
use rustacuda::context::{Context as CudaContext, ContextFlags};
use rustacuda::memory::DeviceBox;
use rustacuda::{init, CudaFlags};
use std::sync::Arc;
use tracing::{debug, info, warn};

#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub id: usize,
    pub name: String,
    pub total_memory: usize,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
    pub multiprocessor_count: i32,
    pub max_threads_per_block: i32,
    pub max_threads_per_multiprocessor: i32,
    pub warp_size: i32,
    pub shared_memory_per_block: usize,
    pub max_grid_dimensions: [i32; 3],
    pub max_block_dimensions: [i32; 3],
    pub supports_managed_memory: bool,
    pub supports_concurrent_kernels: bool,
    pub supports_gpu_direct: bool,
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub device_count: usize,
    pub devices: Vec<GpuDevice>,
    pub multi_gpu_available: bool,
    pub peer_access_matrix: Vec<bool>,
    pub total_memory: usize,
    pub min_compute_capability: (i32, i32),
}

pub struct GpuDetector {
    initialized: bool,
}

impl GpuDetector {
    pub fn new() -> Result<Self> {
        // Initialize CUDA
        init(CudaFlags::empty())
            .context("Failed to initialize CUDA - ensure CUDA drivers are installed")?;
        
        Ok(Self {
            initialized: true,
        })
    }
    
    pub fn detect_gpus(&self) -> Result<GpuInfo> {
        if !self.initialized {
            return Err(anyhow::anyhow!("GPU detector not initialized"));
        }
        
        // Get device count
        let device_count = Device::num_devices()
            .context("Failed to query GPU device count")?;
        
        if device_count == 0 {
            return Ok(GpuInfo {
                device_count: 0,
                devices: vec![],
                multi_gpu_available: false,
                peer_access_matrix: vec![],
                total_memory: 0,
                min_compute_capability: (0, 0),
            });
        }
        
        info!("Detecting {} GPU device(s)...", device_count);
        
        // Enumerate devices
        let mut devices = Vec::new();
        let mut total_memory = 0;
        let mut min_compute = (99, 99);
        
        for id in 0..device_count {
            let device = Device::get_device(id as u32)
                .context(format!("Failed to get device {}", id))?;
            
            let gpu_device = self.query_device_properties(id, &device)?;
            
            // Track totals
            total_memory += gpu_device.total_memory;
            
            // Track minimum compute capability
            if (gpu_device.compute_capability_major, gpu_device.compute_capability_minor) 
                < min_compute {
                min_compute = (
                    gpu_device.compute_capability_major,
                    gpu_device.compute_capability_minor
                );
            }
            
            devices.push(gpu_device);
        }
        
        // Check peer access for multi-GPU
        let peer_access_matrix = if device_count > 1 {
            self.check_peer_access(device_count)?
        } else {
            vec![true] // Single GPU always has self-access
        };
        
        let multi_gpu_available = device_count > 1 && 
            peer_access_matrix.iter().filter(|&&x| x).count() > device_count;
        
        Ok(GpuInfo {
            device_count,
            devices,
            multi_gpu_available,
            peer_access_matrix,
            total_memory,
            min_compute_capability: min_compute,
        })
    }
    
    fn query_device_properties(&self, id: usize, device: &Device) -> Result<GpuDevice> {
        debug!("Querying properties for device {}", id);
        
        // Get device name
        let name = device.name()
            .context("Failed to get device name")?;
        
        // Query memory
        let (free_mem, total_mem) = rustacuda::memory::mem_get_info()
            .context("Failed to get memory info")?;
        
        // Query compute capability
        let major = device.get_attribute(DeviceAttribute::ComputeCapabilityMajor)
            .context("Failed to get compute capability major")?;
        let minor = device.get_attribute(DeviceAttribute::ComputeCapabilityMinor)
            .context("Failed to get compute capability minor")?;
        
        // Query multiprocessor count
        let mp_count = device.get_attribute(DeviceAttribute::MultiprocessorCount)
            .context("Failed to get multiprocessor count")?;
        
        // Query thread limits
        let max_threads_block = device.get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .context("Failed to get max threads per block")?;
        let max_threads_mp = device.get_attribute(DeviceAttribute::MaxThreadsPerMultiprocessor)
            .context("Failed to get max threads per multiprocessor")?;
        
        // Query warp size
        let warp_size = device.get_attribute(DeviceAttribute::WarpSize)
            .context("Failed to get warp size")?;
        
        // Query shared memory
        let shared_mem = device.get_attribute(DeviceAttribute::MaxSharedMemoryPerBlock)
            .context("Failed to get shared memory per block")? as usize;
        
        // Query grid dimensions
        let max_grid_x = device.get_attribute(DeviceAttribute::MaxGridDimX)?;
        let max_grid_y = device.get_attribute(DeviceAttribute::MaxGridDimY)?;
        let max_grid_z = device.get_attribute(DeviceAttribute::MaxGridDimZ)?;
        
        // Query block dimensions
        let max_block_x = device.get_attribute(DeviceAttribute::MaxBlockDimX)?;
        let max_block_y = device.get_attribute(DeviceAttribute::MaxBlockDimY)?;
        let max_block_z = device.get_attribute(DeviceAttribute::MaxBlockDimZ)?;
        
        // Query features
        let managed_memory = device.get_attribute(DeviceAttribute::ManagedMemory)? != 0;
        let concurrent_kernels = device.get_attribute(DeviceAttribute::ConcurrentKernels)? != 0;
        
        // GPUDirect support (simplified check)
        let gpu_direct = major >= 3; // Kepler and newer support GPUDirect
        
        // Validate minimum requirements
        if major < 7 {
            warn!("Device {} has compute capability {}.{} (minimum 7.0 recommended)", 
                  id, major, minor);
        }
        
        if total_mem < 1_073_741_824 { // 1GB
            warn!("Device {} has only {:.2} GB memory (1GB+ recommended)",
                  id, total_mem as f64 / 1_073_741_824.0);
        }
        
        Ok(GpuDevice {
            id,
            name,
            total_memory: total_mem,
            compute_capability_major: major,
            compute_capability_minor: minor,
            multiprocessor_count: mp_count,
            max_threads_per_block: max_threads_block,
            max_threads_per_multiprocessor: max_threads_mp,
            warp_size,
            shared_memory_per_block: shared_mem,
            max_grid_dimensions: [max_grid_x, max_grid_y, max_grid_z],
            max_block_dimensions: [max_block_x, max_block_y, max_block_z],
            supports_managed_memory: managed_memory,
            supports_concurrent_kernels: concurrent_kernels,
            supports_gpu_direct: gpu_direct,
        })
    }
    
    fn check_peer_access(&self, device_count: usize) -> Result<Vec<bool>> {
        let mut matrix = vec![false; device_count * device_count];
        
        for i in 0..device_count {
            for j in 0..device_count {
                if i == j {
                    // Self-access always available
                    matrix[i * device_count + j] = true;
                } else {
                    // Check peer access capability
                    let from_device = Device::get_device(i as u32)?;
                    let to_device = Device::get_device(j as u32)?;
                    
                    let can_access = from_device.can_access_peer(&to_device)
                        .unwrap_or(false);
                    
                    matrix[i * device_count + j] = can_access;
                    
                    if can_access {
                        debug!("GPU {} can access GPU {}", i, j);
                    }
                }
            }
        }
        
        Ok(matrix)
    }
    
    pub fn validate_gpu_requirements(&self, gpu_info: &GpuInfo) -> Result<()> {
        // Check minimum requirements
        if gpu_info.device_count == 0 {
            return Err(anyhow::anyhow!("No GPU devices available"));
        }
        
        // Check compute capability
        let (major, minor) = gpu_info.min_compute_capability;
        if major < 7 {
            return Err(anyhow::anyhow!(
                "GPU compute capability {}.{} is too low (minimum 7.0 required)",
                major, minor
            ));
        }
        
        // Check memory
        if gpu_info.total_memory < 1_073_741_824 {
            return Err(anyhow::anyhow!(
                "Insufficient GPU memory: {:.2} GB (minimum 1GB required)",
                gpu_info.total_memory as f64 / 1_073_741_824.0
            ));
        }
        
        // Check warp size (should always be 32 for NVIDIA)
        for device in &gpu_info.devices {
            if device.warp_size != 32 {
                return Err(anyhow::anyhow!(
                    "Unexpected warp size {} on device {} (expected 32)",
                    device.warp_size, device.id
                ));
            }
        }
        
        info!("✓ All GPU requirements validated");
        Ok(())
    }
    
    pub fn select_best_device(&self, gpu_info: &GpuInfo) -> Option<usize> {
        if gpu_info.devices.is_empty() {
            return None;
        }
        
        // Score devices based on capabilities
        let mut best_device = 0;
        let mut best_score = 0;
        
        for (idx, device) in gpu_info.devices.iter().enumerate() {
            let mut score = 0;
            
            // Score based on compute capability (higher is better)
            score += device.compute_capability_major * 100 + device.compute_capability_minor * 10;
            
            // Score based on multiprocessors (more is better)
            score += device.multiprocessor_count;
            
            // Score based on memory (more is better, in GB)
            score += (device.total_memory / 1_073_741_824) as i32;
            
            // Bonus for features
            if device.supports_managed_memory {
                score += 10;
            }
            if device.supports_concurrent_kernels {
                score += 10;
            }
            if device.supports_gpu_direct {
                score += 20;
            }
            
            debug!("Device {} score: {}", idx, score);
            
            if score > best_score {
                best_score = score;
                best_device = idx;
            }
        }
        
        info!("Selected device {}: {} (score: {})",
              best_device,
              gpu_info.devices[best_device].name,
              best_score);
        
        Some(best_device)
    }
}

// GPU context manager for compilation
pub struct GpuCompilationContext {
    device_id: usize,
    context: CudaContext,
    device: Device,
}

impl GpuCompilationContext {
    pub fn new(device_id: usize) -> Result<Self> {
        let device = Device::get_device(device_id as u32)?;
        
        // Create CUDA context for this device
        let context = CudaContext::create_and_push(
            ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
            device,
        )?;
        
        Ok(Self {
            device_id,
            context,
            device,
        })
    }
    
    pub fn device_id(&self) -> usize {
        self.device_id
    }
    
    pub fn allocate_device_memory(&self, size: usize) -> Result<DeviceBox<u8>> {
        DeviceBox::new(&0u8)
            .context("Failed to allocate GPU memory")
    }
    
    pub fn get_memory_info(&self) -> Result<(usize, usize)> {
        rustacuda::memory::mem_get_info()
            .context("Failed to get memory info")
    }
    
    pub fn synchronize(&self) -> Result<()> {
        rustacuda::context::Context::synchronize()
            .context("Failed to synchronize GPU")
    }
}

// Multi-GPU coordinator
pub struct MultiGpuCoordinator {
    contexts: Vec<Arc<GpuCompilationContext>>,
    current_device: usize,
}

impl MultiGpuCoordinator {
    pub fn new(gpu_info: &GpuInfo, num_gpus: usize) -> Result<Self> {
        let num_gpus = num_gpus.min(gpu_info.device_count);
        
        let mut contexts = Vec::new();
        for i in 0..num_gpus {
            let ctx = GpuCompilationContext::new(i)?;
            contexts.push(Arc::new(ctx));
        }
        
        Ok(Self {
            contexts,
            current_device: 0,
        })
    }
    
    pub fn get_next_context(&mut self) -> Arc<GpuCompilationContext> {
        let ctx = self.contexts[self.current_device].clone();
        self.current_device = (self.current_device + 1) % self.contexts.len();
        ctx
    }
    
    pub fn synchronize_all(&self) -> Result<()> {
        for ctx in &self.contexts {
            ctx.synchronize()?;
        }
        Ok(())
    }
}