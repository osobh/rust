//! rustg - GPU-native Rust compiler
//! 
//! A revolutionary Rust compiler that performs the entire compilation pipeline on GPU,
//! achieving >10x compilation speedup through massive parallelization.

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod core;
pub mod error;
pub mod ffi;
pub mod lexer;
pub mod parser;

// Re-export main types
pub use crate::core::compiler::GpuCompiler;
pub use crate::error::{CompilerError, Result};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize the GPU compiler runtime
/// 
/// This must be called before any GPU operations.
/// It initializes CUDA, allocates GPU memory pools, and sets up profiling.
///
/// # Errors
/// 
/// Returns an error if:
/// - CUDA is not available
/// - GPU memory allocation fails
/// - Insufficient GPU compute capability
pub fn initialize() -> Result<()> {
    tracing::info!("Initializing rustg GPU compiler v{}", VERSION);
    
    // Initialize CUDA runtime
    ffi::cuda::initialize_cuda()?;
    
    // Allocate GPU memory pools
    core::memory::initialize_memory_pools()?;
    
    // Set up profiling if enabled
    #[cfg(feature = "profiling")]
    core::profiling::initialize_profiling()?;
    
    tracing::info!("rustg initialization complete");
    Ok(())
}

/// Shutdown the GPU compiler runtime
/// 
/// Cleans up GPU resources and flushes any pending operations.
///
/// # Errors
/// 
/// Returns an error if GPU cleanup fails
pub fn shutdown() -> Result<()> {
    tracing::info!("Shutting down rustg GPU compiler");
    
    // Flush any pending GPU operations
    ffi::cuda::synchronize_device()?;
    
    // Free GPU memory pools
    core::memory::cleanup_memory_pools()?;
    
    // Cleanup CUDA runtime
    ffi::cuda::cleanup_cuda()?;
    
    tracing::info!("rustg shutdown complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}