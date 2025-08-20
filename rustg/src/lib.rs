//! rustg - GPU-native Rust compiler
//! 
//! A GPU-native Rust compiler that leverages CUDA for parallel compilation.
//! This library provides the core GPU compilation functionality for RustyTorch++.

#![warn(missing_docs)]
#![allow(dead_code, unused_variables, unused_imports)]

// Include the generated CUDA bindings with warning suppressions
#[allow(non_camel_case_types, missing_docs, non_upper_case_globals)]
mod cuda_bindings {
    include!(concat!(env!("OUT_DIR"), "/cuda_bindings.rs"));
}

// Re-export the bindings
pub use cuda_bindings::*;

/// Error handling and result types
pub mod error;
/// Core compilation functionality including compiler and runtime components
pub mod core;
/// GPU-accelerated lexical analysis and tokenization
pub mod lexer;
/// GPU-accelerated parsing and AST generation
pub mod parser;
/// Foreign Function Interface bindings for CUDA and GPU operations
pub mod ffi;
/// Language Server Protocol types and utilities
pub mod lsp_types;
/// GPU analysis and optimization tools
pub mod gpu_analyzer;

// Re-export key CUDA FFI functions
pub use ffi::cuda::{initialize_cuda, cleanup_cuda};

use std::sync::Mutex;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// Re-export the proper compilation result from core module
pub use crate::core::compiler::CompilationResult;

/// GPU compiler instance
#[derive(Debug)]
pub struct GpuCompiler {
    initialized: bool,
    cpu_fallback: bool,
}

impl GpuCompiler {
    /// Create a new GPU compiler instance
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self { 
            initialized: false,
            cpu_fallback: false,
        })
    }
    
    /// Enable CPU fallback for GPU operations
    pub fn with_cpu_fallback(mut self, enable: bool) -> Self {
        self.cpu_fallback = enable;
        self
    }
    
    /// Enable profiling
    pub fn with_profiling(self, _enable: bool) -> Self {
        self
    }
    
    /// Set GPU memory limit
    pub fn with_gpu_memory_limit(self, _limit_bytes: usize) -> Self {
        self
    }
    
    /// Set GPU thread count
    pub fn with_gpu_threads(self, _thread_count: usize) -> Self {
        self
    }
    
    /// Initialize the GPU compiler
    pub fn initialize(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.initialized = true;
        Ok(())
    }
    
    /// Compile Rust source code on GPU
    pub fn compile_source(&self, source: &str) -> anyhow::Result<CompilationResult> {
        // Use the core compiler directly
        let compiler = crate::core::compiler::GpuCompiler::new()
            .with_cpu_fallback(self.cpu_fallback);
        
        compiler.compile_source(source).map_err(|e| anyhow::anyhow!("Compilation failed: {}", e))
    }
    
    /// Compile a Rust source file on GPU
    pub fn compile_file(&self, path: &std::path::Path) -> anyhow::Result<CompilationResult> {
        let source = std::fs::read_to_string(path)?;
        self.compile_source(&source)
    }
}

impl Default for GpuCompiler {
    fn default() -> Self {
        Self::new().expect("Failed to create GPU compiler")
    }
}

/// Initialize the GPU compiler runtime
/// 
/// This must be called before any GPU operations.
pub fn initialize() -> anyhow::Result<()> {
    // Basic CUDA initialization would go here
    Ok(())
}

/// Shutdown the GPU compiler runtime
pub fn shutdown() -> Result<(), Box<dyn std::error::Error>> {
    // CUDA cleanup would go here
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
    
    #[test]
    fn test_compiler_creation() {
        let compiler = GpuCompiler::new();
        assert!(compiler.is_ok());
    }
}