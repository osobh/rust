//! rustg - GPU-native Rust compiler
//! 
//! A GPU-native Rust compiler that leverages CUDA for parallel compilation.
//! This library provides the core GPU compilation functionality for RustyTorch++.

#![warn(missing_docs)]
#![allow(dead_code, unused_variables, unused_imports)]

// Include the generated CUDA bindings
include!(concat!(env!("OUT_DIR"), "/cuda_bindings.rs"));

pub mod error;
pub mod core;
pub mod lexer;
pub mod parser;
pub mod ffi;

use std::sync::Mutex;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// Re-export the proper compilation result from core module
pub use crate::core::compiler::CompilationResult;

/// GPU compiler instance
#[derive(Debug)]
pub struct GpuCompiler {
    initialized: bool,
}

impl GpuCompiler {
    /// Create a new GPU compiler instance
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self { 
            initialized: false 
        })
    }
    
    /// Initialize the GPU compiler
    pub fn initialize(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.initialized = true;
        Ok(())
    }
    
    /// Compile Rust source code on GPU
    pub fn compile(&self, source: &str) -> Result<CompilationResult, Box<dyn std::error::Error>> {
        if !self.initialized {
            return Err("Compiler not initialized".into());
        }
        
        // Placeholder implementation - use the core compiler
        let compiler = crate::core::compiler::GpuCompiler::new();
        Ok(compiler.compile_source(source)?)
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
pub fn initialize() -> Result<(), Box<dyn std::error::Error>> {
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