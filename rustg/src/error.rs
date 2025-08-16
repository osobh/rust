//! Error types for the rustg compiler

use thiserror::Error;

/// Main error type for the rustg compiler
#[derive(Error, Debug)]
pub enum CompilerError {
    /// CUDA initialization or operation failed
    #[error("CUDA error: {0}")]
    Cuda(String),
    
    /// GPU memory allocation failed
    #[error("GPU memory allocation failed: {0}")]
    GpuMemory(String),
    
    /// Insufficient GPU compute capability
    #[error("GPU compute capability {found} is insufficient (requires {required})")]
    InsufficientComputeCapability {
        /// Found compute capability
        found: String,
        /// Required compute capability
        required: String,
    },
    
    /// IO error during file operations
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Source file too large for GPU memory
    #[error("Source file too large: {size_mb}MB exceeds GPU limit of {limit_mb}MB")]
    SourceTooLarge {
        /// File size in MB
        size_mb: usize,
        /// GPU memory limit in MB
        limit_mb: usize,
    },
    
    /// Parsing error
    #[error("Parse error at line {line}, column {column}: {message}")]
    Parse {
        /// Line number
        line: usize,
        /// Column number
        column: usize,
        /// Error message
        message: String,
    },
    
    /// Type checking error
    #[error("Type error: {0}")]
    Type(String),
    
    /// Code generation error
    #[error("Code generation error: {0}")]
    CodeGen(String),
    
    /// Feature not yet implemented
    #[error("Feature not implemented: {0}")]
    NotImplemented(String),
    
    /// Invalid UTF-8 encountered
    #[error("Invalid UTF-8: {0}")]
    InvalidUtf8(String),
    
    /// Generic error with context
    #[error("{0}")]
    Other(String),
}

/// Result type alias for rustg operations
pub type Result<T> = std::result::Result<T, CompilerError>;

impl CompilerError {
    /// Create a CUDA error from an error code
    pub fn from_cuda_error(code: i32) -> Self {
        Self::Cuda(format!("CUDA error code: {}", code))
    }
    
    /// Check if this is a recoverable error
    pub fn is_recoverable(&self) -> bool {
        matches!(self, 
            Self::Parse { .. } | 
            Self::Type(_) | 
            Self::NotImplemented(_)
        )
    }
    
    /// Check if CPU fallback might help
    pub fn can_fallback(&self) -> bool {
        matches!(self,
            Self::Cuda(_) |
            Self::GpuMemory(_) |
            Self::InsufficientComputeCapability { .. } |
            Self::SourceTooLarge { .. } |
            Self::NotImplemented(_)
        )
    }
}