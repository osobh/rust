//! Core GPU infrastructure for the rustg compiler

pub mod compiler;
pub mod memory;
pub mod memory_pool;
pub mod kernel;
pub mod telemetry;

#[cfg(feature = "profiling")]
pub mod profiling;