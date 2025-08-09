//! Core GPU infrastructure for the rustg compiler

pub mod compiler;
pub mod memory;
pub mod kernel;

#[cfg(feature = "profiling")]
pub mod profiling;