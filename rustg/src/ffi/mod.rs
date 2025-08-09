//! FFI bindings for CUDA interop

pub mod cuda;

// Include generated bindings
#[allow(non_upper_case_globals)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(dead_code)]
#[allow(clippy::all)]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/cuda_bindings.rs"));
}

pub use bindings::*;