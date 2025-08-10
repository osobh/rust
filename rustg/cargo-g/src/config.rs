// Configuration module for cargo-g
// Handles build configuration and Cargo.toml parsing

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use toml;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildConfig {
    pub package: PackageConfig,
    pub dependencies: HashMap<String, Dependency>,
    #[serde(default)]
    pub gpu: GpuConfig,
    #[serde(default)]
    pub profile: ProfileConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageConfig {
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub authors: Vec<String>,
    #[serde(default)]
    pub edition: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    pub version: Option<String>,
    pub path: Option<String>,
    pub git: Option<String>,
    #[serde(default)]
    pub features: Vec<String>,
    #[serde(default)]
    pub optional: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    #[serde(default = "default_gpu_arch")]
    pub target_arch: String,
    #[serde(default = "default_num_gpus")]
    pub num_gpus: usize,
    #[serde(default = "default_kernel_config")]
    pub kernel_config: KernelConfig,
    #[serde(default)]
    pub optimization: OptimizationConfig,
    #[serde(default)]
    pub memory_limits: MemoryLimits,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelConfig {
    #[serde(default = "default_threads_per_block")]
    pub threads_per_block: usize,
    #[serde(default = "default_blocks_per_grid")]
    pub blocks_per_grid: usize,
    #[serde(default = "default_shared_memory")]
    pub shared_memory_kb: usize,
    #[serde(default = "default_max_registers")]
    pub max_registers: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    #[serde(default = "default_opt_level")]
    pub level: u8,
    #[serde(default)]
    pub use_fast_math: bool,
    #[serde(default)]
    pub inline_threshold: usize,
    #[serde(default)]
    pub unroll_loops: bool,
    #[serde(default)]
    pub vectorize: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLimits {
    #[serde(default = "default_max_gpu_memory")]
    pub max_gpu_memory_mb: usize,
    #[serde(default = "default_cache_size")]
    pub cache_size_mb: usize,
    #[serde(default)]
    pub allow_unified_memory: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileConfig {
    #[serde(default)]
    pub dev: BuildProfile,
    #[serde(default)]
    pub release: BuildProfile,
    #[serde(default)]
    pub bench: BuildProfile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildProfile {
    #[serde(default = "default_opt_level")]
    pub opt_level: u8,
    #[serde(default)]
    pub debug: bool,
    #[serde(default)]
    pub debug_assertions: bool,
    #[serde(default)]
    pub overflow_checks: bool,
    #[serde(default)]
    pub lto: bool,
    #[serde(default = "default_codegen_units")]
    pub codegen_units: usize,
}

// Default functions for serde
fn default_gpu_arch() -> String {
    "sm_70".to_string()
}

fn default_num_gpus() -> usize {
    1
}

fn default_threads_per_block() -> usize {
    256
}

fn default_blocks_per_grid() -> usize {
    256
}

fn default_shared_memory() -> usize {
    48
}

fn default_max_registers() -> usize {
    64
}

fn default_opt_level() -> u8 {
    2
}

fn default_max_gpu_memory() -> usize {
    1024 // 1GB
}

fn default_cache_size() -> usize {
    256 // 256MB
}

fn default_codegen_units() -> usize {
    16
}

fn default_kernel_config() -> KernelConfig {
    KernelConfig {
        threads_per_block: default_threads_per_block(),
        blocks_per_grid: default_blocks_per_grid(),
        shared_memory_kb: default_shared_memory(),
        max_registers: default_max_registers(),
    }
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            target_arch: default_gpu_arch(),
            num_gpus: default_num_gpus(),
            kernel_config: default_kernel_config(),
            optimization: OptimizationConfig::default(),
            memory_limits: MemoryLimits::default(),
        }
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            level: 2,
            use_fast_math: false,
            inline_threshold: 275,
            unroll_loops: true,
            vectorize: true,
        }
    }
}

impl Default for MemoryLimits {
    fn default() -> Self {
        Self {
            max_gpu_memory_mb: default_max_gpu_memory(),
            cache_size_mb: default_cache_size(),
            allow_unified_memory: true,
        }
    }
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            dev: BuildProfile::dev(),
            release: BuildProfile::release(),
            bench: BuildProfile::bench(),
        }
    }
}

impl Default for BuildProfile {
    fn default() -> Self {
        Self::dev()
    }
}

impl BuildProfile {
    fn dev() -> Self {
        Self {
            opt_level: 0,
            debug: true,
            debug_assertions: true,
            overflow_checks: true,
            lto: false,
            codegen_units: 16,
        }
    }
    
    fn release() -> Self {
        Self {
            opt_level: 3,
            debug: false,
            debug_assertions: false,
            overflow_checks: false,
            lto: true,
            codegen_units: 1,
        }
    }
    
    fn bench() -> Self {
        Self {
            opt_level: 3,
            debug: false,
            debug_assertions: false,
            overflow_checks: false,
            lto: true,
            codegen_units: 1,
        }
    }
}

impl BuildConfig {
    pub fn from_manifest(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .context("Failed to read Cargo.toml")?;
        
        let config: BuildConfig = toml::from_str(&content)
            .context("Failed to parse Cargo.toml")?;
        
        Ok(config)
    }
    
    pub fn validate(&self) -> Result<()> {
        // Validate GPU configuration
        if !self.gpu.target_arch.starts_with("sm_") {
            return Err(anyhow::anyhow!(
                "Invalid GPU architecture: {}. Must start with 'sm_'",
                self.gpu.target_arch
            ));
        }
        
        let arch_num: u32 = self.gpu.target_arch
            .trim_start_matches("sm_")
            .parse()
            .context("Invalid GPU architecture number")?;
        
        if arch_num < 70 {
            return Err(anyhow::anyhow!(
                "GPU architecture sm_{} is too old. Minimum sm_70 required",
                arch_num
            ));
        }
        
        // Validate kernel configuration
        if self.gpu.kernel_config.threads_per_block > 1024 {
            return Err(anyhow::anyhow!(
                "threads_per_block {} exceeds maximum 1024",
                self.gpu.kernel_config.threads_per_block
            ));
        }
        
        if self.gpu.kernel_config.shared_memory_kb > 96 {
            return Err(anyhow::anyhow!(
                "shared_memory_kb {} exceeds typical maximum 96KB",
                self.gpu.kernel_config.shared_memory_kb
            ));
        }
        
        Ok(())
    }
}

// Build options from CLI
#[derive(Debug, Clone)]
pub struct BuildOptions {
    pub release: bool,
    pub target: String,
    pub jobs: usize,
    pub features: Vec<String>,
    pub all_features: bool,
    pub no_default_features: bool,
    pub gpu_arch: String,
    pub num_gpus: usize,
    pub optimization_level: u8,
    pub use_fast_math: bool,
    pub enable_debug: bool,
}

impl BuildOptions {
    pub fn merge_with_config(&mut self, config: &BuildConfig) {
        // Use config values if not specified in CLI
        if self.gpu_arch == "sm_70" && config.gpu.target_arch != "sm_70" {
            self.gpu_arch = config.gpu.target_arch.clone();
        }
        
        if self.num_gpus == 1 && config.gpu.num_gpus > 1 {
            self.num_gpus = config.gpu.num_gpus;
        }
        
        // Apply profile settings
        let profile = if self.release {
            &config.profile.release
        } else {
            &config.profile.dev
        };
        
        if self.optimization_level == 0 {
            self.optimization_level = profile.opt_level;
        }
        
        self.enable_debug = profile.debug;
    }
    
    pub fn to_compiler_flags(&self) -> Vec<String> {
        let mut flags = Vec::new();
        
        flags.push(format!("--gpu-arch={}", self.gpu_arch));
        flags.push(format!("--opt-level={}", self.optimization_level));
        
        if self.use_fast_math {
            flags.push("--fast-math".to_string());
        }
        
        if self.enable_debug {
            flags.push("--debug".to_string());
        }
        
        flags.push(format!("--num-gpus={}", self.num_gpus));
        flags.push(format!("--jobs={}", self.jobs));
        
        if !self.features.is_empty() {
            flags.push(format!("--features={}", self.features.join(",")));
        }
        
        if self.all_features {
            flags.push("--all-features".to_string());
        }
        
        if self.no_default_features {
            flags.push("--no-default-features".to_string());
        }
        
        flags
    }
}