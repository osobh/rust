// Artifact caching module - Content-addressable storage for GPU compilation
// Real caching with SHA256 hashing - NO MOCKS

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

use crate::build::CompilationArtifact;
use crate::config::BuildOptions;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub key: String,
    pub artifacts: Vec<CachedArtifact>,
    pub metadata: CacheMetadata,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedArtifact {
    pub source_hash: String,
    pub object_data: Vec<u8>,
    pub ptx_data: Option<Vec<u8>>,
    pub spirv_data: Option<Vec<u8>>,
    pub compile_flags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetadata {
    pub gpu_arch: String,
    pub optimization_level: u8,
    pub rustg_version: String,
    pub total_size: usize,
    pub compile_time_ms: f64,
}

#[derive(Debug, Default)]
pub struct CacheStatistics {
    pub total_size_mb: f64,
    pub entry_count: usize,
    pub hit_rate: f64,
    pub oldest_entry_days: usize,
    pub avg_entry_size_kb: f64,
    pub recent_entries: Vec<CacheEntryInfo>,
}

#[derive(Debug)]
pub struct CacheEntryInfo {
    pub key: String,
    pub size_mb: f64,
}

pub struct ArtifactCache {
    cache_dir: PathBuf,
    index: HashMap<String, CacheEntry>,
    statistics: CacheStats,
}

#[derive(Debug, Default)]
struct CacheStats {
    hits: usize,
    misses: usize,
    total_size: usize,
}

impl ArtifactCache {
    pub fn new(cache_dir: PathBuf) -> Result<Self> {
        // Create cache directory if it doesn't exist
        fs::create_dir_all(&cache_dir)
            .context("Failed to create cache directory")?;
        
        let index_path = cache_dir.join("index.json");
        
        // Load existing index or create new
        let index = if index_path.exists() {
            let content = fs::read_to_string(&index_path)
                .context("Failed to read cache index")?;
            serde_json::from_str(&content)
                .context("Failed to parse cache index")?
        } else {
            HashMap::new()
        };
        
        // Calculate initial statistics
        let mut statistics = CacheStats::default();
        for entry in index.values() {
            statistics.total_size += entry.metadata.total_size;
        }
        
        Ok(Self {
            cache_dir,
            index,
            statistics,
        })
    }
    
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }
    
    pub fn compute_cache_key(
        &self,
        manifest_path: &Path,
        config: &BuildOptions,
    ) -> Result<String> {
        let mut hasher = Sha256::new();
        
        // Hash manifest content
        let manifest_content = fs::read(manifest_path)
            .context("Failed to read manifest for hashing")?;
        hasher.update(&manifest_content);
        
        // Hash configuration
        hasher.update(config.gpu_arch.as_bytes());
        hasher.update(&[config.optimization_level]);
        hasher.update(&[config.use_fast_math as u8]);
        hasher.update(&[config.enable_debug as u8]);
        
        // Hash features
        for feature in &config.features {
            hasher.update(feature.as_bytes());
        }
        
        let hash = hasher.finalize();
        Ok(format!("{:x}", hash))
    }
    
    pub fn get(&mut self, key: &str) -> Result<Option<CachedArtifact>> {
        if let Some(entry) = self.index.get(key) {
            // Check if cache entry is still valid
            if self.is_entry_valid(entry) {
                self.statistics.hits += 1;
                
                // Load artifact from disk
                let artifact_path = self.cache_dir.join(format!("{}.bin", key));
                if artifact_path.exists() {
                    let data = fs::read(&artifact_path)
                        .context("Failed to read cached artifact")?;
                    
                    let artifact: CachedArtifact = bincode::deserialize(&data)
                        .context("Failed to deserialize cached artifact")?;
                    
                    info!("Cache hit for key: {}", key);
                    return Ok(Some(artifact));
                }
            } else {
                // Remove invalid entry
                self.index.remove(key);
                self.save_index()?;
            }
        }
        
        self.statistics.misses += 1;
        debug!("Cache miss for key: {}", key);
        Ok(None)
    }
    
    pub fn store(&mut self, key: &str, artifacts: &[CompilationArtifact]) -> Result<()> {
        info!("Storing {} artifacts in cache with key: {}", artifacts.len(), key);
        
        let mut cached_artifacts = Vec::new();
        let mut total_size = 0;
        let mut total_compile_time = 0.0;
        
        for artifact in artifacts {
            // Read artifact files
            let object_data = fs::read(&artifact.object_path)
                .context("Failed to read object file")?;
            
            let ptx_data = if let Some(ref ptx_path) = artifact.ptx_path {
                Some(fs::read(ptx_path).context("Failed to read PTX file")?)
            } else {
                None
            };
            
            let spirv_data = if let Some(ref spirv_path) = artifact.spirv_path {
                Some(fs::read(spirv_path).context("Failed to read SPIR-V file")?)
            } else {
                None
            };
            
            // Compute source hash
            let source_content = fs::read(&artifact.source_path)
                .context("Failed to read source file")?;
            let source_hash = format!("{:x}", Sha256::digest(&source_content));
            
            total_size += object_data.len();
            if let Some(ref ptx) = ptx_data {
                total_size += ptx.len();
            }
            if let Some(ref spirv) = spirv_data {
                total_size += spirv.len();
            }
            
            total_compile_time += artifact.metadata.compile_time_ms;
            
            cached_artifacts.push(CachedArtifact {
                source_hash,
                object_data,
                ptx_data,
                spirv_data,
                compile_flags: vec![], // TODO: Store actual flags
            });
        }
        
        // Create cache entry
        let entry = CacheEntry {
            key: key.to_string(),
            artifacts: cached_artifacts.clone(),
            metadata: CacheMetadata {
                gpu_arch: "sm_70".to_string(), // TODO: Get from config
                optimization_level: 2,
                rustg_version: env!("CARGO_PKG_VERSION").to_string(),
                total_size,
                compile_time_ms: total_compile_time,
            },
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        
        // Serialize and store
        let artifact_path = self.cache_dir.join(format!("{}.bin", key));
        let serialized = bincode::serialize(&cached_artifacts)
            .context("Failed to serialize artifacts")?;
        
        fs::write(&artifact_path, serialized)
            .context("Failed to write cached artifact")?;
        
        // Update index
        self.index.insert(key.to_string(), entry);
        self.statistics.total_size += total_size;
        self.save_index()?;
        
        // Check cache size and evict if necessary
        self.check_cache_size()?;
        
        Ok(())
    }
    
    pub fn restore_artifact(
        &self,
        cached: &CachedArtifact,
        config: &BuildOptions,
    ) -> Result<()> {
        let target_dir = PathBuf::from("target")
            .join(if config.release { "release" } else { "debug" })
            .join("gpu-cache");
        
        fs::create_dir_all(&target_dir)?;
        
        // Restore object file
        let obj_path = target_dir.join("cached.o");
        fs::write(&obj_path, &cached.object_data)?;
        
        // Restore PTX if present
        if let Some(ref ptx) = cached.ptx_data {
            let ptx_path = target_dir.join("cached.ptx");
            fs::write(&ptx_path, ptx)?;
        }
        
        // Restore SPIR-V if present
        if let Some(ref spirv) = cached.spirv_data {
            let spirv_path = target_dir.join("cached.spv");
            fs::write(&spirv_path, spirv)?;
        }
        
        info!("Restored cached artifacts to {}", target_dir.display());
        Ok(())
    }
    
    fn is_entry_valid(&self, entry: &CacheEntry) -> bool {
        // Check age (invalidate after 7 days)
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let age_days = (now - entry.timestamp) / 86400;
        if age_days > 7 {
            debug!("Cache entry {} is too old ({} days)", entry.key, age_days);
            return false;
        }
        
        // Check rustg version
        if entry.metadata.rustg_version != env!("CARGO_PKG_VERSION") {
            debug!("Cache entry {} has mismatched version", entry.key);
            return false;
        }
        
        true
    }
    
    fn save_index(&self) -> Result<()> {
        let index_path = self.cache_dir.join("index.json");
        let json = serde_json::to_string_pretty(&self.index)?;
        fs::write(&index_path, json)?;
        Ok(())
    }
    
    fn check_cache_size(&mut self) -> Result<()> {
        const MAX_CACHE_SIZE: usize = 1024 * 1024 * 1024; // 1GB
        
        if self.statistics.total_size > MAX_CACHE_SIZE {
            warn!("Cache size {} exceeds limit, evicting old entries", 
                  self.statistics.total_size);
            
            // Sort entries by timestamp (oldest first)
            let mut entries: Vec<_> = self.index.iter().collect();
            entries.sort_by_key(|(_, e)| e.timestamp);
            
            // Evict oldest entries until under limit
            let target_size = MAX_CACHE_SIZE * 3 / 4; // 75% of max
            
            for (key, _) in entries {
                if self.statistics.total_size <= target_size {
                    break;
                }
                
                self.evict_entry(key)?;
            }
        }
        
        Ok(())
    }
    
    fn evict_entry(&mut self, key: &str) -> Result<()> {
        if let Some(entry) = self.index.remove(key) {
            // Delete artifact file
            let artifact_path = self.cache_dir.join(format!("{}.bin", key));
            if artifact_path.exists() {
                fs::remove_file(&artifact_path)?;
            }
            
            self.statistics.total_size -= entry.metadata.total_size;
            debug!("Evicted cache entry: {}", key);
        }
        
        Ok(())
    }
    
    pub fn clear_all(&mut self) -> Result<()> {
        // Remove all cached files
        for entry in fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            if entry.path().extension().and_then(|s| s.to_str()) == Some("bin") {
                fs::remove_file(entry.path())?;
            }
        }
        
        // Clear index
        self.index.clear();
        self.statistics = CacheStats::default();
        self.save_index()?;
        
        info!("Cleared all cache entries");
        Ok(())
    }
    
    pub fn clear_target(&mut self, target: &str) -> Result<()> {
        let keys_to_remove: Vec<_> = self.index
            .iter()
            .filter(|(_, e)| e.metadata.gpu_arch == target)
            .map(|(k, _)| k.clone())
            .collect();
        
        for key in keys_to_remove {
            self.evict_entry(&key)?;
        }
        
        self.save_index()?;
        info!("Cleared cache entries for target: {}", target);
        Ok(())
    }
    
    pub fn clear_project(&mut self, manifest_path: &Path) -> Result<()> {
        // Clear cache entries specific to a project
        let project_dir = manifest_path.parent()
            .ok_or_else(|| anyhow!("Invalid manifest path"))?;
        
        // Get canonical path for accurate matching
        let canonical_project = project_dir.canonicalize()
            .unwrap_or_else(|_| project_dir.to_path_buf());
        
        // Extract project name from Cargo.toml if available
        let project_name = if manifest_path.exists() {
            fs::read_to_string(manifest_path)
                .ok()
                .and_then(|content| {
                    content.lines()
                        .find(|line| line.starts_with("name"))
                        .and_then(|line| {
                            line.split('=')
                                .nth(1)
                                .map(|s| s.trim().trim_matches('"').to_string())
                        })
                })
        } else {
            None
        };
        
        // Remove entries that match:
        // 1. Project path in metadata
        // 2. Project name in cache key
        // 3. Modified after manifest change
        let manifest_modified = fs::metadata(manifest_path)
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        
        let keys_to_remove: Vec<_> = self.index
            .iter()
            .filter(|(key, entry)| {
                // Check if entry is related to this project
                let path_match = entry.metadata.source_files.iter()
                    .any(|f| f.starts_with(&canonical_project));
                
                let name_match = project_name.as_ref()
                    .map(|name| key.contains(name))
                    .unwrap_or(false);
                
                let time_match = entry.timestamp > manifest_modified;
                
                path_match || name_match || time_match
            })
            .map(|(k, _)| k.clone())
            .collect();
        
        let count = keys_to_remove.len();
        for key in keys_to_remove {
            self.evict_entry(&key)?;
        }
        
        self.save_index()?;
        info!("Cleared {} cache entries for project: {:?}", count, project_name.unwrap_or_else(|| "unknown".to_string()));
        Ok(())
    }
    
    pub fn get_statistics(&self) -> Result<CacheStatistics> {
        let mut stats = CacheStatistics::default();
        
        stats.entry_count = self.index.len();
        stats.total_size_mb = self.statistics.total_size as f64 / (1024.0 * 1024.0);
        
        if self.statistics.hits + self.statistics.misses > 0 {
            stats.hit_rate = self.statistics.hits as f64 / 
                            (self.statistics.hits + self.statistics.misses) as f64;
        }
        
        // Find oldest entry
        if let Some(oldest) = self.index.values().min_by_key(|e| e.timestamp) {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)?
                .as_secs();
            stats.oldest_entry_days = ((now - oldest.timestamp) / 86400) as usize;
        }
        
        // Calculate average size
        if stats.entry_count > 0 {
            stats.avg_entry_size_kb = (self.statistics.total_size / stats.entry_count) as f64 / 1024.0;
        }
        
        // Get recent entries
        let mut entries: Vec<_> = self.index.values().collect();
        entries.sort_by_key(|e| std::cmp::Reverse(e.timestamp));
        
        stats.recent_entries = entries
            .iter()
            .take(10)
            .map(|e| CacheEntryInfo {
                key: e.key.clone(),
                size_mb: e.metadata.total_size as f64 / (1024.0 * 1024.0),
            })
            .collect();
        
        Ok(stats)
    }
    
    pub fn load_baseline(&self, name: &str) -> Result<HashMap<String, f64>> {
        let baseline_path = self.cache_dir.join(format!("baseline_{}.json", name));
        
        if !baseline_path.exists() {
            return Ok(HashMap::new());
        }
        
        let content = fs::read_to_string(&baseline_path)?;
        let baseline: HashMap<String, f64> = serde_json::from_str(&content)?;
        
        Ok(baseline)
    }
    
    pub fn save_baseline(&self, name: &str, results: &crate::build::BenchResults) -> Result<()> {
        let baseline_path = self.cache_dir.join(format!("baseline_{}.json", name));
        
        let json = serde_json::to_string_pretty(&results.benchmarks)?;
        fs::write(&baseline_path, json)?;
        
        info!("Saved benchmark baseline: {}", name);
        Ok(())
    }
}