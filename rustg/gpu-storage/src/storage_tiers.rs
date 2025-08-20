// Storage Tier Management with Real Paths
// Uses actual storage paths: /nvme, /ssd, /hdd

use std::path::{Path, PathBuf};
use anyhow::{Result, Context};
use tokio::fs as async_fs;
use serde::{Deserialize, Serialize};

/// Storage tier configuration with real paths
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageTiers {
    /// NVMe path for GPUDirect Storage (fastest tier)
    pub nvme_path: PathBuf,
    /// SSD path for warm data (mid tier)
    pub ssd_path: PathBuf,
    /// HDD path for archival data (slow tier)
    pub hdd_path: PathBuf,
}

impl Default for StorageTiers {
    fn default() -> Self {
        Self {
            nvme_path: PathBuf::from("/nvme"),
            ssd_path: PathBuf::from("/ssd"),
            hdd_path: PathBuf::from("/hdd"),
        }
    }
}

/// Storage tier type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageTier {
    /// NVMe - GPUDirect capable, >10GB/s
    NVMe,
    /// SSD - Fast tier, 2-5GB/s
    SSD,
    /// HDD - Archive tier, <500MB/s
    HDD,
}

impl StorageTier {
    /// Get expected throughput for tier
    pub fn expected_throughput_gbps(&self) -> f64 {
        match self {
            StorageTier::NVMe => 12.0,  // 12GB/s for GPUDirect
            StorageTier::SSD => 3.5,    // 3.5GB/s for SSD
            StorageTier::HDD => 0.2,    // 200MB/s for HDD
        }
    }
    
    /// Check if tier supports GPUDirect
    pub fn supports_gpudirect(&self) -> bool {
        matches!(self, StorageTier::NVMe)
    }
}

/// Access pattern for tier migration decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    /// Frequently accessed, keep on NVMe
    Hot,
    /// Moderately accessed, keep on SSD
    Warm,
    /// Rarely accessed, move to HDD
    Cold,
}

/// Storage tier manager
pub struct TierManager {
    config: StorageTiers,
    access_counts: std::sync::Arc<parking_lot::RwLock<std::collections::HashMap<PathBuf, usize>>>,
}

impl TierManager {
    /// Create new tier manager with real paths
    pub fn new() -> Result<Self> {
        let config = StorageTiers::default();
        
        // Verify paths exist
        Self::verify_paths(&config)?;
        
        Ok(Self {
            config,
            access_counts: std::sync::Arc::new(parking_lot::RwLock::new(std::collections::HashMap::new())),
        })
    }
    
    /// Create with custom paths
    pub fn with_paths(nvme: impl AsRef<Path>, ssd: impl AsRef<Path>, hdd: impl AsRef<Path>) -> Result<Self> {
        let config = StorageTiers {
            nvme_path: nvme.as_ref().to_path_buf(),
            ssd_path: ssd.as_ref().to_path_buf(),
            hdd_path: hdd.as_ref().to_path_buf(),
        };
        
        Self::verify_paths(&config)?;
        
        Ok(Self {
            config,
            access_counts: std::sync::Arc::new(parking_lot::RwLock::new(std::collections::HashMap::new())),
        })
    }
    
    /// Verify storage paths exist and are accessible
    fn verify_paths(config: &StorageTiers) -> Result<()> {
        // Check NVMe path
        if !config.nvme_path.exists() {
            return Err(anyhow::anyhow!("NVMe path {} does not exist", config.nvme_path.display()));
        }
        if !config.nvme_path.is_dir() {
            return Err(anyhow::anyhow!("NVMe path {} is not a directory", config.nvme_path.display()));
        }
        
        // Check SSD path
        if !config.ssd_path.exists() {
            return Err(anyhow::anyhow!("SSD path {} does not exist", config.ssd_path.display()));
        }
        if !config.ssd_path.is_dir() {
            return Err(anyhow::anyhow!("SSD path {} is not a directory", config.ssd_path.display()));
        }
        
        // Check HDD path
        if !config.hdd_path.exists() {
            return Err(anyhow::anyhow!("HDD path {} does not exist", config.hdd_path.display()));
        }
        if !config.hdd_path.is_dir() {
            return Err(anyhow::anyhow!("HDD path {} is not a directory", config.hdd_path.display()));
        }
        
        Ok(())
    }
    
    /// Get full path for file in tier
    pub fn get_tier_path(&self, tier: StorageTier, filename: &str) -> PathBuf {
        match tier {
            StorageTier::NVMe => self.config.nvme_path.join(filename),
            StorageTier::SSD => self.config.ssd_path.join(filename),
            StorageTier::HDD => self.config.hdd_path.join(filename),
        }
    }
    
    /// Determine which tier a file should be in based on access pattern
    pub fn get_target_tier(&self, filename: &str) -> StorageTier {
        let access_count = {
            let counts = self.access_counts.read();
            counts.get(&PathBuf::from(filename)).copied().unwrap_or(0)
        };
        
        match access_count {
            100.. => StorageTier::NVMe,  // Hot: 100+ accesses
            10..=99 => StorageTier::SSD,  // Warm: 10-99 accesses
            _ => StorageTier::HDD,        // Cold: <10 accesses
        }
    }
    
    /// Record file access for tier migration decisions
    pub fn record_access(&self, filename: &str) {
        let mut counts = self.access_counts.write();
        *counts.entry(PathBuf::from(filename)).or_insert(0) += 1;
    }
    
    /// Find which tier a file currently resides in
    pub async fn find_file_tier(&self, filename: &str) -> Option<StorageTier> {
        // Check NVMe first (fastest)
        if self.get_tier_path(StorageTier::NVMe, filename).exists() {
            return Some(StorageTier::NVMe);
        }
        
        // Check SSD
        if self.get_tier_path(StorageTier::SSD, filename).exists() {
            return Some(StorageTier::SSD);
        }
        
        // Check HDD
        if self.get_tier_path(StorageTier::HDD, filename).exists() {
            return Some(StorageTier::HDD);
        }
        
        None
    }
    
    /// Migrate file between tiers
    pub async fn migrate_file(&self, filename: &str, from: StorageTier, to: StorageTier) -> Result<()> {
        if from == to {
            return Ok(()); // No migration needed
        }
        
        let source = self.get_tier_path(from, filename);
        let dest = self.get_tier_path(to, filename);
        
        // Ensure source exists
        if !source.exists() {
            return Err(anyhow::anyhow!("Source file {} does not exist", source.display()));
        }
        
        // Copy file to new tier
        async_fs::copy(&source, &dest)
            .await
            .with_context(|| format!("Failed to copy {} to {}", source.display(), dest.display()))?;
        
        // Verify copy succeeded
        let source_metadata = async_fs::metadata(&source).await?;
        let dest_metadata = async_fs::metadata(&dest).await?;
        
        if source_metadata.len() != dest_metadata.len() {
            // Copy failed, remove partial file
            let _ = async_fs::remove_file(&dest).await;
            return Err(anyhow::anyhow!("File copy verification failed"));
        }
        
        // Remove from old tier
        async_fs::remove_file(&source)
            .await
            .with_context(|| format!("Failed to remove {} after migration", source.display()))?;
        
        println!("Migrated {} from {:?} to {:?}", filename, from, to);
        
        Ok(())
    }
    
    /// Auto-tier file based on access pattern
    pub async fn auto_tier(&self, filename: &str) -> Result<StorageTier> {
        self.record_access(filename);
        
        let current_tier = self.find_file_tier(filename).await;
        let target_tier = self.get_target_tier(filename);
        
        if let Some(current) = current_tier {
            if current != target_tier {
                self.migrate_file(filename, current, target_tier).await?;
            }
            Ok(target_tier)
        } else {
            // File doesn't exist, return target tier for creation
            Ok(target_tier)
        }
    }
    
    /// Get storage statistics
    pub async fn get_tier_stats(&self) -> Result<TierStats> {
        let nvme_stats = self.get_path_stats(&self.config.nvme_path).await?;
        let ssd_stats = self.get_path_stats(&self.config.ssd_path).await?;
        let hdd_stats = self.get_path_stats(&self.config.hdd_path).await?;
        
        Ok(TierStats {
            nvme: nvme_stats,
            ssd: ssd_stats,
            hdd: hdd_stats,
        })
    }
    
    async fn get_path_stats(&self, path: &Path) -> Result<PathStats> {
        let mut total_size = 0u64;
        let mut file_count = 0u32;
        
        let mut entries = async_fs::read_dir(path).await?;
        while let Some(entry) = entries.next_entry().await? {
            if entry.file_type().await?.is_file() {
                file_count += 1;
                total_size += entry.metadata().await?.len();
            }
        }
        
        Ok(PathStats {
            total_size_gb: total_size as f64 / (1024.0 * 1024.0 * 1024.0),
            file_count,
            path: path.to_path_buf(),
        })
    }
}

/// Statistics for a storage tier
#[derive(Debug)]
pub struct PathStats {
    pub total_size_gb: f64,
    pub file_count: u32,
    pub path: PathBuf,
}

/// Combined tier statistics
#[derive(Debug)]
pub struct TierStats {
    pub nvme: PathStats,
    pub ssd: PathStats,
    pub hdd: PathStats,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_tier_paths() {
        let tiers = StorageTiers::default();
        assert_eq!(tiers.nvme_path, PathBuf::from("/nvme"));
        assert_eq!(tiers.ssd_path, PathBuf::from("/ssd"));
        assert_eq!(tiers.hdd_path, PathBuf::from("/hdd"));
    }
    
    #[test]
    fn test_tier_throughput() {
        assert_eq!(StorageTier::NVMe.expected_throughput_gbps(), 12.0);
        assert_eq!(StorageTier::SSD.expected_throughput_gbps(), 3.5);
        assert_eq!(StorageTier::HDD.expected_throughput_gbps(), 0.2);
    }
    
    #[test]
    fn test_gpudirect_support() {
        assert!(StorageTier::NVMe.supports_gpudirect());
        assert!(!StorageTier::SSD.supports_gpudirect());
        assert!(!StorageTier::HDD.supports_gpudirect());
    }
}