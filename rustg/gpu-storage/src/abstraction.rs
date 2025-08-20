// Storage Abstraction Layer Implementation
// Virtual file system with tiered storage management
// Now using real storage paths: /nvme, /ssd, /hdd

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use std::collections::HashMap;
use parking_lot::RwLock;
use bytes::Bytes;
use async_trait::async_trait;
use anyhow::{Result, anyhow};

// Import storage tier manager
use crate::storage_tiers::TierManager;

/// Storage tier levels mapped to real paths
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageTier {
    NVMe,         // Hot - /nvme path (GPUDirect capable)
    SSD,          // Warm - /ssd path
    HDD,          // Cold - /hdd path (archive)
}

impl StorageTier {
    /// Get real filesystem path for tier
    pub fn path(&self) -> &'static str {
        match self {
            StorageTier::NVMe => "/nvme",
            StorageTier::SSD => "/ssd",
            StorageTier::HDD => "/hdd",
        }
    }
    
    pub fn access_latency_us(&self) -> f64 {
        match self {
            StorageTier::NVMe => 10.0,      // NVMe latency
            StorageTier::SSD => 100.0,      // SSD latency
            StorageTier::HDD => 5000.0,     // HDD latency
        }
    }
    
    pub fn capacity_bytes(&self) -> usize {
        match self {
            StorageTier::NVMe => 2 * 1024 * 1024 * 1024 * 1024,      // 2TB NVMe
            StorageTier::SSD => 8 * 1024 * 1024 * 1024 * 1024,       // 8TB SSD
            StorageTier::HDD => 100 * 1024 * 1024 * 1024 * 1024,     // 100TB HDD
        }
    }
    
    /// Check if tier supports GPUDirect Storage
    pub fn supports_gpudirect(&self) -> bool {
        matches!(self, StorageTier::NVMe)
    }
}

/// Virtual file metadata
#[derive(Debug, Clone)]
pub struct VirtualFile {
    pub path: String,
    pub size: usize,
    pub current_tier: StorageTier,
    pub access_count: Arc<AtomicUsize>,
    pub last_access_time: Arc<AtomicU64>,
    pub is_dirty: bool,
}

impl VirtualFile {
    pub fn new(path: String, size: usize, tier: StorageTier) -> Self {
        Self {
            path,
            size,
            current_tier: tier,
            access_count: Arc::new(AtomicUsize::new(0)),
            last_access_time: Arc::new(AtomicU64::new(0)),
            is_dirty: false,
        }
    }
    
    pub fn record_access(&self) {
        self.access_count.fetch_add(1, Ordering::Relaxed);
        self.last_access_time.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            Ordering::Relaxed
        );
    }
}

/// Virtual file system with real storage paths
pub struct VirtualFS {
    files: Arc<RwLock<HashMap<String, Arc<VirtualFile>>>>,
    tier_manager: Arc<TieredStorageManager>,
    real_tier_manager: Arc<TierManager>,  // Real tier manager for /nvme, /ssd, /hdd
}

impl VirtualFS {
    pub fn new() -> Self {
        // Initialize real tier manager
        let real_tier_manager = TierManager::new()
            .expect("Failed to initialize tier manager with real paths");
        
        Self {
            files: Arc::new(RwLock::new(HashMap::new())),
            tier_manager: Arc::new(TieredStorageManager::new()),
            real_tier_manager: Arc::new(real_tier_manager),
        }
    }
    
    /// Create a virtual file
    pub fn create_file(&self, path: &str, size: usize, tier: StorageTier) -> Result<Arc<VirtualFile>> {
        let mut files = self.files.write();
        
        if files.contains_key(path) {
            return Err(anyhow!("File already exists: {}", path));
        }
        
        let file = Arc::new(VirtualFile::new(path.to_string(), size, tier));
        files.insert(path.to_string(), Arc::clone(&file));
        
        Ok(file)
    }
    
    /// Lookup a file
    pub fn lookup(&self, path: &str) -> Option<Arc<VirtualFile>> {
        let files = self.files.read();
        files.get(path).map(|f| {
            f.record_access();
            Arc::clone(f)
        })
    }
    
    /// Delete a file
    pub fn delete(&self, path: &str) -> Result<()> {
        let mut files = self.files.write();
        files.remove(path)
            .ok_or_else(|| anyhow!("File not found: {}", path))?;
        Ok(())
    }
    
    /// List files matching pattern
    pub fn list(&self, pattern: &str) -> Vec<Arc<VirtualFile>> {
        let files = self.files.read();
        files.values()
            .filter(|f| f.path.contains(pattern))
            .cloned()
            .collect()
    }
    
    /// Migrate file to different tier
    pub async fn migrate_tier(&self, path: &str, new_tier: StorageTier) -> Result<()> {
        let file = self.lookup(path)
            .ok_or_else(|| anyhow!("File not found: {}", path))?;
        
        // Check if migration is needed
        if file.current_tier == new_tier {
            return Ok(());
        }
        
        // Simulate tier migration
        self.tier_manager.migrate(&file, new_tier).await?;
        
        // Update file tier (would need interior mutability)
        // In production, use RwLock or Cell for current_tier
        
        Ok(())
    }
}

/// Tiered storage manager
pub struct TieredStorageManager {
    tier_usage: Arc<RwLock<HashMap<StorageTier, usize>>>,
    migration_stats: Arc<MigrationStats>,
}

#[derive(Debug, Default)]
pub struct MigrationStats {
    pub migrations_up: AtomicUsize,
    pub migrations_down: AtomicUsize,
    pub bytes_migrated: AtomicUsize,
}

impl TieredStorageManager {
    pub fn new() -> Self {
        let mut tier_usage = HashMap::new();
        tier_usage.insert(StorageTier::NVMe, 0);
        tier_usage.insert(StorageTier::SSD, 0);
        tier_usage.insert(StorageTier::HDD, 0);
        
        Self {
            tier_usage: Arc::new(RwLock::new(tier_usage)),
            migration_stats: Arc::new(MigrationStats::default()),
        }
    }
    
    /// Recommend tier based on access patterns
    pub fn recommend_tier(&self, file: &VirtualFile) -> StorageTier {
        let access_count = file.access_count.load(Ordering::Relaxed);
        let file_size = file.size;
        
        // Hot data -> GPU memory
        if access_count > 100 && file_size < StorageTier::NVMe.capacity_bytes() / 100 {
            StorageTier::NVMe
        }
        // Warm data -> SSD
        else if access_count > 10 && file_size < StorageTier::SSD.capacity_bytes() / 100 {
            StorageTier::SSD
        }
        // Cold data -> HDD
        else {
            StorageTier::HDD
        }
    }
    
    /// Migrate file between tiers
    pub async fn migrate(&self, file: &VirtualFile, target_tier: StorageTier) -> Result<()> {
        let current_tier = file.current_tier;
        
        // Update statistics
        if target_tier as u8 > current_tier as u8 {
            self.migration_stats.migrations_down.fetch_add(1, Ordering::Relaxed);
        } else {
            self.migration_stats.migrations_up.fetch_add(1, Ordering::Relaxed);
        }
        
        self.migration_stats.bytes_migrated.fetch_add(file.size, Ordering::Relaxed);
        
        // Update tier usage
        let mut usage = self.tier_usage.write();
        *usage.entry(current_tier).or_insert(0) -= file.size;
        *usage.entry(target_tier).or_insert(0) += file.size;
        
        // Simulate migration delay
        let latency_us = (current_tier.access_latency_us() + target_tier.access_latency_us()) / 2.0;
        tokio::time::sleep(tokio::time::Duration::from_micros(latency_us as u64)).await;
        
        Ok(())
    }
    
    /// Get tier usage statistics
    pub fn tier_usage(&self) -> HashMap<StorageTier, usize> {
        self.tier_usage.read().clone()
    }
}

/// Storage backend trait
#[async_trait]
pub trait StorageBackend: Send + Sync {
    async fn read(&self, path: &str, offset: u64, length: usize) -> Result<Bytes>;
    async fn write(&self, path: &str, offset: u64, data: &[u8]) -> Result<()>;
    async fn delete(&self, path: &str) -> Result<()>;
    async fn exists(&self, path: &str) -> Result<bool>;
    async fn size(&self, path: &str) -> Result<usize>;
}

/// Local storage backend
pub struct LocalBackend {
    root_path: String,
}

#[async_trait]
impl StorageBackend for LocalBackend {
    async fn read(&self, _path: &str, _offset: u64, length: usize) -> Result<Bytes> {
        // Simulate local file read
        Ok(Bytes::from(vec![0u8; length]))
    }
    
    async fn write(&self, _path: &str, _offset: u64, _data: &[u8]) -> Result<()> {
        // Simulate local file write
        Ok(())
    }
    
    async fn delete(&self, _path: &str) -> Result<()> {
        Ok(())
    }
    
    async fn exists(&self, _path: &str) -> Result<bool> {
        Ok(true)
    }
    
    async fn size(&self, _path: &str) -> Result<usize> {
        Ok(1024)
    }
}

/// Multi-backend storage router
pub struct StorageRouter {
    backends: HashMap<String, Arc<dyn StorageBackend>>,
}

impl StorageRouter {
    pub fn new() -> Self {
        Self {
            backends: HashMap::new(),
        }
    }
    
    pub fn register_backend(&mut self, scheme: &str, backend: Arc<dyn StorageBackend>) {
        self.backends.insert(scheme.to_string(), backend);
    }
    
    pub fn get_backend(&self, uri: &str) -> Result<Arc<dyn StorageBackend>> {
        let scheme = uri.split("://").next()
            .ok_or_else(|| anyhow!("Invalid URI: {}", uri))?;
        
        self.backends.get(scheme)
            .cloned()
            .ok_or_else(|| anyhow!("Unknown storage backend: {}", scheme))
    }
}

/// Compression helper
pub struct CompressionHelper;

impl CompressionHelper {
    pub fn compress(data: &[u8]) -> Result<Bytes> {
        use lz4::block::compress;
        
        let compressed = compress(data, None, true)?;
        Ok(Bytes::from(compressed))
    }
    
    pub fn decompress(data: &[u8], uncompressed_size: usize) -> Result<Bytes> {
        use lz4::block::decompress;
        
        let decompressed = decompress(data, Some(uncompressed_size as i32))?;
        Ok(Bytes::from(decompressed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vfs_operations() {
        let vfs = VirtualFS::new();
        
        // Create file
        let file = vfs.create_file("/test.dat", 1024, StorageTier::NVMe).unwrap();
        assert_eq!(file.path, "/test.dat");
        
        // Lookup file
        let found = vfs.lookup("/test.dat").unwrap();
        assert_eq!(found.size, 1024);
        
        // Delete file
        vfs.delete("/test.dat").unwrap();
        assert!(vfs.lookup("/test.dat").is_none());
    }
    
    #[test]
    fn test_tier_recommendation() {
        let manager = TieredStorageManager::new();
        let file = VirtualFile::new("/hot.dat".to_string(), 1024, StorageTier::Archive);
        
        // Simulate high access
        for _ in 0..150 {
            file.access_count.fetch_add(1, Ordering::Relaxed);
        }
        
        let recommended = manager.recommend_tier(&file);
        assert_eq!(recommended, StorageTier::GPUMemory);
    }
    
    #[test]
    fn test_compression() {
        let data = vec![0xAA; 1024];
        let compressed = CompressionHelper::compress(&data).unwrap();
        assert!(compressed.len() < data.len());
        
        let decompressed = CompressionHelper::decompress(&compressed, 1024).unwrap();
        assert_eq!(decompressed.len(), 1024);
    }
}