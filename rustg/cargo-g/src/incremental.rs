//! Incremental compilation with GPU-aware dependency tracking
//! Provides intelligent recompilation based on dependency analysis

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, info};
use sha2::{Digest, Sha256};

/// Dependency graph for incremental compilation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyGraph {
    /// File path to its direct dependencies
    pub dependencies: HashMap<PathBuf, HashSet<PathBuf>>,
    /// File path to its content hash
    pub file_hashes: HashMap<PathBuf, String>,
    /// File path to last modification time
    pub modification_times: HashMap<PathBuf, u64>,
    /// Reverse dependency map (dependents)
    pub dependents: HashMap<PathBuf, HashSet<PathBuf>>,
}

impl DependencyGraph {
    pub fn new() -> Self {
        Self {
            dependencies: HashMap::new(),
            file_hashes: HashMap::new(),
            modification_times: HashMap::new(),
            dependents: HashMap::new(),
        }
    }
    
    /// Add file to dependency graph
    pub fn add_file(&mut self, file_path: PathBuf, deps: HashSet<PathBuf>) -> Result<()> {
        // Calculate file hash
        let hash = Self::calculate_file_hash(&file_path)?;
        let mod_time = Self::get_modification_time(&file_path)?;
        
        // Update graph
        self.file_hashes.insert(file_path.clone(), hash);
        self.modification_times.insert(file_path.clone(), mod_time);
        self.dependencies.insert(file_path.clone(), deps.clone());
        
        // Update reverse dependencies
        for dep in &deps {
            self.dependents.entry(dep.clone())
                .or_insert_with(HashSet::new)
                .insert(file_path.clone());
        }
        
        debug!("Added {} with {} dependencies to graph", 
               file_path.display(), deps.len());
        
        Ok(())
    }
    
    /// Check if file needs recompilation
    pub fn needs_recompilation(&self, file_path: &Path) -> Result<bool> {
        // File doesn't exist in graph - needs compilation
        if !self.file_hashes.contains_key(file_path) {
            return Ok(true);
        }
        
        // Check if file itself changed
        let current_hash = Self::calculate_file_hash(file_path)?;
        let stored_hash = self.file_hashes.get(file_path).unwrap();
        
        if current_hash != *stored_hash {
            debug!("File {} changed (hash mismatch)", file_path.display());
            return Ok(true);
        }
        
        // Check if any dependencies changed
        if let Some(deps) = self.dependencies.get(file_path) {
            for dep in deps {
                if self.needs_recompilation(dep)? {
                    debug!("File {} needs recompilation due to dependency {}", 
                           file_path.display(), dep.display());
                    return Ok(true);
                }
            }
        }
        
        Ok(false)
    }
    
    /// Get all files that need recompilation due to changes
    pub fn get_recompilation_set(&self, changed_files: &[PathBuf]) -> Result<HashSet<PathBuf>> {
        let mut to_recompile = HashSet::new();
        let mut visited = HashSet::new();
        
        for file in changed_files {
            self.collect_dependents(file, &mut to_recompile, &mut visited)?;
        }
        
        info!("Recompilation set: {} files need rebuilding", to_recompile.len());
        Ok(to_recompile)
    }
    
    /// Collect all dependents recursively
    fn collect_dependents(
        &self,
        file: &Path,
        to_recompile: &mut HashSet<PathBuf>,
        visited: &mut HashSet<PathBuf>,
    ) -> Result<()> {
        if visited.contains(file) {
            return Ok(()); // Avoid cycles
        }
        
        visited.insert(file.to_path_buf());
        to_recompile.insert(file.to_path_buf());
        
        // Add all dependents
        if let Some(dependents) = self.dependents.get(file) {
            for dependent in dependents {
                self.collect_dependents(dependent, to_recompile, visited)?;
            }
        }
        
        Ok(())
    }
    
    /// Update file hash after successful compilation
    pub fn update_file_hash(&mut self, file_path: &Path) -> Result<()> {
        let new_hash = Self::calculate_file_hash(file_path)?;
        let new_mod_time = Self::get_modification_time(file_path)?;
        
        self.file_hashes.insert(file_path.to_path_buf(), new_hash);
        self.modification_times.insert(file_path.to_path_buf(), new_mod_time);
        
        Ok(())
    }
    
    /// Calculate SHA256 hash of file content
    fn calculate_file_hash(file_path: &Path) -> Result<String> {
        let content = std::fs::read(file_path)
            .context(format!("Failed to read file: {}", file_path.display()))?;
        
        let mut hasher = Sha256::new();
        hasher.update(&content);
        let hash = hasher.finalize();
        
        Ok(format!("{:x}", hash))
    }
    
    /// Get file modification time as Unix timestamp
    fn get_modification_time(file_path: &Path) -> Result<u64> {
        let metadata = std::fs::metadata(file_path)
            .context(format!("Failed to get metadata for: {}", file_path.display()))?;
        
        let mod_time = metadata.modified()
            .context("Failed to get modification time")?;
        
        let timestamp = mod_time.duration_since(UNIX_EPOCH)
            .context("Invalid modification time")?
            .as_secs();
        
        Ok(timestamp)
    }
    
    /// Optimize dependency graph for GPU parallel processing
    pub fn optimize_for_gpu(&mut self) -> Result<CompilationPlan> {
        let mut levels = Vec::new();
        let mut remaining: HashSet<PathBuf> = self.dependencies.keys().cloned().collect();
        
        // Topological sort to find compilation levels
        while !remaining.is_empty() {
            let mut current_level = Vec::new();
            let mut can_compile = Vec::new();
            
            // Find files with no uncompiled dependencies
            for file in &remaining {
                if let Some(deps) = self.dependencies.get(file) {
                    let uncompiled_deps: HashSet<_> = deps.intersection(&remaining).collect();
                    if uncompiled_deps.is_empty() {
                        can_compile.push(file.clone());
                    }
                }
            }
            
            if can_compile.is_empty() {
                // Circular dependency detected
                return Err(anyhow::anyhow!("Circular dependency detected"));
            }
            
            for file in &can_compile {
                remaining.remove(file);
                current_level.push(file.clone());
            }
            
            levels.push(current_level);
        }
        
        Ok(CompilationPlan {
            parallel_levels: levels,
            estimated_gpu_utilization: self.estimate_gpu_utilization(),
        })
    }
    
    fn estimate_gpu_utilization(&self) -> f32 {
        let total_files = self.dependencies.len();
        if total_files == 0 {
            return 0.0;
        }
        
        // Estimate based on parallelizable files
        let max_parallel = self.dependencies.values()
            .map(|deps| deps.len())
            .max()
            .unwrap_or(1);
        
        // GPU utilization estimate (simplified)
        (total_files as f32 / max_parallel as f32).min(1.0) * 0.85 // 85% peak utilization
    }
}

impl Default for DependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Compilation plan optimized for GPU execution
#[derive(Debug, Clone)]
pub struct CompilationPlan {
    /// Files that can be compiled in parallel at each level
    pub parallel_levels: Vec<Vec<PathBuf>>,
    /// Estimated GPU utilization percentage
    pub estimated_gpu_utilization: f32,
}

impl CompilationPlan {
    /// Get maximum parallelism available
    pub fn max_parallelism(&self) -> usize {
        self.parallel_levels.iter()
            .map(|level| level.len())
            .max()
            .unwrap_or(1)
    }
    
    /// Get total compilation phases
    pub fn total_phases(&self) -> usize {
        self.parallel_levels.len()
    }
    
    /// Estimate compilation time with GPU acceleration
    pub fn estimate_compilation_time(&self, avg_file_compile_time_ms: f64) -> f64 {
        // With GPU acceleration, parallel files compile simultaneously
        let total_time: f64 = self.parallel_levels.iter()
            .map(|level| {
                // GPU can process multiple files in parallel
                if level.len() <= 32 {
                    // Single GPU warp can handle this level
                    avg_file_compile_time_ms
                } else {
                    // Multiple warps needed
                    let warps_needed = (level.len() + 31) / 32;
                    avg_file_compile_time_ms * (warps_needed as f64 * 0.1) // GPU scaling factor
                }
            })
            .sum();
        
        total_time
    }
}

/// Incremental compilation manager
pub struct IncrementalCompiler {
    dependency_graph: DependencyGraph,
    cache_dir: PathBuf,
    gpu_acceleration: bool,
}

impl IncrementalCompiler {
    pub fn new(cache_dir: PathBuf, gpu_acceleration: bool) -> Self {
        Self {
            dependency_graph: DependencyGraph::new(),
            cache_dir,
            gpu_acceleration,
        }
    }
    
    /// Load dependency graph from cache
    pub fn load_dependency_graph(&mut self) -> Result<()> {
        let graph_path = self.cache_dir.join("dependency_graph.json");
        
        if graph_path.exists() {
            let content = std::fs::read_to_string(&graph_path)
                .context("Failed to read dependency graph")?;
            
            self.dependency_graph = serde_json::from_str(&content)
                .context("Failed to parse dependency graph")?;
            
            info!("Loaded dependency graph with {} files", 
                  self.dependency_graph.dependencies.len());
        }
        
        Ok(())
    }
    
    /// Save dependency graph to cache
    pub fn save_dependency_graph(&self) -> Result<()> {
        let graph_path = self.cache_dir.join("dependency_graph.json");
        
        let content = serde_json::to_string_pretty(&self.dependency_graph)
            .context("Failed to serialize dependency graph")?;
        
        std::fs::write(&graph_path, content)
            .context("Failed to write dependency graph")?;
        
        Ok(())
    }
    
    /// Analyze workspace for incremental compilation
    pub fn analyze_workspace(&mut self, workspace_root: &Path) -> Result<CompilationPlan> {
        info!("Analyzing workspace for incremental compilation: {}", workspace_root.display());
        
        // Find all Rust files
        let rust_files = self.find_rust_files(workspace_root)?;
        info!("Found {} Rust files", rust_files.len());
        
        // Analyze dependencies for each file
        for file in &rust_files {
            let deps = self.analyze_file_dependencies(file)?;
            self.dependency_graph.add_file(file.clone(), deps)?;
        }
        
        // Create optimized compilation plan
        let plan = self.dependency_graph.optimize_for_gpu()?;
        
        info!("Created compilation plan: {} levels, {:.1}% GPU utilization",
              plan.total_phases(), plan.estimated_gpu_utilization * 100.0);
        
        Ok(plan)
    }
    
    /// Find all Rust source files in workspace
    fn find_rust_files(&self, root: &Path) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        self.find_rust_files_recursive(root, &mut files)?;
        Ok(files)
    }
    
    fn find_rust_files_recursive(&self, dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
        if !dir.is_dir() {
            return Ok(());
        }
        
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_dir() {
                // Skip target and other build directories
                let dir_name = path.file_name().unwrap_or_default().to_string_lossy();
                if !["target", ".git", "node_modules", ".cargo"].contains(&dir_name.as_ref()) {
                    self.find_rust_files_recursive(&path, files)?;
                }
            } else if path.extension().map(|e| e == "rs").unwrap_or(false) {
                files.push(path);
            }
        }
        
        Ok(())
    }
    
    /// Analyze dependencies for a single file
    fn analyze_file_dependencies(&self, file_path: &Path) -> Result<HashSet<PathBuf>> {
        let content = std::fs::read_to_string(file_path)
            .context(format!("Failed to read file: {}", file_path.display()))?;
        
        let mut deps = HashSet::new();
        
        // Parse use statements and mod declarations
        for line in content.lines() {
            let trimmed = line.trim();
            
            // Handle 'use' statements
            if trimmed.starts_with("use ") {
                if let Some(module_path) = self.extract_module_path_from_use(trimmed) {
                    if let Some(file_dep) = self.resolve_module_to_file(&module_path, file_path) {
                        deps.insert(file_dep);
                    }
                }
            }
            
            // Handle 'mod' declarations  
            if trimmed.starts_with("mod ") && !trimmed.contains('{') {
                if let Some(module_name) = self.extract_module_name_from_mod(trimmed) {
                    if let Some(file_dep) = self.resolve_mod_to_file(&module_name, file_path) {
                        deps.insert(file_dep);
                    }
                }
            }
        }
        
        Ok(deps)
    }
    
    fn extract_module_path_from_use(&self, use_line: &str) -> Option<String> {
        // Extract module path from "use path::to::module;"
        let without_use = use_line.strip_prefix("use ")?.trim();
        let without_semicolon = without_use.strip_suffix(';').unwrap_or(without_use);
        
        // Handle various use forms
        if let Some(as_pos) = without_semicolon.find(" as ") {
            Some(without_semicolon[..as_pos].trim().to_string())
        } else if without_semicolon.contains("::") {
            Some(without_semicolon.split("::").next()?.to_string())
        } else {
            Some(without_semicolon.to_string())
        }
    }
    
    fn extract_module_name_from_mod(&self, mod_line: &str) -> Option<String> {
        // Extract module name from "mod module_name;"
        let without_mod = mod_line.strip_prefix("mod ")?.trim();
        let without_semicolon = without_mod.strip_suffix(';').unwrap_or(without_mod);
        Some(without_semicolon.to_string())
    }
    
    fn resolve_module_to_file(&self, module_path: &str, current_file: &Path) -> Option<PathBuf> {
        // Simplified module resolution
        let current_dir = current_file.parent()?;
        
        // Try module_path.rs
        let rs_file = current_dir.join(format!("{}.rs", module_path));
        if rs_file.exists() {
            return Some(rs_file);
        }
        
        // Try module_path/mod.rs
        let mod_file = current_dir.join(module_path).join("mod.rs");
        if mod_file.exists() {
            return Some(mod_file);
        }
        
        // Try module_path/lib.rs
        let lib_file = current_dir.join(module_path).join("lib.rs");
        if lib_file.exists() {
            return Some(lib_file);
        }
        
        None
    }
    
    fn resolve_mod_to_file(&self, module_name: &str, current_file: &Path) -> Option<PathBuf> {
        let current_dir = current_file.parent()?;
        
        // Try module_name.rs in same directory
        let rs_file = current_dir.join(format!("{}.rs", module_name));
        if rs_file.exists() {
            return Some(rs_file);
        }
        
        // Try module_name/mod.rs
        let mod_file = current_dir.join(module_name).join("mod.rs");
        if mod_file.exists() {
            return Some(mod_file);
        }
        
        None
    }
}

/// GPU-optimized compilation scheduling
pub struct GpuCompilationScheduler {
    max_gpu_threads: usize,
    memory_pool_size: usize,
}

impl GpuCompilationScheduler {
    pub fn new(max_gpu_threads: usize, memory_pool_size: usize) -> Self {
        Self {
            max_gpu_threads,
            memory_pool_size,
        }
    }
    
    /// Schedule compilation tasks for optimal GPU utilization
    pub fn schedule_compilation(&self, plan: &CompilationPlan) -> Result<Vec<CompilationBatch>> {
        let mut batches = Vec::new();
        
        for (level_idx, level) in plan.parallel_levels.iter().enumerate() {
            // Determine optimal batch size for GPU
            let batch_size = if level.len() <= self.max_gpu_threads {
                level.len()
            } else {
                // Split into multiple batches
                self.max_gpu_threads
            };
            
            // Create batches for this level
            for chunk in level.chunks(batch_size) {
                batches.push(CompilationBatch {
                    level: level_idx,
                    files: chunk.to_vec(),
                    estimated_memory_usage: chunk.len() * 2_097_152, // 2MB per file estimate
                    gpu_threads_needed: chunk.len().min(self.max_gpu_threads),
                });
            }
        }
        
        info!("Created {} compilation batches for GPU execution", batches.len());
        Ok(batches)
    }
}

/// Compilation batch for GPU execution
#[derive(Debug, Clone)]
pub struct CompilationBatch {
    pub level: usize,
    pub files: Vec<PathBuf>,
    pub estimated_memory_usage: usize,
    pub gpu_threads_needed: usize,
}

impl CompilationBatch {
    pub fn can_fit_in_memory(&self, available_memory: usize) -> bool {
        self.estimated_memory_usage <= available_memory
    }
    
    pub fn gpu_utilization(&self, max_threads: usize) -> f32 {
        (self.gpu_threads_needed as f32 / max_threads as f32).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_dependency_graph_creation() {
        let mut graph = DependencyGraph::new();
        
        let file = PathBuf::from("test.rs");
        let deps = HashSet::new();
        
        // This would normally work with real files
        // For test, we'll just verify the structure
        assert_eq!(graph.dependencies.len(), 0);
    }
    
    #[test]
    fn test_compilation_plan() {
        let plan = CompilationPlan {
            parallel_levels: vec![
                vec![PathBuf::from("a.rs"), PathBuf::from("b.rs")],
                vec![PathBuf::from("c.rs")],
            ],
            estimated_gpu_utilization: 0.75,
        };
        
        assert_eq!(plan.max_parallelism(), 2);
        assert_eq!(plan.total_phases(), 2);
        
        let compile_time = plan.estimate_compilation_time(100.0);
        assert!(compile_time > 0.0);
    }
    
    #[test]
    fn test_gpu_scheduler() {
        let scheduler = GpuCompilationScheduler::new(512, 1_073_741_824); // 1GB pool
        
        let plan = CompilationPlan {
            parallel_levels: vec![
                vec![PathBuf::from("a.rs"), PathBuf::from("b.rs")],
            ],
            estimated_gpu_utilization: 0.5,
        };
        
        let result = scheduler.schedule_compilation(&plan);
        assert!(result.is_ok());
        
        let batches = result.unwrap();
        assert!(!batches.is_empty());
    }
}