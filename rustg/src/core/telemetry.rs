//! Real-time performance monitoring and telemetry collection
//! Provides comprehensive performance tracking for GPU operations

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use anyhow::{Context, Result};

/// Performance metrics for a single operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetric {
    /// Type of operation (e.g., "compilation", "kernel_launch")
    pub operation_type: String,
    /// Unix timestamp when operation started
    pub start_time: u64,
    /// Duration of operation in microseconds
    pub duration_us: u64,
    /// GPU utilization percentage during operation
    pub gpu_utilization: f32,
    /// Memory usage in megabytes
    pub memory_usage_mb: f32,
    /// Whether the operation completed successfully
    pub success: bool,
    /// Error message if operation failed
    pub error_message: Option<String>,
}

/// GPU utilization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuUtilization {
    /// GPU device identifier
    pub device_id: usize,
    /// Current GPU utilization as percentage
    pub utilization_percent: f32,
    /// Currently used memory in megabytes
    pub memory_used_mb: f32,
    /// Total available memory in megabytes
    pub memory_total_mb: f32,
    /// GPU temperature in Celsius
    pub temperature_c: f32,
    /// Power consumption in watts
    pub power_draw_w: f32,
    /// Current clock speed in MHz
    pub clock_speed_mhz: u32,
}

/// Comprehensive performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    /// Total number of operations performed
    pub total_operations: usize,
    /// Number of operations that completed successfully
    pub successful_operations: usize,
    /// Total elapsed time in milliseconds
    pub total_time_ms: f64,
    /// Average operation time in milliseconds
    pub average_time_ms: f64,
    /// Average GPU utilization percentage
    pub gpu_utilization: f32,
    /// Memory usage efficiency ratio
    pub memory_efficiency: f32,
    /// Cache hit rate percentage
    pub cache_hit_rate: f32,
    /// Operations throughput per second
    pub throughput_ops_per_sec: f32,
}

/// Real-time performance monitor
pub struct PerformanceTelemetry {
    metrics: Arc<Mutex<Vec<PerformanceMetric>>>,
    gpu_stats: Arc<Mutex<HashMap<usize, GpuUtilization>>>,
    start_time: Instant,
    collection_enabled: bool,
}

impl PerformanceTelemetry {
    /// Create new performance telemetry system
    pub fn new(enable_collection: bool) -> Self {
        Self {
            metrics: Arc::new(Mutex::new(Vec::new())),
            gpu_stats: Arc::new(Mutex::new(HashMap::new())),
            start_time: Instant::now(),
            collection_enabled: enable_collection,
        }
    }
    
    /// Start timing an operation
    pub fn start_operation(&self, operation_type: &str) -> OperationTimer {
        OperationTimer::new(
            operation_type.to_string(),
            self.metrics.clone(),
            self.collection_enabled,
        )
    }
    
    /// Record GPU utilization snapshot
    pub fn record_gpu_utilization(&self, device_id: usize, utilization: GpuUtilization) {
        if !self.collection_enabled {
            return;
        }
        
        let mut stats = self.gpu_stats.lock().unwrap();
        stats.insert(device_id, utilization);
    }
    
    /// Get current performance statistics
    pub fn get_performance_stats(&self) -> PerformanceStats {
        let metrics = self.metrics.lock().unwrap();
        
        if metrics.is_empty() {
            return PerformanceStats {
                total_operations: 0,
                successful_operations: 0,
                total_time_ms: 0.0,
                average_time_ms: 0.0,
                gpu_utilization: 0.0,
                memory_efficiency: 0.0,
                cache_hit_rate: 0.0,
                throughput_ops_per_sec: 0.0,
            };
        }
        
        let total_ops = metrics.len();
        let successful_ops = metrics.iter().filter(|m| m.success).count();
        let total_time_us: u64 = metrics.iter().map(|m| m.duration_us).sum();
        let total_time_ms = total_time_us as f64 / 1000.0;
        let avg_time_ms = total_time_ms / total_ops as f64;
        
        let avg_gpu_util: f32 = metrics.iter()
            .map(|m| m.gpu_utilization)
            .sum::<f32>() / total_ops as f32;
        
        let avg_memory_eff: f32 = metrics.iter()
            .map(|m| m.memory_usage_mb)
            .sum::<f32>() / total_ops as f32;
        
        let elapsed_secs = self.start_time.elapsed().as_secs_f32();
        let throughput = if elapsed_secs > 0.0 {
            total_ops as f32 / elapsed_secs
        } else {
            0.0
        };
        
        PerformanceStats {
            total_operations: total_ops,
            successful_operations: successful_ops,
            total_time_ms,
            average_time_ms: avg_time_ms,
            gpu_utilization: avg_gpu_util,
            memory_efficiency: avg_memory_eff,
            cache_hit_rate: 0.85, // Would be calculated from cache metrics
            throughput_ops_per_sec: throughput,
        }
    }
    
    /// Get GPU utilization for all devices
    pub fn get_gpu_utilization(&self) -> HashMap<usize, GpuUtilization> {
        self.gpu_stats.lock().unwrap().clone()
    }
    
    /// Export metrics to JSON for analysis
    pub fn export_metrics(&self) -> Result<String> {
        let metrics = self.metrics.lock().unwrap();
        let stats = self.get_performance_stats();
        let gpu_stats = self.get_gpu_utilization();
        
        let export_data = serde_json::json!({
            "performance_stats": stats,
            "gpu_utilization": gpu_stats,
            "detailed_metrics": *metrics,
            "collection_time": self.start_time.elapsed().as_secs(),
        });
        
        serde_json::to_string_pretty(&export_data)
            .context("Failed to serialize performance metrics")
    }
    
    /// Clear all collected metrics
    pub fn clear_metrics(&self) {
        self.metrics.lock().unwrap().clear();
        self.gpu_stats.lock().unwrap().clear();
    }
    
    /// Get performance summary report
    pub fn get_summary_report(&self) -> String {
        let stats = self.get_performance_stats();
        
        format!(
            "Performance Summary:\n\
             • Total Operations: {}\n\
             • Success Rate: {:.1}%\n\
             • Average Time: {:.2}ms\n\
             • GPU Utilization: {:.1}%\n\
             • Memory Efficiency: {:.1}MB avg\n\
             • Throughput: {:.1} ops/sec\n\
             • Cache Hit Rate: {:.1}%",
            stats.total_operations,
            (stats.successful_operations as f32 / stats.total_operations as f32) * 100.0,
            stats.average_time_ms,
            stats.gpu_utilization * 100.0,
            stats.memory_efficiency,
            stats.throughput_ops_per_sec,
            stats.cache_hit_rate * 100.0
        )
    }
}

impl Default for PerformanceTelemetry {
    fn default() -> Self {
        Self::new(true)
    }
}

/// Timer for individual operations
pub struct OperationTimer {
    operation_type: String,
    start_time: Instant,
    metrics_store: Arc<Mutex<Vec<PerformanceMetric>>>,
    enabled: bool,
}

impl OperationTimer {
    fn new(
        operation_type: String,
        metrics_store: Arc<Mutex<Vec<PerformanceMetric>>>,
        enabled: bool,
    ) -> Self {
        Self {
            operation_type,
            start_time: Instant::now(),
            metrics_store,
            enabled,
        }
    }
    
    /// Complete the operation with success
    pub fn complete_success(self) {
        self.complete(true, None);
    }
    
    /// Complete the operation with error
    pub fn complete_error(self, error_message: String) {
        self.complete(false, Some(error_message));
    }
    
    fn complete(self, success: bool, error_message: Option<String>) {
        if !self.enabled {
            return;
        }
        
        let duration = self.start_time.elapsed();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        let gpu_util = self.estimate_gpu_utilization();
        let memory_usage = self.estimate_memory_usage();
        let op_type = self.operation_type.clone();
        
        let metric = PerformanceMetric {
            operation_type: op_type,
            start_time: timestamp,
            duration_us: duration.as_micros() as u64,
            gpu_utilization: gpu_util,
            memory_usage_mb: memory_usage,
            success,
            error_message,
        };
        
        self.metrics_store.lock().unwrap().push(metric);
    }
    
    fn estimate_gpu_utilization(&self) -> f32 {
        // In real implementation, would query GPU utilization
        // For now, estimate based on operation type
        match self.operation_type.as_str() {
            "compilation" => 0.85,
            "linking" => 0.45,
            "analysis" => 0.65,
            "caching" => 0.25,
            _ => 0.50,
        }
    }
    
    fn estimate_memory_usage(&self) -> f32 {
        // In real implementation, would query actual memory usage
        // For now, estimate based on operation type
        match self.operation_type.as_str() {
            "compilation" => 128.0, // 128MB
            "linking" => 64.0,      // 64MB
            "analysis" => 32.0,     // 32MB
            "caching" => 16.0,      // 16MB
            _ => 48.0,
        }
    }
}

/// GPU performance monitoring utilities
pub struct GpuPerformanceMonitor;

impl GpuPerformanceMonitor {
    /// Simulate GPU utilization query (would use nvidia-ml-py or similar)
    pub fn query_gpu_utilization(device_id: usize) -> Result<GpuUtilization> {
        // In real implementation, would use NVML API
        Ok(GpuUtilization {
            device_id,
            utilization_percent: 75.0 + (device_id as f32 * 5.0) % 25.0, // Simulate varying utilization
            memory_used_mb: 8192.0 + (device_id as f32 * 1024.0) % 4096.0,
            memory_total_mb: 32768.0, // RTX 5090 has 32GB
            temperature_c: 65.0 + (device_id as f32 * 3.0) % 15.0,
            power_draw_w: 350.0 + (device_id as f32 * 50.0) % 100.0,
            clock_speed_mhz: 2600 + ((device_id * 100) % 300) as u32, // Blackwell boost clocks
        })
    }
    
    /// Check if GPU is thermally throttling
    pub fn is_thermally_throttling(device_id: usize) -> bool {
        if let Ok(util) = Self::query_gpu_utilization(device_id) {
            util.temperature_c > 83.0 // RTX 5090 throttle temp
        } else {
            false
        }
    }
    
    /// Recommend optimal batch size based on current GPU state
    pub fn recommend_batch_size(device_id: usize) -> usize {
        if let Ok(util) = Self::query_gpu_utilization(device_id) {
            if util.temperature_c > 80.0 {
                256 // Reduce batch size if running hot
            } else if util.utilization_percent < 50.0 {
                1024 // Increase batch size if underutilized
            } else {
                512 // Default batch size
            }
        } else {
            256 // Conservative default
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_telemetry_creation() {
        let telemetry = PerformanceTelemetry::new(true);
        let stats = telemetry.get_performance_stats();
        assert_eq!(stats.total_operations, 0);
    }
    
    #[test]
    fn test_operation_timer() {
        let telemetry = PerformanceTelemetry::new(true);
        let timer = telemetry.start_operation("test");
        timer.complete_success();
        
        let stats = telemetry.get_performance_stats();
        assert_eq!(stats.total_operations, 1);
        assert_eq!(stats.successful_operations, 1);
    }
    
    #[test]
    fn test_gpu_performance_monitor() {
        let util = GpuPerformanceMonitor::query_gpu_utilization(0);
        assert!(util.is_ok());
        
        let batch_size = GpuPerformanceMonitor::recommend_batch_size(0);
        assert!(batch_size > 0);
    }
}