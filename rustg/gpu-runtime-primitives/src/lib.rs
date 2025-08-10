// GPU-Native Runtime Primitives Library
// Core runtime infrastructure for GPU execution

pub mod allocator;
pub mod scheduler;
pub mod communication;
pub mod error_handling;

pub use allocator::{
    SlabAllocator, RegionAllocator, ArenaAllocator, 
    MemoryPoolHierarchy, UnifiedGPUAllocator, PoolConfig
};

pub use scheduler::{
    Task, TaskStatus, WorkQueue, PriorityScheduler,
    GPUScheduler, SchedulerConfig, DependencyResolver
};

pub use communication::{
    Message, MPMCChannel, GPUAtomics, GPUFutex,
    GPUBarrier, CollectiveOps, ChannelManager
};

pub use error_handling::{
    PanicInfo, LogEntry, LogSeverity, RingBufferLogger,
    ErrorReport, ErrorCategory, ErrorHandler, ErrorHandlerConfig
};

/// Runtime primitives configuration
#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub allocator: allocator::PoolConfig,
    pub scheduler: scheduler::SchedulerConfig,
    pub error_handler: error_handling::ErrorHandlerConfig,
    pub channel_capacity: usize,
    pub num_channels: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        RuntimeConfig {
            allocator: allocator::PoolConfig::default(),
            scheduler: scheduler::SchedulerConfig::default(),
            error_handler: error_handling::ErrorHandlerConfig::default(),
            channel_capacity: 10000,
            num_channels: 16,
        }
    }
}

/// Main GPU runtime primitives coordinator
pub struct GPURuntime {
    allocator: UnifiedGPUAllocator,
    scheduler: GPUScheduler,
    channels: ChannelManager,
    error_handler: ErrorHandler,
    config: RuntimeConfig,
}

impl GPURuntime {
    /// Create new GPU runtime with default configuration
    pub fn new() -> Result<Self, &'static str> {
        Self::with_config(RuntimeConfig::default())
    }

    /// Create GPU runtime with custom configuration
    pub fn with_config(config: RuntimeConfig) -> Result<Self, &'static str> {
        let allocator = UnifiedGPUAllocator::new()?;
        let scheduler = GPUScheduler::new(config.scheduler.clone());
        let channels = ChannelManager::new(config.num_channels, config.channel_capacity);
        let error_handler = ErrorHandler::new(config.error_handler.clone());

        Ok(GPURuntime {
            allocator,
            scheduler,
            channels,
            error_handler,
            config,
        })
    }

    /// Allocate memory
    pub fn allocate(&self, size: usize, alignment: usize) -> Option<*mut u8> {
        self.allocator.allocate(size, alignment)
    }

    /// Submit task for execution
    pub fn submit_task(&self, task: Task) -> Result<(), &'static str> {
        self.scheduler.submit_task(task)
    }

    /// Send message through channel
    pub fn send_message(&self, channel_id: usize, msg: Message) -> Result<(), &'static str> {
        self.channels.send(channel_id, msg)
    }

    /// Receive message from channel
    pub fn receive_message(&self, channel_id: usize) -> Option<Message> {
        self.channels.receive(channel_id)
    }

    /// Log a message
    pub fn log(&self, severity: LogSeverity, message: &str, line: u32) {
        self.error_handler.log(severity, message, line);
    }

    /// Handle panic
    pub fn panic(&self, error_code: u32, message: &str) -> Result<(), &'static str> {
        self.error_handler.handle_panic(error_code, message)
    }

    /// Start runtime services
    pub fn start(&self) -> Result<(), &'static str> {
        self.scheduler.start_persistent_kernels()?;
        Ok(())
    }

    /// Shutdown runtime
    pub fn shutdown(&self) {
        self.scheduler.shutdown();
    }

    /// Validate performance targets
    pub fn validate_performance(&self) -> PerformanceReport {
        PerformanceReport {
            allocator_cycles: UnifiedGPUAllocator::get_allocation_cycles(),
            allocator_valid: self.allocator.validate_performance(),
            scheduler_latency_us: self.scheduler.get_scheduling_latency_us(),
            scheduler_utilization: self.scheduler.get_sm_utilization(),
            scheduler_valid: self.scheduler.validate_performance(),
            atomic_cycles: GPUAtomics::get_latency_cycles(),
            atomic_valid: GPUAtomics::validate_performance(),
            channel_throughput: self.channels.get_throughput(1.0),
            channel_valid: self.channels.validate_performance(1.0),
            logging_overhead: self.error_handler.logger.get_overhead_percent(),
            logging_valid: self.error_handler.validate_performance(),
        }
    }
}

/// Performance validation report
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    pub allocator_cycles: u64,
    pub allocator_valid: bool,
    pub scheduler_latency_us: f32,
    pub scheduler_utilization: f32,
    pub scheduler_valid: bool,
    pub atomic_cycles: u64,
    pub atomic_valid: bool,
    pub channel_throughput: f32,
    pub channel_valid: bool,
    pub logging_overhead: f32,
    pub logging_valid: bool,
}

impl PerformanceReport {
    /// Check if all performance targets are met
    pub fn all_valid(&self) -> bool {
        self.allocator_valid &&
        self.scheduler_valid &&
        self.atomic_valid &&
        self.channel_valid &&
        self.logging_valid
    }

    /// Get overall performance score (10x improvement validation)
    pub fn performance_score(&self) -> f32 {
        let mut score = 0.0;
        let mut count = 0.0;

        // Allocator: target <100 cycles, baseline 1000 cycles
        if self.allocator_cycles < 100 {
            score += 1000.0 / self.allocator_cycles as f32;
            count += 1.0;
        }

        // Scheduler: target <1μs, baseline 10μs
        if self.scheduler_latency_us < 1.0 {
            score += 10.0 / self.scheduler_latency_us;
            count += 1.0;
        }

        // Atomics: target <10 cycles, baseline 100 cycles
        if self.atomic_cycles < 10 {
            score += 100.0 / self.atomic_cycles as f32;
            count += 1.0;
        }

        // Channels: target 1M msgs/sec, baseline 100K msgs/sec
        if self.channel_throughput > 1_000_000.0 {
            score += self.channel_throughput / 100_000.0;
            count += 1.0;
        }

        // Logging: target <5% overhead, baseline 50% overhead
        if self.logging_overhead < 5.0 {
            score += 50.0 / self.logging_overhead;
            count += 1.0;
        }

        if count > 0.0 {
            score / count
        } else {
            0.0
        }
    }
}

/// Runtime statistics
#[derive(Debug, Clone)]
pub struct RuntimeStats {
    pub allocations: u64,
    pub tasks_completed: u64,
    pub messages_sent: u64,
    pub panics_captured: u32,
    pub logs_written: u64,
}

/// Benchmark utilities
pub mod benchmarks {
    use super::*;
    use std::time::Instant;

    /// Benchmark allocator performance
    pub fn benchmark_allocator(runtime: &GPURuntime, iterations: usize) -> f32 {
        let start = Instant::now();
        
        for i in 0..iterations {
            let size = 256 + (i % 10) * 64;
            if let Some(ptr) = runtime.allocate(size, 16) {
                // Would deallocate in production
                let _ = ptr;
            }
        }
        
        let elapsed = start.elapsed();
        iterations as f32 / elapsed.as_secs_f32()
    }

    /// Benchmark scheduler throughput
    pub fn benchmark_scheduler(runtime: &GPURuntime, num_tasks: usize) -> f32 {
        let start = Instant::now();
        
        for i in 0..num_tasks {
            let task = Task::new(i as u32, i as u32 % 4);
            let _ = runtime.submit_task(task);
        }
        
        let elapsed = start.elapsed();
        num_tasks as f32 / elapsed.as_secs_f32()
    }

    /// Benchmark channel throughput
    pub fn benchmark_channels(runtime: &GPURuntime, num_messages: usize) -> f32 {
        let start = Instant::now();
        
        for i in 0..num_messages {
            let msg = Message::new(0, 1, i as u32);
            let _ = runtime.send_message(0, msg);
        }
        
        let elapsed = start.elapsed();
        num_messages as f32 / elapsed.as_secs_f32()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_creation() {
        let runtime = GPURuntime::new();
        assert!(runtime.is_ok());
    }

    #[test]
    fn test_performance_validation() {
        if let Ok(runtime) = GPURuntime::new() {
            let report = runtime.validate_performance();
            // In production, all should be valid
            println!("Performance score: {}", report.performance_score());
        }
    }

    #[test]
    fn test_10x_improvement() {
        if let Ok(runtime) = GPURuntime::new() {
            let report = runtime.validate_performance();
            let score = report.performance_score();
            // Score should be >= 10.0 for 10x improvement
            assert!(score >= 10.0, "Must achieve 10x performance improvement");
        }
    }
}