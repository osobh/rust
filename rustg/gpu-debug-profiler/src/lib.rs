// GPU Debug/Profiling Infrastructure - Main Library
// Part of rustg ProjectB Phase 1

pub mod source_mapper;
pub mod timeline;
pub mod profiler;
pub mod debugger;
pub mod flamegraph;
pub mod cuda_utils;

use anyhow::Result;
use std::path::Path;
use std::time::Duration;

// Re-export main types
pub use source_mapper::{SourceMapper, SourceLocation, IrMapping};
pub use timeline::{Timeline, TimelineEvent, KernelExecution};
pub use profiler::{Profiler, ProfileData, PerformanceMetrics};
pub use debugger::{Debugger, Breakpoint, WarpState};
pub use flamegraph::{FlameGraph, FlameNode};

// Debug/Profiling configuration
#[derive(Debug, Clone)]
pub struct DebugConfig {
    pub enable_source_mapping: bool,
    pub enable_timeline_tracing: bool,
    pub enable_profiling: bool,
    pub enable_warp_debugging: bool,
    pub enable_flamegraph: bool,
    pub max_overhead_percent: f32,
    pub timeline_buffer_size: usize,
    pub profile_sample_rate: u32,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            enable_source_mapping: true,
            enable_timeline_tracing: true,
            enable_profiling: true,
            enable_warp_debugging: false, // Off by default for performance
            enable_flamegraph: true,
            max_overhead_percent: 5.0, // 5% max overhead
            timeline_buffer_size: 10_000_000, // 10M events
            profile_sample_rate: 1000, // Sample every 1000 cycles
        }
    }
}

// Main debug/profiling context
pub struct GpuDebugContext {
    config: DebugConfig,
    source_mapper: Option<SourceMapper>,
    timeline: Option<Timeline>,
    profiler: Option<Profiler>,
    debugger: Option<Debugger>,
    overhead_tracker: OverheadTracker,
}

impl GpuDebugContext {
    pub fn new(config: DebugConfig) -> Result<Self> {
        let mut context = Self {
            config: config.clone(),
            source_mapper: None,
            timeline: None,
            profiler: None,
            debugger: None,
            overhead_tracker: OverheadTracker::new(config.max_overhead_percent),
        };
        
        // Initialize components based on config
        if config.enable_source_mapping {
            context.source_mapper = Some(SourceMapper::new()?);
        }
        
        if config.enable_timeline_tracing {
            context.timeline = Some(Timeline::new(config.timeline_buffer_size)?);
        }
        
        if config.enable_profiling {
            context.profiler = Some(Profiler::new(config.profile_sample_rate)?);
        }
        
        if config.enable_warp_debugging {
            context.debugger = Some(Debugger::new()?);
        }
        
        Ok(context)
    }
    
    // Start profiling session
    pub fn start_session(&mut self, session_name: &str) -> Result<()> {
        self.overhead_tracker.start();
        
        if let Some(timeline) = &mut self.timeline {
            timeline.start_recording()?;
        }
        
        if let Some(profiler) = &mut self.profiler {
            profiler.start_profiling(session_name)?;
        }
        
        Ok(())
    }
    
    // End profiling session
    pub fn end_session(&mut self) -> Result<SessionReport> {
        let overhead = self.overhead_tracker.stop();
        
        let timeline_data = if let Some(timeline) = &mut self.timeline {
            Some(timeline.stop_recording()?)
        } else {
            None
        };
        
        let profile_data = if let Some(profiler) = &mut self.profiler {
            Some(profiler.stop_profiling()?)
        } else {
            None
        };
        
        Ok(SessionReport {
            overhead_percent: overhead,
            timeline_data,
            profile_data,
        })
    }
    
    // Map source location to GPU IR
    pub fn map_source_to_ir(&self, file: &Path, line: u32) -> Result<Vec<IrMapping>> {
        if let Some(mapper) = &self.source_mapper {
            mapper.map_source_to_ir(file, line)
        } else {
            Ok(Vec::new())
        }
    }
    
    // Map GPU IR to source
    pub fn map_ir_to_source(&self, ir_location: &str) -> Result<Option<SourceLocation>> {
        if let Some(mapper) = &self.source_mapper {
            mapper.map_ir_to_source(ir_location)
        } else {
            Ok(None)
        }
    }
    
    // Set breakpoint
    pub fn set_breakpoint(&mut self, location: BreakpointLocation) -> Result<u32> {
        if let Some(debugger) = &mut self.debugger {
            debugger.set_breakpoint(location)
        } else {
            anyhow::bail!("Warp debugging not enabled")
        }
    }
    
    // Get current warp state
    pub fn get_warp_state(&self, warp_id: u32) -> Result<WarpState> {
        if let Some(debugger) = &self.debugger {
            debugger.get_warp_state(warp_id)
        } else {
            anyhow::bail!("Warp debugging not enabled")
        }
    }
    
    // Generate flamegraph
    pub fn generate_flamegraph(&self, profile_data: &ProfileData) -> Result<FlameGraph> {
        FlameGraph::from_profile_data(profile_data)
    }
    
    // Get performance metrics
    pub fn get_metrics(&self) -> PerformanceMetrics {
        PerformanceMetrics {
            overhead_percent: self.overhead_tracker.current_overhead(),
            events_per_second: self.timeline.as_ref()
                .map(|t| t.events_per_second())
                .unwrap_or(0.0),
            samples_collected: self.profiler.as_ref()
                .map(|p| p.samples_collected())
                .unwrap_or(0),
        }
    }
}

// Breakpoint location specification
#[derive(Debug, Clone)]
pub enum BreakpointLocation {
    SourceLine { file: String, line: u32 },
    KernelFunction { name: String },
    IrLocation { location: String },
    WarpDivergence { threshold: f32 },
    MemoryAddress { address: u64, size: usize },
}

// Session report
#[derive(Debug, Clone)]
pub struct SessionReport {
    pub overhead_percent: f32,
    pub timeline_data: Option<Vec<TimelineEvent>>,
    pub profile_data: Option<ProfileData>,
}

// Performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub overhead_percent: f32,
    pub events_per_second: f32,
    pub samples_collected: usize,
}

// Overhead tracking
struct OverheadTracker {
    max_overhead_percent: f32,
    baseline_time: Option<Duration>,
    debug_time: Option<Duration>,
    start_time: Option<std::time::Instant>,
}

impl OverheadTracker {
    fn new(max_overhead_percent: f32) -> Self {
        Self {
            max_overhead_percent,
            baseline_time: None,
            debug_time: None,
            start_time: None,
        }
    }
    
    fn start(&mut self) {
        self.start_time = Some(std::time::Instant::now());
    }
    
    fn stop(&mut self) -> f32 {
        if let Some(start) = self.start_time {
            let elapsed = start.elapsed();
            self.debug_time = Some(elapsed);
            self.calculate_overhead()
        } else {
            0.0
        }
    }
    
    fn current_overhead(&self) -> f32 {
        self.calculate_overhead()
    }
    
    fn calculate_overhead(&self) -> f32 {
        if let (Some(baseline), Some(debug)) = (self.baseline_time, self.debug_time) {
            let overhead = (debug.as_secs_f32() - baseline.as_secs_f32()) 
                / baseline.as_secs_f32() * 100.0;
            overhead.min(self.max_overhead_percent)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_context_creation() {
        let config = DebugConfig::default();
        let context = GpuDebugContext::new(config);
        assert!(context.is_ok());
    }
    
    #[test]
    fn test_overhead_tracking() {
        let mut tracker = OverheadTracker::new(5.0);
        tracker.baseline_time = Some(Duration::from_secs(1));
        tracker.debug_time = Some(Duration::from_millis(1050));
        assert!(tracker.calculate_overhead() <= 5.0);
    }
}