// Debugger Module - Warp-level debugging and breakpoints
// Implements debugging as validated by CUDA tests

use anyhow::{Result, Context};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use dashmap::DashMap;
use serde::{Serialize, Deserialize};
use crate::source_mapper::SourceLocation;

// Breakpoint structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Breakpoint {
    pub id: u32,
    pub location: BreakpointLocation,
    pub condition: Option<String>,
    pub hit_count: u32,
    pub enabled: bool,
    pub temporary: bool,
    pub thread_mask: Option<ThreadMask>,
}

// Breakpoint location types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BreakpointLocation {
    SourceLine { file: String, line: u32 },
    Address(u64),
    Function(String),
    Kernel(String),
    Divergence,
}

// Thread mask for selective breakpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadMask {
    pub warp_mask: u32,
    pub block_mask: Option<u32>,
    pub sm_mask: Option<u32>,
}

// Warp state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarpState {
    pub warp_id: u32,
    pub sm_id: u32,
    pub block_id: u32,
    pub pc: u64,
    pub active_mask: u32,
    pub divergent: bool,
    pub threads: Vec<ThreadState>,
    pub shared_memory: Vec<u8>,
    pub local_memory: Vec<u8>,
}

// Thread state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadState {
    pub thread_id: u32,
    pub lane_id: u32,
    pub active: bool,
    pub registers: HashMap<String, RegisterValue>,
    pub predicates: Vec<bool>,
    pub call_stack: Vec<StackFrame>,
}

// Register value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegisterValue {
    Int32(i32),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Predicate(bool),
}

// Stack frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackFrame {
    pub function: String,
    pub pc: u64,
    pub source_location: Option<SourceLocation>,
    pub locals: HashMap<String, RegisterValue>,
}

// Memory watchpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Watchpoint {
    pub id: u32,
    pub address: u64,
    pub size: usize,
    pub watch_type: WatchType,
    pub condition: Option<String>,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WatchType {
    Read,
    Write,
    ReadWrite,
}

// Debugger implementation
pub struct Debugger {
    // Breakpoints and watchpoints
    breakpoints: Arc<DashMap<u32, Breakpoint>>,
    watchpoints: Arc<DashMap<u32, Watchpoint>>,
    breakpoint_counter: std::sync::atomic::AtomicU32,
    watchpoint_counter: std::sync::atomic::AtomicU32,
    
    // Warp states
    warp_states: Arc<DashMap<u32, WarpState>>,
    suspended_warps: Arc<DashMap<u32, SuspendedWarp>>,
    
    // Step control
    step_mode: Arc<DashMap<u32, StepMode>>,
    
    // Divergence tracking
    divergence_tracker: DivergenceTracker,
    
    // Debug hooks
    debug_hooks: Arc<DashMap<String, DebugHook>>,
}

// Suspended warp information
struct SuspendedWarp {
    warp_id: u32,
    suspend_reason: SuspendReason,
    breakpoint_id: Option<u32>,
    watchpoint_id: Option<u32>,
    saved_state: WarpState,
}

#[derive(Debug, Clone)]
enum SuspendReason {
    Breakpoint,
    Watchpoint,
    Step,
    Exception,
    UserRequest,
}

// Step mode for single-stepping
#[derive(Debug, Clone)]
enum StepMode {
    StepInto,
    StepOver,
    StepOut,
    Continue,
}

// Divergence tracking
struct DivergenceTracker {
    divergence_points: HashMap<u64, DivergencePoint>,
    reconvergence_stack: Vec<u64>,
}

struct DivergencePoint {
    pc: u64,
    taken_mask: u32,
    not_taken_mask: u32,
    reconvergence_pc: u64,
}

// Debug hook for custom debugging
type DebugHook = Box<dyn Fn(&WarpState) -> bool + Send + Sync>;

impl Debugger {
    pub fn new() -> Result<Self> {
        Ok(Self {
            breakpoints: Arc::new(DashMap::new()),
            watchpoints: Arc::new(DashMap::new()),
            breakpoint_counter: std::sync::atomic::AtomicU32::new(0),
            watchpoint_counter: std::sync::atomic::AtomicU32::new(0),
            warp_states: Arc::new(DashMap::new()),
            suspended_warps: Arc::new(DashMap::new()),
            step_mode: Arc::new(DashMap::new()),
            divergence_tracker: DivergenceTracker {
                divergence_points: HashMap::new(),
                reconvergence_stack: Vec::new(),
            },
            debug_hooks: Arc::new(DashMap::new()),
        })
    }
    
    // Set breakpoint
    pub fn set_breakpoint(&mut self, location: crate::BreakpointLocation) -> Result<u32> {
        let id = self.breakpoint_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        let bp_location = match location {
            crate::BreakpointLocation::SourceLine { file, line } => {
                BreakpointLocation::SourceLine { file, line }
            }
            crate::BreakpointLocation::KernelFunction { name } => {
                BreakpointLocation::Kernel(name)
            }
            crate::BreakpointLocation::IrLocation { location } => {
                BreakpointLocation::Address(location.parse().unwrap_or(0))
            }
            crate::BreakpointLocation::WarpDivergence { .. } => {
                BreakpointLocation::Divergence
            }
            crate::BreakpointLocation::MemoryAddress { address, .. } => {
                BreakpointLocation::Address(address)
            }
        };
        
        let breakpoint = Breakpoint {
            id,
            location: bp_location,
            condition: None,
            hit_count: 0,
            enabled: true,
            temporary: false,
            thread_mask: None,
        };
        
        self.breakpoints.insert(id, breakpoint);
        Ok(id)
    }
    
    // Remove breakpoint
    pub fn remove_breakpoint(&mut self, id: u32) -> Result<()> {
        self.breakpoints.remove(&id)
            .ok_or_else(|| anyhow::anyhow!("Breakpoint {} not found", id))?;
        Ok(())
    }
    
    // Set watchpoint
    pub fn set_watchpoint(&mut self, address: u64, size: usize, 
                          watch_type: WatchType) -> Result<u32> {
        let id = self.watchpoint_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        let watchpoint = Watchpoint {
            id,
            address,
            size,
            watch_type,
            condition: None,
            enabled: true,
        };
        
        self.watchpoints.insert(id, watchpoint);
        Ok(id)
    }
    
    // Get warp state
    pub fn get_warp_state(&self, warp_id: u32) -> Result<WarpState> {
        self.warp_states.get(&warp_id)
            .map(|state| state.clone())
            .ok_or_else(|| anyhow::anyhow!("Warp {} not found", warp_id))
    }
    
    // Update warp state
    pub fn update_warp_state(&self, warp_id: u32, state: WarpState) {
        // Check for breakpoints
        if self.check_breakpoints(&state) {
            self.suspend_warp(warp_id, SuspendReason::Breakpoint, state.clone());
        } else {
            self.warp_states.insert(warp_id, state);
        }
    }
    
    // Step control
    pub fn step(&mut self, warp_id: u32, mode: StepMode) -> Result<()> {
        self.step_mode.insert(warp_id, mode);
        self.resume_warp(warp_id)
    }
    
    // Continue execution
    pub fn continue_execution(&mut self, warp_id: u32) -> Result<()> {
        self.step_mode.insert(warp_id, StepMode::Continue);
        self.resume_warp(warp_id)
    }
    
    // Evaluate expression in warp context
    pub fn evaluate(&self, warp_id: u32, expression: &str) -> Result<RegisterValue> {
        let state = self.get_warp_state(warp_id)?;
        
        // Simple expression evaluation
        if let Some(thread) = state.threads.first() {
            if let Some(value) = thread.registers.get(expression) {
                return Ok(value.clone());
            }
        }
        
        // Would implement full expression parser
        anyhow::bail!("Cannot evaluate expression: {}", expression)
    }
    
    // Get call stack
    pub fn get_call_stack(&self, warp_id: u32, thread_id: u32) -> Result<Vec<StackFrame>> {
        let state = self.get_warp_state(warp_id)?;
        
        state.threads.iter()
            .find(|t| t.thread_id == thread_id)
            .map(|t| t.call_stack.clone())
            .ok_or_else(|| anyhow::anyhow!("Thread {} not found in warp {}", thread_id, warp_id))
    }
    
    // Track divergence
    pub fn track_divergence(&mut self, pc: u64, taken_mask: u32, not_taken_mask: u32,
                           reconvergence_pc: u64) {
        let point = DivergencePoint {
            pc,
            taken_mask,
            not_taken_mask,
            reconvergence_pc,
        };
        
        self.divergence_tracker.divergence_points.insert(pc, point);
        self.divergence_tracker.reconvergence_stack.push(reconvergence_pc);
    }
    
    // Get divergence info
    pub fn get_divergence_info(&self, pc: u64) -> Option<(u32, u32)> {
        self.divergence_tracker.divergence_points.get(&pc)
            .map(|p| (p.taken_mask, p.not_taken_mask))
    }
    
    // Private helper methods
    
    fn check_breakpoints(&self, state: &WarpState) -> bool {
        for entry in self.breakpoints.iter() {
            let bp = entry.value();
            if !bp.enabled {
                continue;
            }
            
            match &bp.location {
                BreakpointLocation::Address(addr) => {
                    if state.pc == *addr {
                        return true;
                    }
                }
                BreakpointLocation::Divergence => {
                    if state.divergent {
                        return true;
                    }
                }
                _ => {}
            }
        }
        
        false
    }
    
    fn check_watchpoints(&self, address: u64, size: usize, is_write: bool) -> Option<u32> {
        for entry in self.watchpoints.iter() {
            let wp = entry.value();
            if !wp.enabled {
                continue;
            }
            
            let in_range = address >= wp.address && 
                          address < wp.address + wp.size as u64;
            
            if in_range {
                match wp.watch_type {
                    WatchType::Write if is_write => return Some(wp.id),
                    WatchType::Read if !is_write => return Some(wp.id),
                    WatchType::ReadWrite => return Some(wp.id),
                    _ => {}
                }
            }
        }
        
        None
    }
    
    fn suspend_warp(&self, warp_id: u32, reason: SuspendReason, state: WarpState) {
        let suspended = SuspendedWarp {
            warp_id,
            suspend_reason: reason,
            breakpoint_id: None,
            watchpoint_id: None,
            saved_state: state,
        };
        
        self.suspended_warps.insert(warp_id, suspended);
    }
    
    fn resume_warp(&self, warp_id: u32) -> Result<()> {
        self.suspended_warps.remove(&warp_id)
            .ok_or_else(|| anyhow::anyhow!("Warp {} not suspended", warp_id))?;
        Ok(())
    }
}

// Debugging utilities
pub struct DebuggerUtils;

impl DebuggerUtils {
    // Format warp state for display
    pub fn format_warp_state(state: &WarpState) -> String {
        format!(
            "Warp {} (SM {}, Block {}):\n  PC: 0x{:x}\n  Active: 0x{:08x}\n  Divergent: {}",
            state.warp_id, state.sm_id, state.block_id, 
            state.pc, state.active_mask, state.divergent
        )
    }
    
    // Format thread state
    pub fn format_thread_state(thread: &ThreadState) -> String {
        format!(
            "Thread {} (Lane {}):\n  Active: {}\n  Registers: {}",
            thread.thread_id, thread.lane_id, thread.active,
            thread.registers.len()
        )
    }
    
    // Check if warp is at divergence point
    pub fn is_divergent(state: &WarpState) -> bool {
        let active_count = state.active_mask.count_ones();
        active_count > 0 && active_count < 32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_debugger_creation() {
        let debugger = Debugger::new();
        assert!(debugger.is_ok());
    }
    
    #[test]
    fn test_breakpoint_setting() {
        let mut debugger = Debugger::new().unwrap();
        let id = debugger.set_breakpoint(crate::BreakpointLocation::KernelFunction {
            name: "test_kernel".to_string()
        });
        assert!(id.is_ok());
    }
    
    #[test]
    fn test_divergence_check() {
        let state = WarpState {
            warp_id: 0,
            sm_id: 0,
            block_id: 0,
            pc: 0x1000,
            active_mask: 0x0000FFFF, // Half threads active
            divergent: true,
            threads: Vec::new(),
            shared_memory: Vec::new(),
            local_memory: Vec::new(),
        };
        
        assert!(DebuggerUtils::is_divergent(&state));
    }
}