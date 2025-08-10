// Flamegraph Module - GPU-aware flamegraph generation
// Implements flamegraph visualization for GPU profiling

use anyhow::{Result, Context};
use std::collections::HashMap;
use std::io::Write;
use serde::{Serialize, Deserialize};
use crate::profiler::ProfileData;

// Flamegraph structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlameGraph {
    pub root: FlameNode,
    pub total_samples: u64,
    pub total_time_ns: u64,
    pub metadata: FlameGraphMetadata,
}

// Flamegraph node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlameNode {
    pub name: String,
    pub value: u64,  // Time in nanoseconds or sample count
    pub children: Vec<FlameNode>,
    pub metadata: NodeMetadata,
}

// Node metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetadata {
    pub kernel_name: Option<String>,
    pub source_location: Option<String>,
    pub warp_efficiency: Option<f32>,
    pub memory_efficiency: Option<f32>,
    pub sm_id: Option<u32>,
    pub color_hint: Option<String>,
}

// Flamegraph metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlameGraphMetadata {
    pub title: String,
    pub units: String,
    pub gpu_model: String,
    pub total_sms: u32,
    pub max_warps_per_sm: u32,
}

impl FlameGraph {
    // Create from profile data
    pub fn from_profile_data(profile_data: &ProfileData) -> Result<Self> {
        let mut builder = FlameGraphBuilder::new();
        
        // Build call tree from samples
        for sample in &profile_data.samples {
            builder.add_sample(&sample.kernel_name, sample.timestamp_ns);
        }
        
        // Add kernel statistics
        for (kernel_name, stats) in &profile_data.kernel_stats {
            builder.add_kernel_stats(kernel_name, stats.total_time_ns);
        }
        
        builder.build()
    }
    
    // Generate SVG output
    pub fn to_svg(&self) -> Result<String> {
        let mut svg = String::new();
        
        // SVG header
        svg.push_str(&format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="800">
            <rect x="0" y="0" width="1200" height="800" fill="white"/>
            <text x="600" y="20" text-anchor="middle" font-size="16" font-weight="bold">{}</text>"#,
            self.metadata.title
        ));
        
        // Render nodes recursively
        self.render_node(&self.root, &mut svg, 10.0, 40.0, 1180.0, 20.0, 0);
        
        svg.push_str("</svg>");
        Ok(svg)
    }
    
    // Generate folded stack format
    pub fn to_folded(&self) -> String {
        let mut output = String::new();
        self.write_folded_node(&self.root, &mut output, Vec::new());
        output
    }
    
    // Generate Brendan Gregg's flamegraph format
    pub fn to_flamegraph_format(&self) -> Result<Vec<u8>> {
        let folded = self.to_folded();
        
        // Use inferno crate to generate flamegraph
        let mut opts = inferno::flamegraph::Options::default();
        opts.title = self.metadata.title.clone();
        opts.count_name = self.metadata.units.clone();
        
        let mut output = Vec::new();
        inferno::flamegraph::from_reader(
            &mut opts,
            folded.as_bytes(),
            &mut output
        )?;
        
        Ok(output)
    }
    
    // Find hottest path
    pub fn find_hottest_path(&self) -> Vec<String> {
        let mut path = Vec::new();
        let mut current = &self.root;
        
        path.push(current.name.clone());
        
        while !current.children.is_empty() {
            // Find child with maximum value
            current = current.children.iter()
                .max_by_key(|c| c.value)
                .unwrap();
            path.push(current.name.clone());
        }
        
        path
    }
    
    // Calculate total time for kernel
    pub fn kernel_time(&self, kernel_name: &str) -> u64 {
        self.sum_matching_nodes(&self.root, |node| {
            node.metadata.kernel_name.as_ref() == Some(&kernel_name.to_string())
        })
    }
    
    // Private helper methods
    
    fn render_node(&self, node: &FlameNode, svg: &mut String, 
                   x: f64, y: f64, width: f64, height: f64, depth: usize) {
        if width < 1.0 {
            return; // Too small to render
        }
        
        // Calculate color based on efficiency
        let color = self.get_node_color(node);
        
        // Render rectangle
        svg.push_str(&format!(
            r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" stroke="white" stroke-width="1"/>"#,
            x, y, width, height, color
        ));
        
        // Render text if wide enough
        if width > 50.0 {
            let text = if node.name.len() * 8 > width as usize {
                &node.name[..width as usize / 8]
            } else {
                &node.name
            };
            
            svg.push_str(&format!(
                r#"<text x="{}" y="{}" font-size="12" text-anchor="middle">{}</text>"#,
                x + width / 2.0, y + height / 2.0 + 4.0, text
            ));
        }
        
        // Render children
        if !node.children.is_empty() {
            let total_value: u64 = node.children.iter().map(|c| c.value).sum();
            let mut child_x = x;
            
            for child in &node.children {
                let child_width = (child.value as f64 / total_value as f64) * width;
                self.render_node(child, svg, child_x, y + height + 1.0, 
                               child_width, height, depth + 1);
                child_x += child_width;
            }
        }
    }
    
    fn get_node_color(&self, node: &FlameNode) -> String {
        if let Some(hint) = &node.metadata.color_hint {
            return hint.clone();
        }
        
        // Color based on efficiency
        if let Some(efficiency) = node.metadata.warp_efficiency {
            if efficiency > 0.9 {
                "#4CAF50".to_string() // Green - efficient
            } else if efficiency > 0.7 {
                "#FFC107".to_string() // Yellow - moderate
            } else {
                "#F44336".to_string() // Red - inefficient
            }
        } else {
            "#2196F3".to_string() // Blue - default
        }
    }
    
    fn write_folded_node(&self, node: &FlameNode, output: &mut String, 
                        mut stack: Vec<String>) {
        stack.push(node.name.clone());
        
        if node.children.is_empty() {
            // Leaf node - write stack
            output.push_str(&stack.join(";"));
            output.push_str(&format!(" {}\n", node.value));
        } else {
            // Recurse to children
            for child in &node.children {
                self.write_folded_node(child, output, stack.clone());
            }
        }
    }
    
    fn sum_matching_nodes<F>(&self, node: &FlameNode, predicate: F) -> u64
    where
        F: Fn(&FlameNode) -> bool + Copy,
    {
        let mut sum = 0;
        if predicate(node) {
            sum += node.value;
        }
        
        for child in &node.children {
            sum += self.sum_matching_nodes(child, predicate);
        }
        
        sum
    }
}

// Flamegraph builder
struct FlameGraphBuilder {
    root: FlameNode,
    stacks: HashMap<String, Vec<String>>,
    total_samples: u64,
    total_time_ns: u64,
}

impl FlameGraphBuilder {
    fn new() -> Self {
        Self {
            root: FlameNode {
                name: "root".to_string(),
                value: 0,
                children: Vec::new(),
                metadata: NodeMetadata {
                    kernel_name: None,
                    source_location: None,
                    warp_efficiency: None,
                    memory_efficiency: None,
                    sm_id: None,
                    color_hint: None,
                },
            },
            stacks: HashMap::new(),
            total_samples: 0,
            total_time_ns: 0,
        }
    }
    
    fn add_sample(&mut self, kernel_name: &str, time_ns: u64) {
        self.total_samples += 1;
        self.total_time_ns += time_ns;
        
        // Build stack for kernel
        let stack = vec![kernel_name.to_string()];
        self.add_stack(&stack, time_ns);
    }
    
    fn add_kernel_stats(&mut self, kernel_name: &str, total_time_ns: u64) {
        let stack = vec![kernel_name.to_string()];
        self.add_stack(&stack, total_time_ns);
    }
    
    fn add_stack(&mut self, stack: &[String], value: u64) {
        let mut current = &mut self.root;
        current.value += value;
        
        for frame in stack {
            // Find or create child
            let child_idx = current.children.iter()
                .position(|c| c.name == *frame);
            
            if let Some(idx) = child_idx {
                current = &mut current.children[idx];
            } else {
                current.children.push(FlameNode {
                    name: frame.clone(),
                    value: 0,
                    children: Vec::new(),
                    metadata: NodeMetadata {
                        kernel_name: Some(frame.clone()),
                        source_location: None,
                        warp_efficiency: None,
                        memory_efficiency: None,
                        sm_id: None,
                        color_hint: None,
                    },
                });
                current = current.children.last_mut().unwrap();
            }
            
            current.value += value;
        }
    }
    
    fn build(self) -> Result<FlameGraph> {
        Ok(FlameGraph {
            root: self.root,
            total_samples: self.total_samples,
            total_time_ns: self.total_time_ns,
            metadata: FlameGraphMetadata {
                title: "GPU Kernel Flamegraph".to_string(),
                units: "nanoseconds".to_string(),
                gpu_model: "Unknown".to_string(),
                total_sms: 80, // Default for modern GPUs
                max_warps_per_sm: 32,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_flamegraph_builder() {
        let mut builder = FlameGraphBuilder::new();
        builder.add_sample("kernel_a", 1000);
        builder.add_sample("kernel_b", 2000);
        
        let flamegraph = builder.build().unwrap();
        assert_eq!(flamegraph.total_samples, 2);
        assert_eq!(flamegraph.total_time_ns, 3000);
    }
    
    #[test]
    fn test_folded_format() {
        let flamegraph = FlameGraph {
            root: FlameNode {
                name: "root".to_string(),
                value: 100,
                children: vec![
                    FlameNode {
                        name: "kernel_a".to_string(),
                        value: 60,
                        children: Vec::new(),
                        metadata: NodeMetadata::default(),
                    },
                    FlameNode {
                        name: "kernel_b".to_string(),
                        value: 40,
                        children: Vec::new(),
                        metadata: NodeMetadata::default(),
                    },
                ],
                metadata: NodeMetadata::default(),
            },
            total_samples: 100,
            total_time_ns: 100,
            metadata: FlameGraphMetadata {
                title: "Test".to_string(),
                units: "samples".to_string(),
                gpu_model: "Test GPU".to_string(),
                total_sms: 1,
                max_warps_per_sm: 1,
            },
        };
        
        let folded = flamegraph.to_folded();
        assert!(folded.contains("kernel_a"));
        assert!(folded.contains("kernel_b"));
    }
}

impl Default for NodeMetadata {
    fn default() -> Self {
        Self {
            kernel_name: None,
            source_location: None,
            warp_efficiency: None,
            memory_efficiency: None,
            sm_id: None,
            color_hint: None,
        }
    }
}