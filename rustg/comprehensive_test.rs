/// Comprehensive test library for rustdoc-g
/// 
/// This library demonstrates various Rust constructs and documentation features
/// that rustdoc-g can process.
/// 
/// # Examples
/// 
/// ```rust
/// use comprehensive_test::*;
/// let processor = DataProcessor::new();
/// assert_eq!(processor.process_data(42), 84);
/// ```

/// A trait defining data processing capabilities
pub trait Processor {
    /// Process a single value
    fn process_data(&self, value: i32) -> i32;
    
    /// Batch process multiple values  
    fn batch_process(&self, values: &[i32]) -> Vec<i32> {
        values.iter().map(|&v| self.process_data(v)).collect()
    }
}

/// Main data processor implementing the Processor trait
/// 
/// This struct handles data processing operations with configurable behavior.
/// It demonstrates struct documentation and field documentation.
pub struct DataProcessor {
    /// Multiplier for processing operations
    pub multiplier: i32,
    /// Flag to enable debug mode
    pub debug: bool,
}

impl DataProcessor {
    /// Create a new DataProcessor with default settings
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// let processor = DataProcessor::new();
    /// assert_eq!(processor.multiplier, 2);
    /// ```
    pub fn new() -> Self {
        Self {
            multiplier: 2,
            debug: false,
        }
    }
    
    /// Create a DataProcessor with custom multiplier
    /// 
    /// # Arguments
    /// 
    /// * `multiplier` - The multiplication factor to use
    /// 
    /// # Returns
    /// 
    /// A new DataProcessor instance
    pub fn with_multiplier(multiplier: i32) -> Self {
        Self {
            multiplier,
            debug: false,
        }
    }
}

impl Processor for DataProcessor {
    /// Process data by multiplying with the configured multiplier
    fn process_data(&self, value: i32) -> i32 {
        if self.debug {
            println!("Processing value: {}", value);
        }
        value * self.multiplier
    }
}

/// Enumeration of processing results
#[derive(Debug, PartialEq)]
pub enum ProcessingResult {
    /// Operation completed successfully
    Success(i32),
    /// Operation failed with error message
    Error(String),
    /// Operation was skipped
    Skipped,
}

/// Configuration options for data processing
#[derive(Clone)]
pub struct ProcessingConfig {
    /// Maximum value to process
    pub max_value: i32,
    /// Whether to skip negative values
    pub skip_negative: bool,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            max_value: 1000,
            skip_negative: true,
        }
    }
}

/// Advanced processor with configuration and error handling
pub struct AdvancedProcessor {
    config: ProcessingConfig,
    processor: DataProcessor,
}

impl AdvancedProcessor {
    /// Create a new AdvancedProcessor
    pub fn new(config: ProcessingConfig) -> Self {
        Self {
            config,
            processor: DataProcessor::new(),
        }
    }
    
    /// Process value with error handling and validation
    /// 
    /// # Arguments
    /// 
    /// * `value` - The value to process
    /// 
    /// # Returns
    /// 
    /// A ProcessingResult indicating success or failure
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// let config = ProcessingConfig::default();
    /// let processor = AdvancedProcessor::new(config);
    /// match processor.safe_process(42) {
    ///     ProcessingResult::Success(result) => println!("Result: {}", result),
    ///     ProcessingResult::Error(msg) => println!("Error: {}", msg),
    ///     ProcessingResult::Skipped => println!("Skipped"),
    /// }
    /// ```
    pub fn safe_process(&self, value: i32) -> ProcessingResult {
        if value < 0 && self.config.skip_negative {
            return ProcessingResult::Skipped;
        }
        
        if value > self.config.max_value {
            return ProcessingResult::Error(format!("Value {} exceeds maximum {}", value, self.config.max_value));
        }
        
        ProcessingResult::Success(self.processor.process_data(value))
    }
}

/// Utility functions for data processing
pub mod utils {
    /// Calculate the sum of processed values
    /// 
    /// # Arguments  
    /// 
    /// * `values` - Slice of values to sum after processing
    /// * `multiplier` - Factor to multiply each value by
    /// 
    /// # Returns
    /// 
    /// The sum of all processed values
    pub fn sum_processed(values: &[i32], multiplier: i32) -> i32 {
        values.iter().map(|&v| v * multiplier).sum()
    }
    
    /// Find the maximum processed value
    pub fn max_processed(values: &[i32], multiplier: i32) -> Option<i32> {
        values.iter().map(|&v| v * multiplier).max()
    }
}

/// Constants for processing
pub mod constants {
    /// Default processing multiplier
    pub const DEFAULT_MULTIPLIER: i32 = 2;
    /// Maximum safe processing value
    pub const MAX_SAFE_VALUE: i32 = 10_000;
    /// Processing version string
    pub const VERSION: &str = "1.0.0";
}