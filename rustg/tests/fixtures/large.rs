// Large test fixture for performance testing

use std::collections::HashMap;
use std::vec::Vec;

#[derive(Debug, Clone)]
struct DataPoint {
    id: u64,
    value: f64,
    label: String,
    metadata: HashMap<String, String>,
}

impl DataPoint {
    fn new(id: u64, value: f64) -> Self {
        let mut metadata = HashMap::new();
        metadata.insert("created".to_string(), "2024-01-01".to_string());
        metadata.insert("version".to_string(), "1.0".to_string());
        
        Self {
            id,
            value,
            label: format!("point_{}", id),
            metadata,
        }
    }
    
    fn process(&mut self) -> Result<(), String> {
        if self.value < 0.0 {
            return Err("Negative value not allowed".to_string());
        }
        
        self.value = self.value.sqrt() * 2.0;
        self.metadata.insert("processed".to_string(), "true".to_string());
        
        Ok(())
    }
}

fn generate_data(count: usize) -> Vec<DataPoint> {
    let mut data = Vec::with_capacity(count);
    
    for i in 0..count {
        let point = DataPoint::new(i as u64, (i as f64) * 1.5);
        data.push(point);
    }
    
    data
}

fn process_batch(data: &mut [DataPoint]) -> Result<usize, String> {
    let mut processed = 0;
    
    for point in data.iter_mut() {
        match point.process() {
            Ok(_) => processed += 1,
            Err(e) => eprintln!("Error processing {}: {}", point.id, e),
        }
    }
    
    Ok(processed)
}

fn main() {
    println!("Starting large-scale data processing...");
    
    // Generate test data
    let mut data = generate_data(10000);
    
    // Process in batches
    let batch_size = 100;
    let mut total_processed = 0;
    
    for chunk in data.chunks_mut(batch_size) {
        match process_batch(chunk) {
            Ok(count) => total_processed += count,
            Err(e) => eprintln!("Batch processing error: {}", e),
        }
    }
    
    println!("Processed {} data points", total_processed);
    
    // Additional complex code patterns for tokenizer testing
    let closure = |x: i32, y: i32| -> i32 {
        let sum = x + y;
        let product = x * y;
        if sum > product {
            sum
        } else {
            product
        }
    };
    
    let result = closure(5, 10);
    println!("Closure result: {}", result);
    
    // Pattern matching
    let value = Some(42);
    match value {
        Some(x) if x > 40 => println!("Large value: {}", x),
        Some(x) => println!("Small value: {}", x),
        None => println!("No value"),
    }
    
    // Loop constructs
    let mut counter = 0;
    'outer: loop {
        counter += 1;
        
        for i in 0..10 {
            if i == 5 && counter > 3 {
                break 'outer;
            }
        }
        
        if counter > 100 {
            break;
        }
    }
    
    // Macro usage
    macro_rules! create_function {
        ($func_name:ident, $value:expr) => {
            fn $func_name() -> i32 {
                $value
            }
        };
    }
    
    create_function!(get_answer, 42);
    println!("The answer is: {}", get_answer());
}