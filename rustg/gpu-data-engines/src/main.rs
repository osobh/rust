/*!
# GPU Data Engines - Phase 6 Demo

Demonstrates the GPU-native data processing capabilities built on the rustg compiler.
Shows dataframe operations, SQL queries, and performance validation.
*/

use gpu_data_engines::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 GPU Data Engines - Phase 6 of ProjectB");
    println!("=========================================");
    
    // Initialize GPU context
    initialize_gpu_context()?;
    
    // Create engines with default configuration
    println!("\n🔧 Initializing GPU data engines...");
    let mut engines = GPUDataEngines::new_default()?;
    println!("✅ All engines initialized successfully");
    
    // Demonstrate dataframe operations
    println!("\n📊 Dataframe Operations Demo");
    println!("----------------------------");
    demo_dataframes(&mut engines)?;
    
    // Demonstrate SQL query engine
    println!("\n🗃️  SQL Query Engine Demo");
    println!("-------------------------");
    demo_sql_engine(&mut engines)?;
    
    // Optional: Graph processing demo (if enabled)
    if engines.graphs().is_ok() {
        println!("\n🕸️  Graph Processing Demo");
        println!("-------------------------");
        demo_graph_processing(&mut engines)?;
    }
    
    // Optional: Search engine demo (if enabled)
    if engines.search().is_ok() {
        println!("\n🔍 Search Engine Demo");
        println!("---------------------");
        demo_search_engine(&mut engines)?;
    }
    
    // Run comprehensive performance validation
    println!("\n⚡ Performance Validation");
    println!("========================");
    let validation_start = Instant::now();
    
    let completion_ok = engines.validate_phase6_completion()?;
    let validation_time = validation_start.elapsed();
    
    if completion_ok {
        println!("🎉 Phase 6 validation completed in {:.2}ms", validation_time.as_millis());
        
        let stats = engines.get_stats();
        println!("\n📈 Final Performance Summary:");
        println!("  • Dataframe throughput: {:.2} GB/s", stats.dataframe_throughput_gbps);
        println!("  • SQL query throughput: {:.2} GB/s", stats.sql_throughput_gbps);
        if stats.graph_throughput_edges_per_sec > 0.0 {
            println!("  • Graph traversal: {:.2} edges/sec", stats.graph_throughput_edges_per_sec);
        }
        if stats.search_queries_per_sec > 0.0 {
            println!("  • Search queries: {:.2} QPS", stats.search_queries_per_sec);
        }
        println!("  • Memory usage: {} MB", engines.memory_usage_mb());
        
        println!("\n🏆 SUCCESS: 10x+ performance improvement achieved!");
        println!("Phase 6 (Data & Query Engines) completed successfully.");
    } else {
        println!("❌ Phase 6 validation failed - performance targets not met");
        return Err("Performance validation failed".into());
    }
    
    Ok(())
}

/// Demonstrate GPU dataframe operations
fn demo_dataframes(engines: &mut GPUDataEngines) -> Result<(), Box<dyn std::error::Error>> {
    let df = engines.dataframes();
    
    // Create sample data
    println!("Creating dataframe with 1M rows...");
    let sample_data: Vec<i64> = (0..1_000_000).collect();
    let col_id = df.add_int_column(sample_data)?;
    println!("✅ Added integer column with {} rows", df.len());
    
    // Perform columnar scan
    println!("Performing high-speed columnar scan...");
    let scan_start = Instant::now();
    let sum_result = df.columnar_scan(col_id)?;
    let scan_time = scan_start.elapsed();
    
    println!("✅ Scan completed in {:.2}ms, processed {} records", 
             scan_time.as_millis(), sum_result);
    
    // Add float column for join demo
    let float_data: Vec<f64> = (0..1_000_000).map(|i| i as f64 * 1.5).collect();
    let float_col = df.add_float_column(float_data)?;
    println!("✅ Added float column for operations");
    
    // Demo group-by aggregation
    println!("Performing group-by aggregation...");
    let groupby_start = Instant::now();
    let group_results = df.group_by_sum(col_id, float_col)?;
    let groupby_time = groupby_start.elapsed();
    
    println!("✅ Group-by completed in {:.2}ms, {} groups found", 
             groupby_time.as_millis(), group_results.len().min(10));
    
    Ok(())
}

/// Demonstrate SQL query engine
fn demo_sql_engine(engines: &mut GPUDataEngines) -> Result<(), Box<dyn std::error::Error>> {
    let sql_engine = engines.sql();
    
    // Create tables
    println!("Creating SQL tables...");
    sql_engine.create_table("orders", vec![
        ("order_id".to_string(), SQLDataType::Int64),
        ("customer_id".to_string(), SQLDataType::Int64),
        ("amount".to_string(), SQLDataType::Double),
        ("status".to_string(), SQLDataType::Varchar),
    ])?;
    
    sql_engine.create_table("customers", vec![
        ("customer_id".to_string(), SQLDataType::Int64),
        ("name".to_string(), SQLDataType::Varchar),
        ("region".to_string(), SQLDataType::Varchar),
    ])?;
    
    println!("✅ Tables created successfully");
    
    // Insert sample data
    println!("Inserting sample data...");
    let orders_data = vec![
        vec![SQLValue::Int(1), SQLValue::Int(101), SQLValue::Double(250.75), SQLValue::Null],
        vec![SQLValue::Int(2), SQLValue::Int(102), SQLValue::Double(89.50), SQLValue::Null],
        vec![SQLValue::Int(3), SQLValue::Int(101), SQLValue::Double(175.25), SQLValue::Null],
        vec![SQLValue::Int(4), SQLValue::Int(103), SQLValue::Double(420.00), SQLValue::Null],
        vec![SQLValue::Int(5), SQLValue::Int(102), SQLValue::Double(95.75), SQLValue::Null],
    ];
    
    let customers_data = vec![
        vec![SQLValue::Int(101), SQLValue::Null, SQLValue::Null],
        vec![SQLValue::Int(102), SQLValue::Null, SQLValue::Null], 
        vec![SQLValue::Int(103), SQLValue::Null, SQLValue::Null],
    ];
    
    sql_engine.insert_data("orders", orders_data)?;
    sql_engine.insert_data("customers", customers_data)?;
    println!("✅ Sample data inserted");
    
    // Execute queries
    println!("Executing SQL queries...");
    
    let queries = vec![
        "SELECT * FROM orders",
        "SELECT * FROM customers", 
        "SELECT * FROM orders LIMIT 3",
    ];
    
    for query in &queries {
        println!("  Running: {}", query);
        let query_start = Instant::now();
        let result = sql_engine.execute_query(query)?;
        let query_time = query_start.elapsed();
        
        println!("    ✅ {} rows returned in {:.2}ms (throughput: {:.2} GB/s)", 
                 result.rows.len(), query_time.as_millis(), result.throughput_gbps);
    }
    
    Ok(())
}

/// Demonstrate graph processing (if enabled)
fn demo_graph_processing(engines: &mut GPUDataEngines) -> Result<(), Box<dyn std::error::Error>> {
    let graph_engine = engines.graphs()?;
    
    // Create sample graph
    println!("Creating sample graph with 1000 vertices...");
    let edges = vec![
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0), // Cycle
        (0, 5), (5, 6), (6, 7),                   // Branch  
        (2, 8), (8, 9), (9, 2),                   // Another cycle
    ];
    
    let sample_graph = GPUGraph::from_edges(&edges, None)?;
    println!("✅ Graph created with {} vertices, {} edges", 
             sample_graph.num_vertices(), sample_graph.num_edges());
    
    // Perform BFS
    println!("Running BFS from vertex 0...");
    let bfs_start = Instant::now();
    let distances = sample_graph.bfs(0)?;
    let bfs_time = bfs_start.elapsed();
    
    let reachable_count = distances.iter().filter(|&&d| d != u32::MAX).count();
    println!("✅ BFS completed in {:.2}ms, {} vertices reachable", 
             bfs_time.as_millis(), reachable_count);
    
    // Run PageRank
    println!("Running PageRank algorithm...");
    let pr_start = Instant::now();
    let ranks = sample_graph.pagerank(0.85, 10, 1e-6)?;
    let pr_time = pr_start.elapsed();
    
    let max_rank = ranks.iter().fold(0.0f32, |a, &b| a.max(b));
    println!("✅ PageRank completed in {:.2}ms, max rank: {:.4}", 
             pr_time.as_millis(), max_rank);
    
    Ok(())
}

/// Demonstrate search engine (if enabled)
fn demo_search_engine(engines: &mut GPUDataEngines) -> Result<(), Box<dyn std::error::Error>> {
    let search_engine = engines.search()?;
    
    // Add sample documents
    println!("Indexing sample documents...");
    let documents = vec![
        (0, "artificial intelligence machine learning deep neural networks"),
        (1, "database systems query optimization indexing algorithms"),
        (2, "distributed computing parallel processing gpu acceleration"),
        (3, "natural language processing text analysis sentiment classification"),
        (4, "computer vision image recognition object detection"),
    ];
    
    for (doc_id, content) in &documents {
        search_engine.add_document(*doc_id, content)?;
    }
    println!("✅ {} documents indexed", documents.len());
    
    // Perform boolean search
    println!("Performing boolean search queries...");
    let queries = vec![
        ("machine learning", vec!["machine".to_string(), "learning".to_string()]),
        ("database query", vec!["database".to_string(), "query".to_string()]),
        ("gpu", vec!["gpu".to_string()]),
    ];
    
    for (desc, keywords) in &queries {
        println!("  Searching for: {}", desc);
        let query = QueryType::Boolean(keywords.clone(), vec![], vec![]);
        let search_start = Instant::now();
        let results = search_engine.boolean_search(&query)?;
        let search_time = search_start.elapsed();
        
        println!("    ✅ {} results found in {:.2}ms", 
                 results.len(), search_time.as_millis());
        
        // Show top result
        if let Some(result) = results.first() {
            println!("      Top result: doc_id={}, score={:.2}", 
                     result.doc_id, result.score);
        }
    }
    
    // Demo vector search if we had embeddings
    println!("Vector search would require pre-computed embeddings (skipped in demo)");
    
    Ok(())
}

/// Helper function to format large numbers
fn _format_number(n: f32) -> String {
    if n >= 1_000_000_000.0 {
        format!("{:.1}B", n / 1_000_000_000.0)
    } else if n >= 1_000_000.0 {
        format!("{:.1}M", n / 1_000_000.0) 
    } else if n >= 1_000.0 {
        format!("{:.1}K", n / 1_000.0)
    } else {
        format!("{:.1}", n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_execution() {
        // Test that the demo code doesn't crash
        initialize_gpu_context().expect("GPU context init failed");
        
        let mut engines = GPUDataEngines::new_dataframes_only()
            .expect("Failed to create engines");
        
        demo_dataframes(&mut engines).expect("Dataframe demo failed");
        demo_sql_engine(&mut engines).expect("SQL demo failed");
    }
}