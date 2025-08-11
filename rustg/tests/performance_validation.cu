#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include "../include/gpu_types.h"

using namespace std::chrono;

namespace rustg {

// External kernel launch functions
extern "C" void launch_tokenizer_optimized(
    const char* source, size_t source_len,
    Token* tokens, uint32_t* token_count, uint32_t max_tokens);

extern "C" void launch_ast_construction(
    const Token* tokens, uint32_t token_count,
    ASTNodeGPU* nodes, uint32_t* node_count, uint32_t max_nodes);

extern "C" void launch_fused_tokenizer_ast(
    const char* source, size_t source_len,
    rustg::Token* tokens, uint32_t* token_count,
    rustg::ASTNode* nodes, uint32_t* node_count,
    uint32_t max_tokens, uint32_t max_nodes);

extern "C" void launch_advanced_syntax_processor(
    const char* source, size_t source_len,
    Token* tokens, uint32_t* token_count, uint32_t max_tokens);

// Performance metrics structure
struct PerfMetrics {
    double throughput_gbps;      // Gigabytes per second
    double tokens_per_second;    // Tokens processed per second
    double kernel_time_ms;       // Kernel execution time
    double memory_bandwidth_pct; // Memory bandwidth utilization
    double warp_efficiency;      // Warp execution efficiency
    double speedup_vs_cpu;       // Speedup compared to CPU
    size_t memory_used_bytes;    // GPU memory used
    size_t source_size_bytes;    // Input size
    uint32_t tokens_generated;   // Number of tokens produced
};

// CPU reference tokenizer for comparison
class CPUReferenceTokenizer {
public:
    uint32_t tokenize(const char* source, size_t len, Token* tokens, uint32_t max_tokens) {
        uint32_t count = 0;
        size_t pos = 0;
        
        while (pos < len && count < max_tokens) {
            char ch = source[pos];
            
            // Skip whitespace
            if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
                pos++;
                continue;
            }
            
            Token& token = tokens[count++];
            token.start_pos = pos;
            token.line = 1; // Simplified
            token.column = pos + 1;
            
            // Identify token type (simplified)
            if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_') {
                // Identifier
                token.type = TokenType::Identifier;
                while (pos < len && (isalnum(source[pos]) || source[pos] == '_')) {
                    pos++;
                }
            } else if (ch >= '0' && ch <= '9') {
                // Number
                token.type = TokenType::IntegerLiteral;
                while (pos < len && isdigit(source[pos])) {
                    pos++;
                }
            } else {
                // Punctuation
                token.type = TokenType::Plus; // Simplified
                pos++;
            }
            
            token.length = pos - token.start_pos;
        }
        
        return count;
    }
};

// Performance validation class
class PerformanceValidator {
private:
    cudaDeviceProp device_props;
    CPUReferenceTokenizer cpu_tokenizer;
    
    // Generate test source code
    std::string generate_test_source(size_t target_size) {
        std::string source;
        source.reserve(target_size);
        
        const char* code_snippets[] = {
            "fn process_data(input: &[u8], count: usize) -> Result<Vec<Token>, Error> {\n",
            "    let mut tokens = Vec::with_capacity(count * 2);\n",
            "    for (idx, &byte) in input.iter().enumerate() {\n",
            "        match classify_byte(byte) {\n",
            "            ByteClass::Identifier => tokens.push(Token::new(idx, 1)),\n",
            "            ByteClass::Number => process_number(&input[idx..], &mut tokens),\n",
            "            ByteClass::String => handle_string(&input[idx..], &mut tokens)?,\n",
            "            _ => continue,\n",
            "        }\n",
            "    }\n",
            "    Ok(tokens)\n",
            "}\n\n",
            "impl<'a, T: Clone + Send + Sync> ParallelProcessor<'a, T> {\n",
            "    pub async fn execute(&mut self, data: &'a [T]) -> Vec<ProcessedItem<T>> {\n",
            "        let chunks = data.chunks(self.chunk_size);\n",
            "        let futures: Vec<_> = chunks.map(|chunk| async move {\n",
            "            self.process_chunk(chunk).await\n",
            "        }).collect();\n",
            "        futures::future::join_all(futures).await.flatten().collect()\n",
            "    }\n",
            "}\n\n"
        };
        
        while (source.size() < target_size) {
            for (const char* snippet : code_snippets) {
                source.append(snippet);
                if (source.size() >= target_size) break;
            }
        }
        
        return source;
    }
    
    // Measure CPU baseline performance
    double measure_cpu_baseline(const std::string& source) {
        const int num_runs = 10;
        std::vector<Token> cpu_tokens(source.size());
        
        auto start = high_resolution_clock::now();
        for (int i = 0; i < num_runs; ++i) {
            cpu_tokenizer.tokenize(source.c_str(), source.size(), 
                                   cpu_tokens.data(), cpu_tokens.size());
        }
        auto end = high_resolution_clock::now();
        
        duration<double, std::milli> elapsed = end - start;
        return elapsed.count() / num_runs;
    }
    
public:
    PerformanceValidator() {
        cudaGetDeviceProperties(&device_props, 0);
    }
    
    // Validate tokenizer performance
    PerfMetrics validate_tokenizer(size_t source_size) {
        PerfMetrics metrics = {};
        metrics.source_size_bytes = source_size;
        
        // Generate test source
        std::string source = generate_test_source(source_size);
        
        // Allocate GPU memory
        char* d_source;
        Token* d_tokens;
        uint32_t* d_token_count;
        uint32_t max_tokens = source_size; // Conservative estimate
        
        cudaMalloc(&d_source, source_size);
        cudaMalloc(&d_tokens, max_tokens * sizeof(rustg::Token));
        cudaMalloc(&d_token_count, sizeof(uint32_t));
        
        // Copy source to GPU
        cudaMemcpy(d_source, source.c_str(), source_size, cudaMemcpyHostToDevice);
        cudaMemset(d_token_count, 0, sizeof(uint32_t));
        
        // Warm-up run
        launch_tokenizer_optimized(d_source, source_size, 
                                  d_tokens, d_token_count, max_tokens);
        cudaDeviceSynchronize();
        
        // Performance measurement
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        const int num_runs = 100;
        nvtxRangePush("Tokenizer Performance Test");
        
        cudaEventRecord(start);
        for (int i = 0; i < num_runs; ++i) {
            cudaMemset(d_token_count, 0, sizeof(uint32_t));
            launch_tokenizer_optimized(d_source, source_size, 
                                      d_tokens, d_token_count, max_tokens);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        nvtxRangePop();
        
        // Calculate metrics
        float gpu_time_ms;
        cudaEventElapsedTime(&gpu_time_ms, start, stop);
        metrics.kernel_time_ms = gpu_time_ms / num_runs;
        
        // Get token count
        uint32_t token_count;
        cudaMemcpy(&token_count, d_token_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        metrics.tokens_generated = token_count;
        
        // Calculate throughput
        double seconds = metrics.kernel_time_ms / 1000.0;
        metrics.throughput_gbps = (source_size / 1e9) / seconds;
        metrics.tokens_per_second = token_count / seconds;
        
        // Measure CPU baseline
        double cpu_time_ms = measure_cpu_baseline(source);
        metrics.speedup_vs_cpu = cpu_time_ms / metrics.kernel_time_ms;
        
        // Estimate memory bandwidth utilization
        size_t total_memory = source_size + (token_count * sizeof(Token));
        // Use estimated memory bandwidth since memoryClockRate is deprecated
        double theoretical_bandwidth = device_props.memoryBusWidth / 8.0 * 1000.0; // GB/s estimate
        double actual_bandwidth = (total_memory / 1e9) / seconds;
        metrics.memory_bandwidth_pct = (actual_bandwidth / theoretical_bandwidth) * 100;
        
        // Memory usage
        metrics.memory_used_bytes = source_size + max_tokens * sizeof(Token) + sizeof(uint32_t);
        
        // Cleanup
        cudaFree(d_source);
        cudaFree(d_tokens);
        cudaFree(d_token_count);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        return metrics;
    }
    
    // Validate kernel fusion performance
    PerfMetrics validate_fusion(size_t source_size) {
        PerfMetrics metrics = {};
        metrics.source_size_bytes = source_size;
        
        std::string source = generate_test_source(source_size);
        
        // Allocate GPU memory
        char* d_source;
        rustg::Token* d_tokens;
        rustg::ASTNode* d_nodes;
        uint32_t* d_token_count;
        uint32_t* d_node_count;
        uint32_t max_tokens = source_size;
        uint32_t max_nodes = source_size / 2;
        
        cudaMalloc(&d_source, source_size);
        cudaMalloc(&d_tokens, max_tokens * sizeof(rustg::Token));
        cudaMalloc(&d_nodes, max_nodes * sizeof(rustg::ASTNode));
        cudaMalloc(&d_token_count, sizeof(uint32_t));
        cudaMalloc(&d_node_count, sizeof(uint32_t));
        
        cudaMemcpy(d_source, source.c_str(), source_size, cudaMemcpyHostToDevice);
        
        // Measure fused kernel
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        const int num_runs = 50;
        
        cudaEventRecord(start);
        for (int i = 0; i < num_runs; ++i) {
            cudaMemset(d_token_count, 0, sizeof(uint32_t));
            cudaMemset(d_node_count, 0, sizeof(uint32_t));
            launch_fused_tokenizer_ast(d_source, source_size,
                                      d_tokens, d_token_count,
                                      d_nodes, d_node_count,
                                      max_tokens, max_nodes);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float gpu_time_ms;
        cudaEventElapsedTime(&gpu_time_ms, start, stop);
        metrics.kernel_time_ms = gpu_time_ms / num_runs;
        
        // Calculate throughput
        double seconds = metrics.kernel_time_ms / 1000.0;
        metrics.throughput_gbps = (source_size / 1e9) / seconds;
        
        // Cleanup
        cudaFree(d_source);
        cudaFree(d_tokens);
        cudaFree(d_nodes);
        cudaFree(d_token_count);
        cudaFree(d_node_count);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        return metrics;
    }
    
    // Print performance report
    void print_report(const PerfMetrics& metrics, const std::string& test_name) {
        std::cout << "\n========== " << test_name << " Performance Report ==========\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Source Size:        " << metrics.source_size_bytes / 1024.0 << " KB\n";
        std::cout << "Tokens Generated:   " << metrics.tokens_generated << "\n";
        std::cout << "Kernel Time:        " << metrics.kernel_time_ms << " ms\n";
        std::cout << "Throughput:         " << metrics.throughput_gbps << " GB/s\n";
        std::cout << "Tokens/Second:      " << metrics.tokens_per_second / 1e6 << " M\n";
        std::cout << "Memory Bandwidth:   " << metrics.memory_bandwidth_pct << "%\n";
        std::cout << "Speedup vs CPU:     " << metrics.speedup_vs_cpu << "x\n";
        std::cout << "Memory Used:        " << metrics.memory_used_bytes / (1024*1024) << " MB\n";
        
        // Performance verdict
        std::cout << "\nPerformance Targets:\n";
        bool throughput_met = metrics.throughput_gbps >= 1.0;
        bool speedup_met = metrics.speedup_vs_cpu >= 100.0;
        bool bandwidth_met = metrics.memory_bandwidth_pct >= 80.0;
        
        std::cout << "  [" << (throughput_met ? "✓" : "✗") 
                  << "] Throughput >= 1 GB/s (actual: " 
                  << metrics.throughput_gbps << " GB/s)\n";
        std::cout << "  [" << (speedup_met ? "✓" : "✗") 
                  << "] Speedup >= 100x (actual: " 
                  << metrics.speedup_vs_cpu << "x)\n";
        std::cout << "  [" << (bandwidth_met ? "✓" : "✗") 
                  << "] Bandwidth >= 80% (actual: " 
                  << metrics.memory_bandwidth_pct << "%)\n";
        
        if (throughput_met && speedup_met && bandwidth_met) {
            std::cout << "\n✅ All performance targets MET!\n";
        } else {
            std::cout << "\n⚠️  Some performance targets not met yet.\n";
        }
    }
};

// Main validation function
extern "C" void run_performance_validation() {
    std::cout << "\n🚀 rustg GPU Compiler - Performance Validation Suite\n";
    std::cout << "================================================\n";
    
    PerformanceValidator validator;
    
    // Test different file sizes
    size_t test_sizes[] = {
        1024 * 100,      // 100 KB
        1024 * 1024,     // 1 MB
        1024 * 1024 * 10 // 10 MB
    };
    
    for (size_t size : test_sizes) {
        std::cout << "\n📊 Testing with " << size / 1024 << " KB source file...\n";
        
        // Test tokenizer
        PerfMetrics tokenizer_metrics = validator.validate_tokenizer(size);
        validator.print_report(tokenizer_metrics, "Tokenizer");
        
        // Test fusion
        PerfMetrics fusion_metrics = validator.validate_fusion(size);
        validator.print_report(fusion_metrics, "Fused Pipeline");
        
        // Compare fusion vs separate
        double fusion_benefit = (tokenizer_metrics.kernel_time_ms / 
                                fusion_metrics.kernel_time_ms - 1) * 100;
        std::cout << "\n🔄 Kernel Fusion Benefit: " 
                  << std::fixed << std::setprecision(1) 
                  << fusion_benefit << "% faster\n";
    }
    
    std::cout << "\n✅ Performance validation complete!\n";
}

} // namespace rustg

// Test runner
int main() {
    rustg::run_performance_validation();
    return 0;
}